"""Read-only CPU audit of completed rolling-publication runs and native masters."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from scripts.miles import analyze_engine_drain, audit_workflow
from scripts.miles.checkpoint_weights import SafeTensorState
from scripts.miles.core_checkpoint_stream import CoreCheckpointState


def audit(root):
    root = Path(root)
    plan = json.loads((root / "resolved-plan.json").read_text())
    core, miles = plan["core"], plan["miles"]
    checkpoints = root / "checkpoints"
    contracts = {
        p.stem: analyze_engine_drain.records(p) for p in sorted(checkpoints.glob("training_contract_rank*.jsonl"))
    }
    if len(contracts) != miles["actor_num_nodes"] * miles["actor_num_gpus_per_node"]:
        raise ValueError("Missing trainer-rank contract file")
    ranks = [[r for r in rows if r["event"] == "optimizer"] for rows in contracts.values()]
    clocks = [[r["step"] for r in rows] for rows in ranks]
    if not clocks[0] or any(steps != clocks[0] for steps in clocks) or clocks[0][-1] != miles["num_rollout"]:
        raise ValueError("Trainer ranks did not complete the requested optimizer steps")
    lag = Counter()
    for rows in ranks:
        for row in rows:
            if row["optimizer_skipped"]:
                raise ValueError("Optimizer step was skipped")
            for version in row["local_behavior_versions"]:
                age = row["step"] - 1 - version
                if not 0 <= age <= core["max_policy_lag"]:
                    raise ValueError("Consumed policy outside the lag budget")
                lag[age] += 1
    responses, tokens, active_tokens, mixed_reward_groups = 0, 0, 0, 0
    rewards = []
    for step in clocks[0]:
        samples = audit_workflow.load_rollout(root / "rollouts" / f"{step - 1}.pt")["samples"]
        if len(samples) != miles["global_batch_size"]:
            raise ValueError("Retained rollout does not contain one complete optimizer batch")
        groups = defaultdict(list)
        for sample in samples:
            versions = {int(v) for v in sample["weight_versions"]}
            if len(versions) != 1 or not 0 <= step - 1 - next(iter(versions)) <= core["max_policy_lag"]:
                raise ValueError("Retained sample behavior version is invalid")
            length = sample["response_length"]
            probs = sample["rollout_log_probs"]
            if length <= 0 or len(probs) != length or not all(math.isfinite(p) for p in probs):
                raise ValueError("Missing or nonfinite behavior log probabilities")
            routes = np.asarray(sample["rollout_routed_experts"])
            if (
                routes.ndim != 3
                or routes.shape[0] != len(sample["tokens"]) - 1
                or not np.issubdtype(routes.dtype, np.integer)
            ):
                raise ValueError("Replay rows do not cover the retained tokens")
            ownership = (sample.get("metadata") or {}).get("engine_drain")
            if core["publication_mode"] == "engine_drain" and (
                ownership is None or ownership["version"] != next(iter(versions))
            ):
                raise ValueError("Actual engine ownership was not retained")
            groups[sample["group_index"]].append(sample)
            responses += 1
            tokens += length
            mask = sample.get("loss_mask")
            if mask is not None and (len(mask) != length or any(value not in (0, 1) for value in mask)):
                raise ValueError("Invalid retained response loss mask")
            if not sample.get("remove_sample_loss", False):
                active_tokens += length if mask is None else sum(mask)
            rewards.append(float(sample["reward"]))
        for group in groups.values():
            if len(group) != miles["n_samples_per_prompt"] or len({tuple(s["weight_versions"]) for s in group}) != 1:
                raise ValueError("Incomplete or mixed-policy GRPO group")
            mixed_reward_groups += len({float(s["reward"]) for s in group}) > 1
    if not mixed_reward_groups:
        raise ValueError("No retained prompt group has nonzero GRPO advantages")
    # Use the native factory's exact saved layout, then compare only router
    # masters against their named BF16 source tensors. No off-grid assumption.
    path = checkpoints / "core" / f"rollout_{clocks[0][-1] - 1:07d}"
    manifest = json.loads((path / "complete.json").read_text())
    if manifest["clock"]["completed_steps"] != clocks[0][-1]:
        raise ValueError("Checkpoint boundary differs from completed training")
    native = CoreCheckpointState(path / "model", Olmo3MoeConfig(**manifest["hf_config"]), category="fp32_masters")
    changes = {}
    with SafeTensorState(miles["hf_checkpoint"]) as initial:
        for name in native:
            if not name.endswith(".routed_experts_router.weight"):
                continue
            layer = int(name.split(".")[1])
            hf_name = f"model.layers.{layer}.mlp.router.gate.weight"
            expected, actual = initial[hf_name], native[name]
            if expected.dtype != torch.bfloat16 or expected.shape != actual.shape:
                raise ValueError("Qualification requires an exact-shaped BF16 initial router checkpoint")
            if actual.dtype != torch.float32 or not torch.isfinite(actual).all():
                raise ValueError("Expected finite FP32 native router masters")
            difference = actual - expected.float()
            changes[hf_name] = {
                "changed_elements": int(torch.count_nonzero(difference)),
                "max_abs_change": float(difference.abs().max()),
            }
    if not changes or not any(row["changed_elements"] for row in changes.values()):
        raise ValueError("No measured router master change from the initial checkpoint")
    report = {
        "run_root": str(root),
        "completed_steps": clocks[0],
        "trainer_ranks": len(ranks),
        "consumed_lag_rank_version_counts": dict(lag),
        "trained_responses": responses,
        "response_tokens": tokens,
        "active_response_tokens": active_tokens,
        "mixed_reward_groups": mixed_reward_groups,
        "mean_training_reward": sum(rewards) / len(rewards),
        "router_fp32_master_changes": changes,
        "master_check_scope": "router tensors only; proves an update from initial HF weights, not backend equivalence",
        "driver_stages": analyze_engine_drain.records(checkpoints / "driver_timing.jsonl"),
    }
    if core["publication_mode"] == "engine_drain":
        protocol = analyze_engine_drain.analyze(
            analyze_engine_drain.records(checkpoints / "engine_drain.jsonl"), report["driver_stages"]
        )
        if (
            protocol["ownership_errors"]
            or protocol["mixed_groups"]
            or protocol["unreleased_snapshots"]
            or protocol["requests_without_terminal_decode"]
            or any(
                protocol["event_counts"].get(name, 0)
                for name in ("engine_unavailable", "request_outcome_unknown", "request_failed")
            )
        ):
            raise ValueError("Protocol timeline failed ownership or retention checks")
        report["protocol"] = protocol
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps([audit(root) for root in args.roots], indent=2) + "\n")


if __name__ == "__main__":
    main()
