"""Read-only CPU audit of completed rolling-publication runs and native masters."""

import argparse
import hashlib
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

from open_instruct.miles import policy_versions


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
            versions = set(policy_versions.versions(sample["weight_versions"]))
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
            if (
                len(group) != miles["n_samples_per_prompt"]
                or len({tuple(sorted(set(policy_versions.versions(s["weight_versions"])))) for s in group}) != 1
            ):
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
            shape = (manifest["hf_config"]["n_routed_experts"], manifest["hf_config"]["hidden_size"])
            # Core's canonical converter flattens router matrices on import and
            # restores exactly [n_experts, d_model] on export (no transpose).
            if tuple(actual.shape) != (math.prod(shape),):
                raise ValueError(f"Unexpected native flat router layout for {name}: {actual.shape}")
            actual = actual.reshape(shape)
            if tuple(expected.shape) != shape:
                raise ValueError(
                    f"Router shape mismatch for {hf_name}: initial={expected.shape}, native={actual.shape}"
                )
            if actual.dtype != torch.float32 or not torch.isfinite(actual).all():
                raise ValueError("Expected finite FP32 native router masters")
            # models.build explicitly loads this HF model with BF16 dtype before
            # copying into native FP32 routers/masters. The saved HF tensor can
            # itself be FP32; compare against the actual import, not raw storage.
            imported = expected.bfloat16().float()
            if not torch.isfinite(imported).all():
                raise ValueError(f"Nonfinite initial router: {hf_name}")
            difference = actual - imported
            changes[hf_name] = {
                "checkpoint_dtype": str(expected.dtype),
                "import_dtype": "torch.bfloat16",
                "changed_elements": int(torch.count_nonzero(difference)),
                "max_abs_change": float(difference.abs().max()),
            }
    if not changes or not any(row["changed_elements"] for row in changes.values()):
        raise ValueError("No measured router master change from the initial checkpoint")
    resumed = None
    if miles.get("load"):
        # Audit the exact boundary immediately preceding this run, not a mutable
        # latest pointer. Changes from the original HF model alone could conceal
        # an inert resumed optimizer whose first six updates were already saved.
        previous_step = clocks[0][0] - 1
        source_root = Path(miles["load"])
        source = source_root / "core" / f"rollout_{previous_step - 1:07d}"
        source_manifest = json.loads((source / "complete.json").read_text())
        cursor = source_root / "rollout" / f"global_dataset_state_dict_{previous_step - 1}.pt"
        if (
            source_manifest["clock"]["completed_steps"] != previous_step
            or hashlib.sha256(cursor.read_bytes()).hexdigest() != source_manifest["cursor_sha256"]
            or clocks[0] != list(range(previous_step + 1, miles["num_rollout"] + 1))
        ):
            raise ValueError("Resume boundary, committed cursor or consecutive optimizer clocks disagree")
        previous = CoreCheckpointState(
            source / "model", Olmo3MoeConfig(**source_manifest["hf_config"]), category="fp32_masters"
        )
        drift = {}
        for name in native:
            if not name.endswith(".routed_experts_router.weight"):
                continue
            before, after = previous[name], native[name]
            if before.shape != after.shape or before.dtype != torch.float32 or not torch.isfinite(before).all():
                raise ValueError("Invalid resumed router master reference")
            difference = after - before
            drift[name] = {
                "changed_elements": int(torch.count_nonzero(difference)),
                "max_abs_change": float(difference.abs().max()),
            }
        if not drift or not any(row["changed_elements"] for row in drift.values()):
            raise ValueError("No router master changed after resuming")
        resumed = {"checkpoint": str(source), "completed_steps": previous_step, "router_master_changes": drift}
    report = {
        "run_root": str(root),
        "completed_steps": clocks[0],
        "resumed_from": resumed,
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
