"""Read-only audit of retained refresh training, replay, publication and resume."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from open_instruct.miles.publication import policy_refresh, policy_versions
from open_instruct.miles.training import replay_diagnostics


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root, *, updates=5, mode="refresh", require_mixed=True):
    root = Path(root)
    reports, seen, total_mixed, mixed_reward_groups = [], set(), 0, 0
    for update in range(updates):
        payload = torch.load(root / f"rollouts/{update}.pt", map_location="cpu", weights_only=False)
        samples = payload["samples"]
        if len(samples) != 16:
            raise ValueError("Qualification requires 16 responses per optimizer update")
        groups = defaultdict(list)
        lengths, historical, tokens, mixed = [], 0, 0, 0
        for sample in samples:
            length = sample["response_length"]
            probs = sample["rollout_log_probs"]
            if length <= 0 or len(probs) != length or not all(math.isfinite(p) for p in probs):
                raise ValueError("Missing, misaligned or nonfinite original rollout probabilities")
            metadata = sample.get("train_metadata")
            if mode == "barrier":
                versions = sorted(set(policy_versions.versions(sample["weight_versions"])))
                if len(versions) != 1:
                    raise ValueError("Barrier baseline unexpectedly contains mixed-version responses")
                version = policy_refresh.version_number(versions[0])
                metadata = {
                    "policy_refresh": {
                        "replay_version": version,
                        "spans": [dict(version=version, start=0, end=length)],
                    }
                }
            spans = policy_refresh.validate_batch(
                dict(metadata=[metadata], response_lengths=[length], weight_versions=[sample["weight_versions"]])
            )[0]
            if any(not 0 <= update - span["version"] <= 2 for span in spans):
                raise ValueError("Training consumed a token outside its optimizer-step lag budget")
            if mode == "refresh" and metadata["policy_refresh"] != sample["metadata"]["policy_refresh"]:
                raise ValueError("Training and retained provenance differ")
            mask = sample.get("loss_mask")
            if mask is not None and (len(mask) != length or any(x != 1 for x in mask)):
                raise ValueError("Refresh qualification unexpectedly masks retained response tokens")
            routes = np.asarray(sample["rollout_routed_experts"])
            if (
                routes.ndim != 3
                or routes.shape[0] != len(sample["tokens"]) - 1
                or not np.issubdtype(routes.dtype, np.integer)
            ):
                raise ValueError("Latest replay routes do not align with the complete forwarded prefix")
            group = sample["group_index"]
            if group in seen:
                raise ValueError("Consumed the same prompt group twice, including across resume")
            groups[group].append(sample)
            mixed += len(spans) > 1
            tokens += length
            lengths.append(length)
            historical += sum(
                s["end"] - s["start"] for s in spans if s["version"] < metadata["policy_refresh"]["replay_version"]
            )
        if len(groups) != 4 or any(len(group) != 4 for group in groups.values()):
            raise ValueError("Incomplete GRPO prompt group")
        mixed_reward_groups += sum(len({float(s["reward"]) for s in group}) > 1 for group in groups.values())
        seen.update(groups)
        total_mixed += mixed
        reports.append(
            dict(
                update=update,
                mixed_responses=mixed,
                tokens=tokens,
                historical_prefix_tokens=historical,
                median_response_tokens=float(np.median(lengths)),
            )
        )
    if mode == "refresh" and require_mixed and not total_mixed:
        raise ValueError("No refreshed response reached an optimizer step; continuation is not yet exercised")
    if not mixed_reward_groups:
        raise ValueError("No nonzero group-relative policy advantage was exercised")
    contracts = {str(rank): records(root / f"metrics/training_contract_rank{rank}.jsonl") for rank in range(2)}
    for rank, rows in contracts.items():
        steps = [r for r in rows if r["event"] == "optimizer"]
        if [r["step"] for r in steps] != list(range(1, updates + 1)) or any(r["optimizer_skipped"] for r in steps):
            raise ValueError(f"Rank {rank} skipped or repeated an optimizer step")
        for row in steps:
            if any(not 0 <= row["step"] - 1 - v <= 2 for v in row["local_behavior_versions"]):
                raise ValueError("Trainer clock disagrees with behavior lag")
    replay = replay_diagnostics.audit_contracts(contracts, updates, local_samples=8)
    for update in (1, 3, updates - 1):
        manifest = json.loads((root / f"metrics/core/rollout_{update:07d}/complete.json").read_text())
        if manifest["clock"]["completed_steps"] != update + 1:
            raise ValueError("Checkpoint manifest disagrees with the optimizer boundary")
    initial = json.loads((root / "initial-result.json").read_text())
    resumed = json.loads((root / "resume-result.json").read_text())
    if initial["completed_rollout_ids"] != [0, 1, 2, 3] or resumed["completed_rollout_ids"] != [4]:
        raise ValueError("Fresh-process resume did not consume exactly the next update")
    timing = records(root / "metrics/driver_timing.jsonl")
    if any(not r["passed"] for r in timing):
        raise ValueError("A driver lifecycle stage failed")
    return dict(
        passed=True,
        mode=mode,
        updates=updates,
        consumed_mixed_responses=total_mixed,
        groups_with_policy_advantage=mixed_reward_groups,
        rollouts=reports,
        replay=replay,
        initial_wall_seconds=initial["wall_seconds"],
        resume_wall_seconds=resumed["wall_seconds"],
        timings=timing,
        publications=records(root / "metrics/publication.jsonl"),
        scope="Actual optimizer/replay/retained-prefix/resume sanity check, not an RL quality comparison",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("refresh", "barrier"), default="refresh")
    opt = parser.parse_args()
    result = audit(opt.root, mode=opt.mode)
    opt.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key not in ("timings", "publications")}, indent=2)
    )


if __name__ == "__main__":
    main()
