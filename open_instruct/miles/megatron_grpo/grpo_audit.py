"""Check that registered verifier rewards, centered GRPO advantages and weights reached training."""

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from open_instruct.miles.distillation import opd_audit
from open_instruct.miles.execution import workflow
from open_instruct.miles.megatron_grpo import grpo_args, grpo_hooks
from open_instruct.miles.rewards import rewards


def expected_advantages(rows, fanout, std_normalization):
    """``({sample_index: centered reward}, mixed group count)`` for one rollout's evidence rows."""
    expected, mixed = {}, 0
    for offset in range(0, len(rows), fanout):
        group = rows[offset : offset + fanout]
        values = [row["reward"] for row in group]
        mean = sum(values) / fanout
        mixed += int(min(values) != max(values))
        denominator = 1.0
        if std_normalization:
            denominator = math.sqrt(sum((value - mean) ** 2 for value in values) / (fanout - 1)) + 1e-6
        for row in group:
            expected[row["sample_index"]] = (row["reward"] - mean) / denominator
    return expected, mixed


def audit(spec, prepared):
    """Audit a finished run's mechanics and write ``audit.json``.

    Assumes ``grpo_hooks.reconcile_evidence`` ran before this launch's training, so the evidence
    file holds exactly one record per trained response.
    """
    root = Path(spec.output["root"])
    count = spec.document["training"]["num_rollouts"]
    # Zero reward variation in every group is a legitimate zero-gradient batch, not a failure.
    steps = opd_audit.audit_optimizer(root, count, require_positive_grad_norm=False)
    inference = spec.document["inference"]
    batch = grpo_args.responses_per_rollout(spec)
    fanout = inference["samples_per_prompt"]
    records = [json.loads(line) for line in (root / grpo_hooks.EVIDENCE).read_text().splitlines()]
    if len(records) != count * batch:
        raise ValueError("Verifier evidence does not cover every training response")
    scoring = grpo_hooks.scoring_args(
        environment=grpo_hooks.reward_environment(prepared["data"]["reward_config"], spec.document["reward"])
    )

    async def rescore():
        for record in records:
            value = await rewards.registered_reward(
                scoring, SimpleNamespace(**record, prompt=record["metadata"].get("query", ""))
            )
            if value != record["reward"]:
                raise ValueError("Independent registered verifier rescore differs")

    asyncio.run(rescore())
    rollouts = []
    for rollout_id in range(count):
        rows = records[rollout_id * batch : (rollout_id + 1) * batch]
        expected, mixed = expected_advantages(rows, fanout, spec.document["objective"]["std_normalization"])
        seen = set()
        for path in sorted((root / "debug/train_data").glob(f"{rollout_id}_*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=False)["rollout_data"]
            if data.get("teacher_log_probs") is not None:
                raise ValueError("Unexpected teacher probabilities in verifier GRPO training")
            for index, advantage in zip(data["sample_indices"], data["advantages"], strict=True):
                tensor = torch.as_tensor(advantage)
                if not tensor.numel() or not torch.isfinite(tensor).all():
                    raise ValueError("Missing or nonfinite response advantages")
                torch.testing.assert_close(
                    tensor.float(), torch.full_like(tensor.float(), expected[int(index)]), rtol=1e-6, atol=1e-5
                )
                seen.add(int(index))
        if seen != set(expected):
            raise ValueError("Trainer dumps do not cover every verifier-scored sample")
        rollouts.append(
            {
                "rollout_id": rollout_id,
                "responses": batch,
                "mixed_groups": mixed,
                "zero_signal": mixed == 0,
                "reward_sum": sum(row["reward"] for row in rows),
                "advantage_centering_verified": True,
            }
        )
    export = opd_audit.complete_export(root, Path(prepared["model"]), count - 1)
    replayed, torn = root / grpo_hooks.REPLAYED, root / grpo_hooks.TORN
    result = {
        "passed": True,
        "algorithm": "grpo",
        "teacher": None,
        "optimizer": steps,
        "rollouts": rollouts,
        # Records grpo_hooks.reconcile_evidence set aside, cumulative over every launch into this root
        # (an upper bound if a launch died mid-reconcile).
        "replayed_records": len(replayed.read_text().splitlines()) if replayed.exists() else 0,
        # Partial final lines of launches that stopped mid-write.
        "torn_fragments": len(torn.read_text().splitlines()) if torn.exists() else 0,
        "export": export,
        "limits": "Mechanics audit; no learned-quality or trainer/inference numerical-equivalence claim.",
    }
    workflow.write_json(root / "audit.json", result)
    return result
