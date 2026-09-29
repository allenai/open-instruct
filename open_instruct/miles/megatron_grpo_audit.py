"""Check that registered verifier rewards, centered GRPO advantages and weights reached training."""

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from open_instruct.miles import opd_audit, rewards, workflow


def audit(spec, prepared):
    root = Path(spec.output["root"])
    count = spec.document["training"]["num_rollouts"]
    steps = opd_audit.audit_optimizer(root, count)
    batch = spec.document["inference"]["rollout_batch_size"] * spec.document["inference"]["samples_per_prompt"]
    fanout = spec.document["inference"]["samples_per_prompt"]
    records = [json.loads(line) for line in (root / "verifier-rewards.jsonl").read_text().splitlines()]
    if len(records) != count * batch:
        raise ValueError("Verifier evidence does not cover every training response")

    async def rescore():
        for record in records:
            value = await rewards.score(
                SimpleNamespace(**record, prompt=record["metadata"].get("query", "")),
                prepared["data"]["reward_config"],
            )
            if value != record["reward"]:
                raise ValueError("Independent registered verifier rescore differs")

    asyncio.run(rescore())
    rollouts = []
    for rollout_id in range(count):
        rows = records[rollout_id * batch : (rollout_id + 1) * batch]
        expected = {}
        mixed = 0
        for offset in range(0, batch, fanout):
            group = rows[offset : offset + fanout]
            values = [row["reward"] for row in group]
            mean = sum(values) / fanout
            mixed += int(min(values) != max(values))
            denominator = 1.0
            if spec.document["objective"]["std_normalization"]:
                denominator = math.sqrt(sum((value - mean) ** 2 for value in values) / (fanout - 1)) + 1e-6
            for row in group:
                expected[row["sample_index"]] = (row["reward"] - mean) / denominator
        if mixed == 0:
            raise ValueError(f"Rollout {rollout_id} lacks a mixed verifier-reward group")
        seen = set()
        for path in sorted((root / "debug/train_data").glob(f"{rollout_id}_*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=False)["rollout_data"]
            if data.get("teacher_log_probs") is not None:
                raise ValueError("Unexpected teacher probabilities in pure GRPO training")
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
                "reward_sum": sum(row["reward"] for row in rows),
                "advantage_centering_verified": True,
            }
        )
    export = opd_audit.complete_export(root, Path(prepared["model"]), count - 1)
    result = {
        "passed": True,
        "algorithm": "grpo",
        "teacher": None,
        "optimizer": steps,
        "rollouts": rollouts,
        "export": export,
        "limits": "Mechanics audit; no learned-quality or trainer/inference numerical-equivalence claim.",
    }
    workflow.write_json(root / "audit.json", result)
    return result
