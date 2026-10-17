"""Check that registered verifier rewards, centered GRPO advantages and weights reached training."""

import asyncio
import collections
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from open_instruct.miles.distillation import opd_audit
from open_instruct.miles.execution import workflow
from open_instruct.miles.megatron_grpo import grpo_hooks
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


# Rescoring every record serially against the remote code service takes hours on a long run, and
# the registered verifiers are not deterministic: an IFEval instruction stored without kwargs draws
# random parameters, language detection is unseeded, and the code service's per-test time limit
# passes or fails a borderline program depending on load. A wiring fault (wrong target, verifier or gate) disagrees on
# far more than this fraction of a verifier's records; the audit reports every disagreement it accepts.
RESCORE_SAMPLE = 4096
RESCORE_CONCURRENCY = 32
RESCORE_TOLERANCE = 0.01


def rescore(scoring, records):
    """Independently rescore a fixed sample of records and bound the disagreement per verifier."""
    chosen = sorted(random.Random(0).sample(range(len(records)), min(len(records), RESCORE_SAMPLE)))

    async def score_all():
        gate = asyncio.Semaphore(RESCORE_CONCURRENCY)

        async def score(index):
            record = records[index]
            metadata = {key: value for key, value in record["metadata"].items() if key != "verifier_diagnostics"}
            async with gate:
                return await rewards.registered_reward(
                    scoring, SimpleNamespace(**record | {"metadata": metadata}, prompt=metadata.get("query", ""))
                )

        return await asyncio.gather(*(score(index) for index in chosen))

    checked, mismatches = collections.Counter(), collections.defaultdict(list)
    for index, value in zip(chosen, asyncio.run(score_all()), strict=True):
        record = records[index]
        name = "+".join(sorted(verifier["name"] for verifier in record["metadata"].get("verifiers", []))) or "unnamed"
        checked[name] += 1
        if value != record["reward"]:
            mismatches[name].append({"index": index, "trained": record["reward"], "rescored": value})
    for name, rows in mismatches.items():
        if len(rows) > RESCORE_TOLERANCE * checked[name]:
            raise ValueError(
                f"Independent registered verifier rescore differs for {len(rows)} of {checked[name]} {name} records"
            )
    return {"tolerance": RESCORE_TOLERANCE, "checked": dict(checked), "mismatches": dict(mismatches)}


def audit(spec, prepared):
    root = Path(spec.output["root"])
    count = spec.document["training"]["num_rollouts"]
    # Zero reward variation in every group is a legitimate zero-gradient batch, not a failure.
    steps = opd_audit.audit_optimizer(root, count, require_positive_grad_norm=False)
    inference = spec.document["inference"]
    batch = inference["rollout_batch_size"] * inference["samples_per_prompt"]
    fanout = inference["samples_per_prompt"]
    records = [json.loads(line) for line in (root / grpo_hooks.EVIDENCE).read_text().splitlines()]
    if len(records) != count * batch:
        raise ValueError("Verifier evidence does not cover every training response")
    scoring = grpo_hooks.scoring_args(
        environment=grpo_hooks.reward_environment(prepared["data"]["reward_config"], spec.document["reward"])
    )
    rescored = rescore(scoring, records)
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
    replayed = root / grpo_hooks.REPLAYED
    result = {
        "passed": True,
        "algorithm": "grpo",
        "teacher": None,
        "optimizer": steps,
        "rollouts": rollouts,
        # Records of rollouts a resumed trainer replayed; grpo_hooks.reconcile_evidence set them aside.
        "replayed_records": len(replayed.read_text().splitlines()) if replayed.exists() else 0,
        "export": export,
        "rescore": rescored,
        "limits": "Mechanics audit; no learned-quality or trainer/inference numerical-equivalence claim.",
    }
    workflow.write_json(root / "audit.json", result)
    return result
