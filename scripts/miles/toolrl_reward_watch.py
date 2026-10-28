"""Bounded ToolRL experiment guard, with unchanged truncation/advantage semantics.

Allow four optimizer updates, then abort before update five if none of their
finished tool-call responses earned positive reward. No-call format rewards
and truncated responses cannot satisfy the guard. Persist every scored response
for independent audit; this hook runs on training batches, not held-out evals.
"""

import json
from pathlib import Path

from miles.utils.types import Sample

from open_instruct import logger_utils
from open_instruct.miles.rewards import truncation

logger = logger_utils.setup_logger(__name__)


def post_process(args, samples):
    batches, first_four_positive = getattr(args, "_toolrl_reward_watch", (0, 0))
    if batches == 4 and first_four_positive == 0:
        logger.error("TOOLRL_REWARD_WATCH abort: no positive finished tool-call rewards in the first four updates")
        raise RuntimeError("ToolRL reward watch aborted before update five: no positive rewards in the first four")
    raw, normalized = truncation.exclude_truncated(args, samples)
    records = []
    for sample, score in zip(samples, raw, strict=True):
        metadata = sample.metadata or {}
        verifiers = metadata.get("verifiers", [])
        targets = [v["target"] for v in verifiers if v.get("name") in {"toolrl", "toolrl_think_prefix"}]
        if len(targets) != 1:
            raise ValueError("ToolRL reward watch requires exactly one ToolRL target per sample")
        records.append(
            {
                "sample_id": metadata.get("prepared_sample_id"),
                "group_index": sample.group_index,
                "response": sample.response,
                "target": targets[0],
                "reward": score,
                "status": sample.status.value,
                "eligible_call": bool(targets[0]["calls"]) and sample.status == Sample.Status.COMPLETED,
                "diagnostics": metadata.get("verifier_diagnostics", {}),
            }
        )
    positive = sum(r["eligible_call"] and r["reward"] > 0 for r in records)
    if batches < 4:
        first_four_positive += positive
    summary = {
        "batch": batches,
        "samples": len(records),
        "finished_call_responses": sum(r["eligible_call"] for r in records),
        "positive_call_responses": positive,
        "full_credit_call_responses": sum(r["eligible_call"] and r["reward"] == 4 for r in records),
        "mean_raw_reward": sum(raw) / len(raw) if raw else None,
        "first_four_positive_call_responses": first_four_positive,
        "guard_passed": first_four_positive > 0 if batches >= 3 else None,
    }
    directory = Path(args.save).parent
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "toolrl-reward-watch.jsonl").open("a") as stream:
        stream.write(json.dumps(summary, allow_nan=False) + "\n")
    with (directory / "toolrl-reward-samples.jsonl").open("a") as stream:
        for record in records:
            stream.write(json.dumps({"batch": batches, **record}, allow_nan=False) + "\n")
    logger.info("TOOLRL_REWARD_WATCH %s", json.dumps(summary, allow_nan=False))
    args._toolrl_reward_watch = (batches + 1, first_four_positive)
    return raw, normalized
