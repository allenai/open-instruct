"""Bounded ToolRL experiment guard, with unchanged truncation/advantage semantics.

Allow four optimizer updates, then abort before update five if none of their
finished tool-call responses earned positive reward. No-call format rewards
and truncated responses cannot satisfy the guard. Persist every scored response
for independent audit; this hook runs on training batches, not held-out evals.
"""

import json
from collections import defaultdict
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
        "mean_abs_advantage": sum(abs(x) for x in normalized) / len(normalized) if normalized else None,
    }
    directory = Path(args.save).parent
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "toolrl-reward-watch.jsonl").open("a") as stream:
        stream.write(json.dumps(summary, allow_nan=False) + "\n")
    samples_directory = directory / "toolrl-reward-samples"
    samples_directory.mkdir(exist_ok=True)
    with (samples_directory / f"batch_{batches:05d}.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps({"batch": batches, **record}, allow_nan=False) + "\n")
    logger.info("TOOLRL_REWARD_WATCH %s", json.dumps(summary, allow_nan=False))
    args._toolrl_reward_watch = (batches + 1, first_four_positive)
    return raw, normalized


def log_eval(rollout_id, args, data, extra_metrics):
    """Persist common strict-success and partial-credit metrics for either reward arm.

    Never compare the arms' native mean rewards: they optimize different scores.
    Include truncated/failed generations in denominators as unsuccessful trials.
    Keep each response and both scores so initial-zero-success subsets can be
    selected once and followed on independent later generations.
    """
    directory = Path(args.save).parent / "toolrl-heldout"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    buckets = defaultdict(list)
    for dataset, item in data.items():
        samples = item.get("samples")
        if samples is None:
            raise ValueError("ToolRL held-out audit requires evaluation samples")
        for sample in samples:
            metadata = sample.metadata or {}
            target = next(v["target"] for v in metadata["verifiers"] if v["name"] in {"toolrl", "toolrl_think_prefix"})
            diagnostics = next(iter(metadata.get("verifier_diagnostics", {}).values()), {})
            finished = sample.status == Sample.Status.COMPLETED
            count = len(target["calls"])
            record = {
                "dataset": dataset,
                "sample_id": metadata.get("prepared_sample_id"),
                "group_index": sample.group_index,
                "required_calls": count,
                "status": sample.status.value,
                "full_success": finished and diagnostics.get("full_success", False),
                "partial_score": diagnostics.get("partial_score"),
                "binary_score": diagnostics.get("binary_score"),
                "response": sample.response,
                "target": target,
            }
            records.append(record)
            bucket = (
                "no_call" if count == 0 else "one_call" if count == 1 else "two_calls" if count == 2 else "three_plus"
            )
            buckets[bucket].append(record)
    summary = {
        name: {
            "responses": len(rows),
            "full_successes": sum(r["full_success"] for r in rows),
            "full_success_rate": sum(r["full_success"] for r in rows) / len(rows) if name != "no_call" else None,
        }
        for name, rows in buckets.items()
    }
    (directory / f"step_{rollout_id:05d}.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
    )
    with (directory / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps({"rollout_id": rollout_id, "buckets": summary}, allow_nan=False) + "\n")
    logger.info("TOOLRL_HELDOUT %s", json.dumps({"rollout_id": rollout_id, "buckets": summary}))
    return False
