"""Termination, timing and admission summaries for continuation curricula."""

import math
import statistics
import time
from collections import Counter


def timed_generator(generate):
    """Measure client admission wait separately from the generation call."""

    async def wrapped(input):
        started = time.time()
        output = await generate(input)
        finished = time.time()
        samples = output.samples if isinstance(output.samples, list) else [output.samples]
        for sample in samples:
            sample.metadata = dict(sample.metadata or {})
            submitted = sample.metadata.get("bread_request_submitted_unix")
            sample.metadata["bread_timing"] = {
                "generation_started_unix": started,
                "generation_finished_unix": finished,
                "client_admission_wait_seconds": max(0.0, started - submitted) if submitted is not None else None,
                "generation_call_seconds": finished - started,
            }
        return output

    return wrapped


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": values[math.ceil(0.50 * len(values)) - 1],
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "max": values[-1],
    }


def summarize(events):
    """Separate all generated work from the groups selected for optimization."""
    metrics = {}

    def put(prefix, values):
        metrics.update({f"bread/{prefix}/{key}": value for key, value in distribution(values).items()})

    admitted = [e for e in events if e["accepted"]]
    rescued = [e for e in admitted if e["outcome"] == "rescued"]
    metrics["bread/admitted_groups"] = len(admitted)
    metrics["bread/admitted_bread_fraction"] = len(rescued) / len(admitted) if admitted else 0.0
    metrics["bread/overflow_groups"] = sum(e["eligible"] and not e["accepted"] for e in events)
    for kind in ("all_truncated", "mixed_truncation", "finished_wrong"):
        subset = [e for e in events if e.get("failure_kind") == kind]
        metrics[f"bread/{kind}/searched_groups"] = len(subset)
        metrics[f"bread/{kind}/rescued_groups"] = sum(e["outcome"] == "rescued" for e in subset)
        metrics[f"bread/{kind}/admitted_groups"] = sum(e["accepted"] for e in subset)
        if subset:
            metrics[f"bread/{kind}/rescue_rate"] = sum(e["outcome"] == "rescued" for e in subset) / len(subset)
    for kind in ("bread", "ordinary"):
        subset = [e for e in events if bool(e["attempts"]) == (kind == "bread")]
        for field in ("elapsed_seconds", "initial_seconds", "search_seconds"):
            put(f"timing/{kind}/{field}", [e[field] for e in subset])
        for field in ("completion_to_batch_seconds", "arrival_to_batch_seconds"):
            put(f"timing/{kind}/{field}", [e[field] for e in subset if e["accepted"]])
    stages = {
        "initial": [r for e in events for r in e["initial_responses"]],
        "probe": [r for e in events for a in e["attempts"] for r in a["responses"]],
        "admitted": [r for e in admitted for r in e["selected_responses"]],
    }
    stages["generated"] = stages["initial"] + stages["probe"]
    for stage, rows in stages.items():
        counts = Counter(r["status"] for r in rows)
        metrics[f"bread/{stage}/responses"] = len(rows)
        metrics[f"bread/{stage}/generated_tokens"] = sum(r["generated_tokens"] for r in rows)
        for status in ("completed", "truncated"):
            metrics[f"bread/{stage}/{status}"] = counts[status]
        if rows:
            metrics[f"bread/{stage}/truncated_fraction"] = counts["truncated"] / len(rows)
            metrics[f"bread/{stage}/accuracy"] = sum(r["reward"] for r in rows) / len(rows)
        finished = [r for r in rows if r["status"] == "completed"]
        if finished:
            metrics[f"bread/{stage}/accuracy_finished"] = sum(r["reward"] for r in finished) / len(finished)
        for field in ("client_admission_wait_seconds", "generation_call_seconds"):
            put(
                f"timing/{stage}/{field}",
                [r["timing"][field] for r in rows if r.get("timing", {}).get(field) is not None],
            )
    for category in ("finished_correct", "finished_wrong", "truncated"):
        advantages = []
        for event in admitted:
            rows = event["selected_responses"]
            mean = statistics.mean(r["reward"] for r in rows)
            for row in rows:
                kind = (
                    "truncated"
                    if row["status"] == "truncated"
                    else "finished_correct"
                    if row["reward"]
                    else "finished_wrong"
                )
                if kind == category:
                    advantages.append(row["reward"] - mean)
        put(f"admitted/centered_reward/{category}", advantages)
    return metrics
