"""Explain measured throughput from a completed run's retained artifacts.
The analyzer checks optimizer and rollout accounting, separates warmup from the
measurement window, and reports time spent waiting, training and publishing along
with discarded or unused generation work. This helps compare configurations using
useful completed work rather than a generation-rate number alone.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(root, *, warmup=3, allow_incomplete_workflow=False):
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    root = Path(root)
    metrics = root / "checkpoints"
    stages = rows(metrics / "driver_timing.jsonl")
    flow = rows(metrics / "rollout_flow.jsonl")
    if any(not r["passed"] for r in stages):
        raise ValueError("Failed driver stage; do not report a successful throughput run")
    options = json.loads((root / "plan.json").read_text())["runtime"]["miles"]
    expected = options["num_rollout"]
    if [r["rollout_id"] for r in flow] != list(range(expected)):
        raise ValueError("Missing or repeated consumed collections")
    ranks = options.get("actor_num_nodes", 1) * options.get("actor_num_gpus_per_node", 1)
    contracts = []
    for rank in range(ranks):
        path = metrics / f"training_contract_rank{rank}.jsonl"
        if not path.exists():
            raise ValueError(f"Missing trainer contract for rank {rank}")
        records = rows(path)
        updates = [r for r in records if r["event"] == "optimizer"]
        if [r["step"] for r in updates] != list(range(1, expected + 1)) or any(
            r["optimizer_skipped"] for r in updates
        ):
            raise ValueError(f"Missing or skipped optimizer updates on rank {rank}")
        contracts.append(records)
    contract = contracts[0]
    chosen = [r for r in flow if r["rollout_id"] >= warmup]
    durations = {
        name: [
            r["seconds"]
            for r in stages
            if r["stage"] == name and r["rollout_id"] is not None and r["rollout_id"] >= warmup
        ]
        for name in ("generation_wait", "training", "publication")
    }
    if not chosen or any(len(values) != len(chosen) for values in durations.values()):
        raise ValueError("Incomplete warm window")
    seconds = sum(sum(v) for v in durations.values())
    prefix = "rollout/fully_async/completed_queue/"
    if options.get("fully_async", False):
        for row in flow:
            if not all(
                prefix + key in row["queue_metrics"]
                for key in ("dropped_response_tokens", "delivered_response_tokens")
            ):
                raise ValueError("Missing async queue counters; absence is not a zero discard rate")
            if row["queue_metrics"][prefix + "delivered_response_tokens"] != row["response_tokens"]:
                raise ValueError(
                    f"Queue delivery accounting differs from consumed tokens at rollout {row['rollout_id']}"
                )
    if seconds <= 0 or any(not math.isfinite(v) or v < 0 for values in durations.values() for v in values):
        raise ValueError("Invalid measured duration")
    queue_waits = [r["queue_metrics"].get(prefix + "consumer_wait_seconds") for r in chosen]
    wait_breakdown = None
    if all(value is not None for value in queue_waits):
        if any(
            not math.isfinite(value) or value < 0 or value > outer + 0.001
            for value, outer in zip(queue_waits, durations["generation_wait"], strict=True)
        ):
            raise ValueError("Completed-queue wait is invalid or exceeds its enclosing collection stage")
        queue_seconds = sum(queue_waits)
        wait_breakdown = {
            "scope": "Completed-buffer get time includes expiry filtering; the remainder includes collection and handoff, not solely transfer.",
            "completed_queue_get_seconds": queue_seconds,
            "completed_queue_get_cycle_fraction": queue_seconds / seconds,
            "other_collection_seconds": max(0, sum(durations["generation_wait"]) - queue_seconds),
            "other_collection_cycle_fraction": max(0, sum(durations["generation_wait"]) - queue_seconds) / seconds,
        }
    dropped = sum(r["queue_metrics"].get(prefix + "dropped_response_tokens", 0) for r in chosen)
    delivered = sum(r["response_tokens"] for r in chosen)
    components = {
        "standalone_scoring": [
            r["seconds"] for r in contract if r["event"] == "score_timing" and r["rollout_id"] >= warmup
        ],
        "forward_backward_optimizer": [
            r["elapsed_seconds"]
            for r in contract
            if r["event"] == "optimizer" and r.get("rollout_id", r["step"] - 1) >= warmup and "elapsed_seconds" in r
        ],
    }
    provenance = [
        r for records in contracts for r in records if r["event"] == "refresh_scores" and r["rollout_id"] >= warmup
    ]
    provenance_tokens = sum(r["active_tokens"] for r in provenance)
    result = {
        "current_policy_token_fraction": (
            sum(r["active_tokens"] * r["current_version_token_fraction"] for r in provenance) / provenance_tokens
            if provenance_tokens
            else None
        ),
        "provenance_active_tokens": provenance_tokens,
        "scope": "Warm awaited driver-cycle throughput, excluding startup, checkpoints, evaluation and final drain; generation overlaps training.",
        "completed_updates": expected,
        "validated_trainer_ranks": ranks,
        "training_components_rank0": {
            name: {
                "count": len(values),
                "total_seconds": sum(values),
                "median_seconds": statistics.median(values) if values else None,
            }
            for name, values in components.items()
        },
        "warmup_updates": warmup,
        "measured_updates": len(chosen),
        "warm_cycle_seconds": seconds,
        "useful_response_tokens": delivered,
        "useful_response_tokens_per_second": delivered / seconds,
        "trainer_wait_fraction": sum(durations["generation_wait"]) / seconds,
        "batch_collection_breakdown": wait_breakdown,
        "discarded_response_tokens": dropped,
        "discarded_token_fraction": dropped / max(1, dropped + delivered),
        "mixed_responses": sum(r["mixed_responses"] for r in chosen),
        "per_update": [
            {
                "rollout_id": row["rollout_id"],
                "response_tokens": row["response_tokens"],
                "mixed_responses": row["mixed_responses"],
                "completed_queue_get_seconds": queue_waits[index],
                "other_collection_seconds": (
                    max(0, durations["generation_wait"][index] - queue_waits[index])
                    if wait_breakdown is not None
                    else None
                ),
                **{name + "_seconds": values[index] for name, values in durations.items()},
            }
            for index, row in enumerate(chosen)
        ],
        "median_seconds": {name: statistics.median(values) for name, values in durations.items()},
        "all_driver_stage_seconds": {
            name: sum(r["seconds"] for r in stages if r["stage"] == name)
            for name in sorted({r["stage"] for r in stages})
        },
        "workflow": json.loads((root / "workflow.json").read_text()),
    }
    inventory = root / "checkpoints/pipeline_lifecycle.jsonl"
    result["terminal_inventory"] = rows(inventory) if inventory.exists() else None
    result["terminal_unused_work"] = terminal_unused_work(
        result["terminal_inventory"],
        delivered_tokens=sum(row["response_tokens"] for row in flow),
        stale_dropped_tokens=sum(row["queue_metrics"].get(prefix + "dropped_response_tokens", 0) for row in flow),
    )
    result["end_to_end_passed"] = result["workflow"]["status"] == "complete"
    if not result["end_to_end_passed"] and not allow_incomplete_workflow:
        raise ValueError("Workflow did not complete")
    return result


def terminal_unused_work(lifecycle, *, delivered_tokens, stale_dropped_tokens):
    """Do not mistake unobserved final buffers for empty buffers."""
    if not lifecycle or lifecycle[-1]["event"] != "shutdown_complete":
        return None
    terminal = lifecycle[-1]
    sections = [terminal.get(key) for key in ("completed_queue", "producer_ready", "shutdown_unqueued")]
    if any(section is None for section in sections):
        return None
    remaining = {key: sum(section[key] for section in sections) for key in ("groups", "samples", "response_tokens")}
    accounted = delivered_tokens + stale_dropped_tokens + remaining["response_tokens"]
    return {
        **remaining,
        "delivered_response_tokens_all_updates": delivered_tokens,
        "stale_dropped_response_tokens_all_updates": stale_dropped_tokens,
        "fraction_of_accounted_response_tokens": remaining["response_tokens"] / accounted if accounted else None,
        "scope": "Final buffered, producer-ready and shutdown-unqueued completions; distinct from stale drops. Denominator excludes unobserved aborted or dynamically filtered work.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    report = analyze(args.root, warmup=args.warmup)
    (args.root / "throughput-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
