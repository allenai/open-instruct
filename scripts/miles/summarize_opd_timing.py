"""Summarize OPD intervals without mistaking overlapping request time for utilization."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def union_seconds(intervals):
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["stage"]].append(row)
    result = {}
    for stage, values in groups.items():
        durations = sorted(r["seconds"] for r in values if "seconds" in r)
        entry = dict(records=len(values), outcomes=dict(Counter(r.get("outcome", "snapshot") for r in values)))
        if durations:
            intervals = [(r["started_unix"], r["started_unix"] + r["seconds"]) for r in values if "seconds" in r]
            entry.update(
                summed_seconds=sum(durations),
                mean_seconds=sum(durations) / len(durations),
                p95_seconds=durations[math.ceil(0.95 * len(durations)) - 1],
                observed_union_seconds=union_seconds(intervals),
            )
        if stage == "learner_memory":
            entry["ranks"] = {
                rank: {
                    k: max(r[k] for r in values if str(r["rank"]) == rank)
                    for k in ("peak_allocated_bytes", "peak_reserved_bytes", "device_total_bytes")
                }
                for rank in sorted({str(r["rank"]) for r in values})
            }
        result[stage] = entry
    return dict(
        stages=result,
        interpretation=[
            "Request seconds overlap; do not sum student, teacher and learner durations as wall time.",
            "Student/teacher request intervals include transport, server queuing and processing; they are not GPU kernel timings.",
            "Learner batch and prefetch waits are driver blocking; collection time can overlap training.",
            "Compare steady training windows separately from startup/evaluation/checkpoint intervals.",
            "Join attempt_id and sample_index with selection events to separate delivered and discarded work.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for path in sorted(args.directory.glob("events-*.jsonl"))
        for line in path.read_text().splitlines()
        if line
    ]
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
