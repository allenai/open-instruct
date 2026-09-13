"""Audit ownership and overlap from retained engine-drain and driver timelines.

This reports observed protocol evidence, not a learning or restart certificate.
Pass the directory containing engine_drain.jsonl and driver_timing.jsonl.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze(events, stages):
    assignments, responses = {}, {}
    states, versions, starts = {}, {}, {}
    intervals, independent, independent_responses = [], [], []
    counts = Counter()
    failures = []
    group_versions = defaultdict(set)
    phase_seconds = defaultdict(list)
    delivered_bytes = 0
    max_retained = 0
    retained = {}
    for row in events:
        kind = row["event"]
        counts[kind] += 1
        identity, timestamp = row.get("engine"), row.get("time")
        if kind == "group_reserved":
            for request in row["requests"]:
                if request in assignments:
                    failures.append(f"duplicate reservation {request}")
                assignments[request] = (identity, row["version"], row["group"], row.get("attempt", 0))
        elif kind == "decode_finished":
            request = row["request"]
            expected = (identity, row["executed_version"], row["group"], row.get("attempt", 0))
            if assignments.get(request) != expected or row["assigned_version"] != row["executed_version"]:
                failures.append(f"response ownership/version mismatch {request}")
            if request in responses:
                failures.append(f"duplicate response {request}")
            responses[request] = row["tokens"]
            group_versions[row["group"], row.get("attempt", 0)].add(row["executed_version"])
            peers = [
                peer
                for peer, state in states.items()
                if peer != identity
                and state == "draining"
                and versions.get(peer, row["executed_version"]) < row["executed_version"]
            ]
            if peers:
                independent_responses.append(
                    {
                        "engine": identity,
                        "request": request,
                        "version": row["executed_version"],
                        "older_draining_peers": peers,
                    }
                )
        elif kind in ("drain_started", "update_started"):
            state = "draining" if kind == "drain_started" else "updating"
            states[identity] = state
            starts[identity, state] = timestamp
            if "version" in row:
                versions[identity] = row["version"]
        elif kind in ("drain_finished", "engine_reopened", "update_finished"):
            phase = "draining" if kind == "drain_finished" else "updating"
            start = starts.pop((identity, phase), None)
            if start is None:
                failures.append(f"missing {phase} start for {identity}")
            else:
                intervals.append((identity, phase, start, timestamp))
                phase_seconds[phase].append(timestamp - start)
            if kind == "engine_reopened":
                peers = [
                    peer
                    for peer, state in states.items()
                    if peer != identity and state == "draining" and versions.get(peer, row["version"]) < row["version"]
                ]
                if peers:
                    independent.append(
                        {
                            "engine": identity,
                            "version": row["version"],
                            "time": timestamp,
                            "older_draining_peers": peers,
                        }
                    )
                states[identity] = "serving"
                versions[identity] = row["version"]
        elif kind == "snapshot_ready":
            retained[row["version"]] = row["bytes"]
            max_retained = max(max_retained, sum(retained.values()))
            phase_seconds["snapshot_capture"].append(row["capture_seconds"])
        elif kind == "snapshot_released":
            if retained.pop(row["version"], None) is None:
                failures.append(f"unowned snapshot release {row['version']}")
        elif kind == "delivery_measured":
            delivered_bytes += row["bytes"]
            phase_seconds["delivery"].append(row["delivery_seconds"])
    overlaps = []
    for row in stages:
        if row["stage"] != "training" or not row["passed"]:
            continue
        completed = row["started_unix"] + row["seconds"]
        active = [
            (engine, phase)
            for engine, phase, start, end in intervals
            if start <= row["started_unix"] < completed < end
        ]
        if active:
            overlaps.append({"rollout_id": row["rollout_id"], "completed_unix": completed, "active": active})
    mixed = [group for group, seen in group_versions.items() if len(seen) != 1]
    return {
        "scope": "protocol timeline only; numerical weights, learning, lag, and resume need separate audits",
        "event_counts": dict(counts),
        "ownership_errors": failures,
        "mixed_groups": mixed,
        "reserved_requests": len(assignments),
        "completed_responses": len(responses),
        "generated_tokens_observed": sum(responses.values()),
        "requests_without_terminal_decode": sorted(set(assignments) - set(responses)),
        "fast_reopened_while_older_peer_drained": independent,
        "optimizer_completions_during_publication": overlaps,
        "new_version_responses_completed_while_peer_draining": independent_responses,
        "peak_retained_snapshot_bytes": max_retained,
        "unreleased_snapshots": sorted(retained),
        "delivered_bytes": delivered_bytes,
        "phase_seconds": {
            phase: {"count": len(values), "mean": sum(values) / len(values), "max": max(values)}
            for phase, values in phase_seconds.items()
        },
        "warning": "Phase durations overlap; do not sum them as elapsed wall time. Unknown HTTP outcomes have unknown discarded tokens.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(records(args.directory / "engine_drain.jsonl"), records(args.directory / "driver_timing.jsonl"))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
