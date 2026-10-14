"""Compare delivered OPD batches at matched update IDs using existing logs only."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


def read_batches(path):
    batches = {}
    for line in path.read_text().splitlines():
        match = re.search(r" - perf (\d+): (\{.*\})", line)
        if match is None:
            continue
        row = ast.literal_eval(match[2])
        if "rollout/episode_response_length/mean" not in row:
            continue
        index = int(match[1])
        if index in batches:
            raise ValueError(f"Duplicate rollout {index}; select one execution attempt explicitly")
        batches[index] = row
    if not batches:
        raise ValueError(f"No rollout batches found in {path}")
    return batches


def summarize(batches, indices):
    rows = [batches[index] for index in indices]
    samples = sum(row["rollout/num_training_samples"] for row in rows)
    result = {
        "updates": len(rows),
        "delivered_responses": samples,
        "response_length_mean": sum(
            row["rollout/episode_response_length/mean"] * row["rollout/num_training_samples"] for row in rows
        )
        / samples,
        "response_cap_fraction": sum(
            row["rollout/truncated_ratio"] * row["rollout/num_training_samples"] for row in rows
        )
        / samples,
    }
    for name in ("aborted_groups_filtered", "stale_groups_filtered"):
        key = f"rollout/fully_async/{name}"
        result[name] = sum(row[key] for row in rows) if all(key in row for row in rows) else None
    for name in ("dropped_groups", "dropped_samples", "dropped_response_tokens"):
        key = f"rollout/fully_async/completed_queue/{name}"
        result[f"completed_queue_{name}"] = sum(row[key] for row in rows) if all(key in row for row in rows) else None
    return result


def compare(sync_path, async_path):
    paths = {"sync": sync_path, "async": async_path}
    batches = {arm: read_batches(path) for arm, path in paths.items()}
    indices = sorted(batches["sync"].keys() & batches["async"].keys())
    if not indices:
        raise ValueError("No common rollout IDs")
    windows = {"all_matched": indices, "first_10": indices[:10], "last_10": indices[-10:]}
    return {
        "scope": "Matched update IDs, not paired prompts or a causal estimate; one existing trajectory per arm.",
        "counter_units": "Abort counts are group filter events; delivery counts are responses. Do not divide them.",
        "sources": {
            arm: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for arm, path in paths.items()
        },
        "unmatched_updates": {arm: sorted(set(rows) - set(indices)) for arm, rows in batches.items()},
        "windows": {
            name: {"rollout_ids": selected, **{arm: summarize(rows, selected) for arm, rows in batches.items()}}
            for name, selected in windows.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-log", type=Path, required=True)
    parser.add_argument("--async-log", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.sync_log, args.async_log), indent=2))


if __name__ == "__main__":
    main()
