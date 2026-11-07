"""Stream prompt-level low-success and truncation history from explicitly selected run attempts.

This exports IDs and statistics, not generations, and does not change rewards or
build a training dataset. Use attempt-specific version bounds after a recovery.
"""

import argparse
import collections
import json
from pathlib import Path

from open_instruct.miles.datasets import replay_evidence


def summarize(manifests, output, *, max_version=400, ranges=None, max_successes=1, truncation="any"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seen, prompts = set(), {}
    counts = collections.Counter()
    by_source = collections.defaultdict(collections.Counter)
    attempts = []
    errors = []
    for path in manifests:
        path = Path(path)
        manifest = json.loads(path.read_text())
        attempt = manifest["run"]["attempt"]
        low, high = (ranges or {}).get(attempt, [0, max_version])
        records = path.with_name(f"records-{attempt}.jsonl")
        attempts.append(
            {
                "manifest": str(path),
                "run": manifest["run"],
                "policy_range": [low, high],
                "protocol_sha256": manifest.get("protocol_sha256"),
            }
        )
        with records.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except ValueError:
                    counts["malformed_rows"] += 1
                    if len(errors) < 20:
                        errors.append({"path": str(records), "line": number})
                    continue
                if record.get("kind") != "group":
                    continue
                observation = record.get("observation_id")
                if not observation:
                    counts["missing_observation_id"] += 1
                    continue
                if observation in seen:
                    counts["duplicate_observations"] += 1
                    continue
                seen.add(observation)
                versions = [v for r in record.get("responses", []) for v in r.get("policy_versions", [])]
                if not versions or any(not str(v).isdigit() for v in versions):
                    counts["unknown_policy_version"] += 1
                    continue
                if any(not low <= int(v) <= high for v in versions):
                    counts["outside_policy_range"] += 1
                    continue
                source = record.get("mixture_source") or record.get("source_dataset") or "unknown"
                task = record.get("task_key")
                if not task:
                    counts["missing_task_key"] += 1
                    continue
                key = f"{source}:{task}"
                outcome = replay_evidence.classify(record)
                category = outcome["category"]
                counts["legacy_if_validity_reconciled_responses"] += outcome["legacy_if_validity_reconciled"]
                counts[category] += 1
                by_source[source][category] += 1
                if outcome["raw_all_zero"]:
                    counts["raw_all_zero"] += 1
                    by_source[source]["raw_all_zero"] += 1
                entry = prompts.setdefault(
                    key,
                    {
                        "prompt_key": key,
                        "task_key": task,
                        "source": source,
                        "prepared_sample_id": record.get("prepared_sample_id"),
                        "source_row": record.get("source_row"),
                        "counts": {},
                        "zero_truncation_histogram": {},
                        "low_success_histogram": {},
                        "eligible_observations": 0,
                        "latest_valid_unix": -1,
                        "latest_valid_category": None,
                    },
                )
                entry["counts"][category] = entry["counts"].get(category, 0) + 1
                if category in replay_evidence.ZERO_CATEGORIES:
                    bucket = f"{outcome['truncated']}/{outcome['group_size']}"
                    entry["zero_truncation_histogram"][bucket] = entry["zero_truncation_histogram"].get(bucket, 0) + 1
                    counts[f"zero_truncation_{bucket}"] += 1
                successes = outcome["binary_successes"]
                if category not in ("invalid", "aborted") and successes is not None and successes <= 1:
                    bucket = f"{int(successes)}/{outcome['group_size']}"
                    cap_bucket = f"{outcome['truncated']}/{outcome['group_size']}"
                    histogram = entry["low_success_histogram"].setdefault(bucket, {})
                    histogram[cap_bucket] = histogram.get(cap_bucket, 0) + 1
                    metric = f"successes_{bucket}_truncated_{cap_bucket}"
                    counts[metric] += 1
                    by_source[source][metric] += 1
                if replay_evidence.eligible(outcome, max_successes=max_successes, truncation=truncation):
                    entry["eligible_observations"] += 1
                    counts["eligible_observations"] += 1
                    by_source[source]["eligible_observations"] += 1
                timestamp = record.get("recorded_unix", -1)
                if category not in ("invalid", "aborted") and timestamp >= entry["latest_valid_unix"]:
                    entry["latest_valid_unix"] = timestamp
                    entry["latest_valid_category"] = category
                for response in record.get("responses", []):
                    probe = response.get("literal_reference_in_text")
                    if type(probe) is bool:
                        counts["literal_reference_checked"] += 1
                        counts["literal_reference_found"] += probe
    zero_prompts = [p for p in prompts.values() if any(p["counts"].get(k, 0) for k in replay_evidence.ZERO_CATEGORIES)]
    pool_prompts = [p for p in prompts.values() if p["eligible_observations"]]
    low_prompts = [p for p in prompts.values() if p["low_success_histogram"]]
    summary = {
        "schema_version": 1,
        "pool_criteria": {"max_successes": max_successes, "truncation": truncation},
        "unique_ever_eligible": len(pool_prompts),
        "unique_eligible_by_source": dict(collections.Counter(p["source"] for p in pool_prompts)),
        "attempts": attempts,
        "counts": dict(counts),
        "by_source": dict(by_source),
        "unique_prompts": len(prompts),
        "unique_ever_zero": len(zero_prompts),
        "ever_zero_latest_nonzero": sum(
            p["latest_valid_category"] in ("mixed", "all_one", "constant_nonzero") for p in zero_prompts
        ),
        "malformed_examples": errors,
        "limits": [
            "Counts describe recorded observations; writer drops and incomplete attempts can leave gaps.",
            "Select manifests and policy-version ranges explicitly to exclude abandoned recovery branches.",
            "Ever-zero prompts may later succeed. Zero counts are not estimates of impossibility.",
            "Literal-reference matches are diagnostics, not verifier-confirmed answers; absent checks are unknown.",
            "No frozen-policy retry control: 0/k to mixed can happen through ordinary resampling.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "zero-prompts.jsonl").open("w") as stream:
        for entry in sorted(zero_prompts, key=lambda r: r["prompt_key"]):
            stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
    for name, rows in (("low-success-prompts.jsonl", low_prompts), ("replay-prompts.jsonl", pool_prompts)):
        with (output / name).open("w") as stream:
            for entry in sorted(rows, key=lambda r: r["prompt_key"]):
                stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-policy-version", type=int, default=400)
    parser.add_argument(
        "--attempt-ranges", type=Path, help="JSON map: attempt ID -> [inclusive min, inclusive max policy version]"
    )
    parser.add_argument("--max-successes", type=int, choices=(0, 1), default=1)
    parser.add_argument("--truncation", choices=("none", "any", "half", "all"), default="any")
    args = parser.parse_args(argv)
    ranges = json.loads(args.attempt_ranges.read_text()) if args.attempt_ranges else None
    result = summarize(
        args.manifest,
        args.output,
        max_version=args.max_policy_version,
        ranges=ranges,
        max_successes=args.max_successes,
        truncation=args.truncation,
    )
    print(json.dumps({k: result[k] for k in ("counts", "unique_prompts", "unique_ever_zero")}))


if __name__ == "__main__":
    main()
