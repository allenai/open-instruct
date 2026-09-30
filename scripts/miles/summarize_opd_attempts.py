"""Summarize an OPD selection trace without counting group events as responses.

Unresolved attempts are censored as of the input file's end. Submission is a
Python producer event, not proof that an inference engine admitted the request.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def length_stats(rows):
    lengths = [length for row in rows for length in row["response_lengths"]]
    return {
        "groups": len(rows),
        "responses": len(lengths),
        "observed_tokens": sum(lengths),
        "mean_observed_response_length": sum(lengths) / len(lengths) if lengths else None,
        "responses_at_16384_cap": sum(length == 16384 for length in lengths),
    }


def summarize(rows):
    attempts, by_event = defaultdict(list), defaultdict(list)
    for row in rows:
        if not row["attempt_id"]:
            raise ValueError("Trace contains an event without an attempt identity")
        attempts[row["attempt_id"]].append(row)
        by_event[row["event"]].append(row)
    submissions, deliveries, outcomes = Counter(), Counter(), Counter()
    retry_waits, batches = [], defaultdict(list)
    delivered_attempts = set()
    for attempt_id, events in attempts.items():
        event_counts = Counter(row["event"] for row in events)
        if event_counts["submitted"] != 1 or any(count != 1 for count in event_counts.values()):
            raise ValueError("Expected one submission and at most one event of each kind per attempt")
        prompts = {row["prompt_sha256"] for row in events}
        if len(prompts) != 1:
            raise ValueError("An attempt changed its prompt")
        prompt = prompts.pop()
        submissions[prompt] += 1
        delivered = next((row for row in events if row["event"] == "batch_delivered"), None)
        rejected = any(row["event"].endswith("_rejected") for row in events)
        if delivered and rejected:
            raise ValueError("An attempt was both delivered and rejected")
        if delivered:
            deliveries[prompt] += 1
            delivered_attempts.add(attempt_id)
            batches[delivered["rollout_id"]].append(delivered)
            submitted = next(row for row in events if row["event"] == "submitted")
            retry_waits.append((delivered["time_ns"] - submitted["time_ns"]) / 1e9)
            outcomes["delivered"] += 1
        elif rejected:
            outcomes["rejected"] += 1
        elif "generation_failed" in event_counts:
            outcomes["generation_failed"] += 1
        elif "generation_cancelled" in event_counts:
            outcomes["generation_cancelled"] += 1
        else:
            outcomes["unresolved_at_file_end"] += 1
    completed_returned = [
        row
        for row in by_event["generation_returned"]
        if all(status in ("completed", "truncated") for status in row["statuses"])
    ]
    return {
        "scope": __doc__,
        "event_counts_in_groups": {name: len(values) for name, values in by_event.items()},
        "attempts": len(attempts),
        "attempt_outcomes": dict(outcomes),
        "unique_submitted_prompt_hashes": len(submissions),
        "unique_delivered_prompt_hashes": len(deliveries),
        "unique_submitted_prompts_without_delivery": len(submissions.keys() - deliveries.keys()),
        "submission_count_histogram_by_prompt": dict(sorted(Counter(submissions.values()).items())),
        "delivery_count_histogram_by_prompt": dict(sorted(Counter(deliveries.values()).items())),
        "mean_submitted_to_delivery_seconds_for_delivered_attempts": sum(retry_waits) / len(retry_waits)
        if retry_waits
        else None,
        "delivered": length_stats(by_event["batch_delivered"]),
        "completed_returns": length_stats(completed_returned),
        "completed_returns_not_delivered_as_of_file_end": length_stats(
            [row for row in completed_returned if row["attempt_id"] not in delivered_attempts]
        ),
        "rejection_observations_partial_lengths_are_lower_bounds": {
            event: length_stats(values) for event, values in by_event.items() if event.endswith("_rejected")
        },
        "batches": {str(key): length_stats(values) for key, values in sorted(batches.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    result = summarize([json.loads(line) for line in raw.splitlines()])
    result["source"] = {"path": str(args.trace), "sha256": hashlib.sha256(raw).hexdigest()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
