"""Pool observations and optimizer-boundary dashboard scalars, independent of policy."""

from open_instruct.miles.datasets import replay_evidence

OBSERVATIONS = (
    "attempted",
    "fresh_attempted",
    "replay_attempted",
    "scored",
    "invalid",
    "aborted",
    "filtered",
    "filtered_all_zero",
    "filtered_all_one",
    "filtered_constant_nonzero",
    "filtered_other",
    "discarded",
    "discarded_unobserved",
    "groups_with_truncation",
    "truncated_responses",
    "returned_responses",
)


def observe(counts, entry, record):
    """Count a selection's first returned group, even if grading is invalid.

    The provider owns duplicate suppression. A proposal has not necessarily
    generated anything; an observed group has not necessarily reached training.
    """
    pool = entry["pool"]
    outcome = replay_evidence.classify(record)
    replay = entry["metadata"].get(replay_evidence.REPLAY, False)
    counts[f"{pool}/attempted"] += 1
    counts[f"{pool}/{'replay' if replay else 'fresh'}_attempted"] += 1
    category = outcome["category"]
    counts[f"{pool}/" + (category if category in ("invalid", "aborted") else "scored")] += 1
    caps = sum(response.get("truncated") is True for response in record.get("responses", []))
    counts[f"{pool}/groups_with_truncation"] += caps > 0
    counts[f"{pool}/truncated_responses"] += caps
    counts[f"{pool}/returned_responses"] += len(record.get("responses", []))
    if record.get("filter_decision") == "filtered":
        counts[f"{pool}/filtered"] += 1
        if category in replay_evidence.ZERO_CATEGORIES:
            reason = "all_zero"
        elif category == "all_one":
            reason = "all_one"
        elif category == "constant_nonzero":
            reason = "constant_nonzero"
        else:
            reason = "other"
        counts[f"{pool}/filtered_{reason}"] += 1


def dashboard(metrics, update):
    """Metric-first names make all pools selectable together in W&B panels.

    These snapshots are emitted after optimizer completion using a dedicated
    update axis, including the final update. Recorder metrics may precede training.
    """
    result = {"pool/update": update}
    for key, value in metrics.items():
        if key.startswith("rollout/pool/"):
            pool, metric = key.removeprefix("rollout/pool/").split("/", 1)
            if "/" not in metric:  # Replay's detailed diagnostics remain in recorder metrics.
                result[f"pool/{metric}/{pool}"] = value
    return result
