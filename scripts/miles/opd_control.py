"""Pure contracts for the bounded prompt-selection / policy-age experiment."""

import hashlib
import json
import random
from collections import Counter


def prompt_hash(prompt):
    return hashlib.sha256(
        json.dumps(prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def schedules(events, updates=8, groups_per_update=128):
    count = updates * groups_per_update
    submitted = [row["prompt_sha256"] for row in events if row["event"] == "submitted"]
    delivered = [row for row in events if row["event"] == "batch_delivered"]
    counts = Counter(row["rollout_id"] for row in delivered)
    if counts != {i: groups_per_update for i in range(updates)}:
        raise ValueError("Trace must contain exactly the expected complete delivered batches")
    # Preserve observed membership per update, but use a canonical order within
    # each cohort so original completion order does not also choose DP partitions.
    selected = [
        row["prompt_sha256"]
        for i in range(updates)
        for row in sorted((r for r in delivered if r["rollout_id"] == i), key=lambda r: r["prompt_sha256"])
    ]
    admitted = submitted[:count]
    if len(admitted) != count or len(set(admitted)) != count or len(set(selected)) != count:
        raise ValueError("This screen requires unique prompts and a complete admission prefix")
    admitted = [
        value
        for offset in range(0, count, groups_per_update)
        for value in sorted(admitted[offset : offset + groups_per_update])
    ]
    return {"admitted": admitted, "selected": selected}


def publication_due(rollout_id, period, updates):
    if period < 1 or updates < 1:
        raise ValueError("Positive publication period and update count required")
    return rollout_id is None or (rollout_id + 1) % period == 0 or rollout_id + 1 == updates


def expected_age(rollout_id, period):
    if rollout_id < 0 or period < 1:
        raise ValueError("Invalid rollout or publication period")
    return rollout_id % period


def resolve_cohorts(rows, events, seed=42, updates=8, groups_per_update=128):
    """Recover exact original rows, retaining duplicate prompts' distinct metadata.

    The pinned native Dataset shuffles row indices with Python random(seed) at
    epoch zero. Verify every submitted group before relying on that reconstruction;
    filters, recycling, epoch wrap or a changed seed fail instead of guessing.
    """
    expected = schedules(events, updates, groups_per_update)
    permutation = list(range(len(rows)))
    random.Random(seed).shuffle(permutation)
    submitted = [row for row in events if row["event"] == "submitted"]
    if [row["group_index"] for row in submitted] != list(range(len(submitted))):
        raise ValueError("Trace is not an unrecycled epoch-zero admission sequence")
    for event in submitted:
        index = event["group_index"]
        if index >= len(permutation) or prompt_hash(rows[permutation[index]]["input"]) != event["prompt_sha256"]:
            raise ValueError("Original row order does not match the submitted trace")
    delivered = [row for row in events if row["event"] == "batch_delivered"]
    chosen = {
        "admitted": [
            row
            for step in range(updates)
            for row in sorted(
                submitted[step * groups_per_update : (step + 1) * groups_per_update], key=lambda r: r["prompt_sha256"]
            )
        ],
        "selected": [
            row
            for step in range(updates)
            for row in sorted((r for r in delivered if r["rollout_id"] == step), key=lambda r: r["prompt_sha256"])
        ],
    }
    output = {}
    for name, selected in chosen.items():
        indices = [permutation[row["group_index"]] for row in selected]
        cohort = [rows[index] for index in indices]
        if [prompt_hash(row["input"]) for row in cohort] != expected[name]:
            raise ValueError("Resolved cohort differs from the frozen prompt schedule")
        output[name] = dict(
            rows=cohort, source_row_indices=indices, group_indices=[r["group_index"] for r in selected]
        )
    return output
