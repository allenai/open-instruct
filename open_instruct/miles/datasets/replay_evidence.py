"""Classify zero-reward evidence without turning failures or text matches into rewards."""

import math
import re

SELECTION_ID = "prompt_replay_selection_id"
REPLAY = "prompt_replay_review"
KEY = "prompt_replay_prompt_key"
ZERO_CATEGORIES = ("zero_completed", "zero_some_truncated", "zero_all_truncated")


def classify(record):
    responses = record.get("responses") or []
    size = record.get("group_size")
    complete = type(size) is int and size > 1 and len(responses) == size
    complete = complete and record.get("expected_group_size", size) == size
    rewards, errors, unknown, truncated, reconciled = [], 0, 0, 0, 0
    lengths = []
    for response in responses:
        reward = response.get("reward")
        if isinstance(reward, dict):
            reward = reward.get(record.get("reward_key"))
        finite = type(reward) in (int, float) and math.isfinite(reward)
        rewards.append(reward if finite else None)
        cap = response.get("truncated") is True and response.get("status") == "truncated"
        truncated += cap
        validity = response.get("validity") or {}
        # The truncation gate deliberately skips verification. Other unverified
        # zeroes must not be promoted into evidence of task difficulty.
        gated = cap and record.get("reward_zero_truncated") is True and reward == 0
        components = validity.get("components") or {}
        # Historical MILES allowlisted ok/completed but not IF's successful
        # "scored" status. Reconcile this exact known adapter mismatch only.
        legacy_if = (
            bool(components)
            and components.get("ifeval") == "scored"
            and all(
                status in ("ok", "completed") or (name == "ifeval" and status == "scored")
                for name, status in components.items()
            )
        )
        reconciled += legacy_if and validity.get("valid") is not True and not gated
        valid = validity.get("valid") is True or gated or legacy_if
        unknown += not valid and validity.get("valid") is None
        errors += not valid and validity.get("valid") is False
        complete = complete and finite and response.get("status") in ("completed", "truncated") and valid
        length = response.get("response_tokens")
        if type(length) is int and length >= 0:
            lengths.append(length)
    raw_zero = bool(responses) and all(r == 0 for r in rewards)
    if record.get("filter_decision") == "aborted":
        category = "aborted"
    elif not complete:
        category = "invalid"
    elif raw_zero:
        category = ZERO_CATEGORIES[2 if truncated == size else 1 if truncated else 0]
    elif len(set(rewards)) > 1:
        category = "mixed"
    elif all(r == 1 for r in rewards):
        category = "all_one"
    else:
        category = "constant_nonzero"
    return {
        "category": category,
        "raw_all_zero": raw_zero,
        "group_size": size,
        "truncated": truncated,
        "verifier_errors": errors,
        "unknown_validity": unknown,
        "legacy_if_validity_reconciled": reconciled,
        "binary_successes": sum(rewards) if complete and all(r in (0, 1) for r in rewards) else None,
        "positive_responses": sum(r is not None and r > 0 for r in rewards),
        "response_tokens": sum(lengths) if len(lengths) == len(responses) and responses else None,
    }


def literal_answer_probe(text, target):
    """Conservative literal-reference diagnostic, NOT a mathematical verifier.

    Only scalar math references are passed by the caller. Boundaries keep 42 from
    matching 142 or 42.5. A mention in unfinished reasoning is not a correct final
    answer; equivalence, negation and contradictions are intentionally not inferred.
    """
    if not isinstance(text, str) or type(target) not in (str, int, float):
        return None
    answer = " ".join(str(target).split())
    if not answer or len(answer) > 256:
        return None
    pattern = r"(?<![\w.])" + re.escape(answer) + r"(?!\w|\.\d)"
    return bool(re.search(pattern, " ".join(text.split())))


def eligible(outcome, *, max_successes=1, truncation="any"):
    successes, size = outcome["binary_successes"], outcome["group_size"]
    if (
        outcome["category"] in ("aborted", "invalid")
        or successes is None
        or successes > max_successes
        or successes >= size
    ):
        return False
    caps = outcome["truncated"]
    return {"none": True, "any": caps > 0, "half": caps * 2 >= size, "all": caps == size}[truncation]
