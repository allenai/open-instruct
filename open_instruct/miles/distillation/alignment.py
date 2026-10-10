"""Validate exact token alignment for same-tokenizer context distillation."""

import math


def extract_scores(result, ids, response_length):
    entries = result.get("meta_info", {}).get("input_token_logprobs")
    if not isinstance(entries, list) or len(entries) != len(ids):
        raise ValueError("Teacher must return one input score per requested token")
    if not 0 < response_length < len(ids):
        raise ValueError("Response must follow a nonempty teacher context")
    if any(len(entry) < 2 or entry[1] != token for entry, token in zip(entries, ids, strict=True)):
        raise ValueError("Teacher returned different token IDs or positions")
    scores = [entry[0] for entry in entries[-response_length:]]
    if any(not isinstance(v, int | float) or not math.isfinite(v) or v > 0 for v in scores):
        raise ValueError("Teacher response scores must be finite nonpositive log probabilities")
    return scores
