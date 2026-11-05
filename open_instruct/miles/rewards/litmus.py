"""Litmus v0.2 row-carried extraction and scalar reward contracts.

This version requires an explicit one-capture regex and unambiguous scalar
captures; it never searches a malformed capture for an incidental number.
"""

import math
import os

import regex

from open_instruct.miles.rewards import candidate_verifiers as common
from open_instruct.miles.rewards import task_verifiers

DEFAULT_RULES = {"float": "isclose", "bool": "bool_eq", "string": "string_eq"}
RULES = {"exact", "isclose", "abs_window", "rel_window", "bool_eq", "string_eq"}


def scalar(value, kind):
    text = str(value).strip()
    if kind == "string":
        return " ".join(text.casefold().split())
    if kind == "bool":
        if text.lower() in {"true", "yes", "1", "1.0"}:
            return True
        if text.lower() in {"false", "no", "0", "0.0"}:
            return False
        raise ValueError("Expected explicit boolean")
    result = float(text)
    if not math.isfinite(result):
        raise ValueError("Expected finite number")
    return result


def validate_target(target):
    if not isinstance(target, dict) or target.get("answer_type") not in DEFAULT_RULES:
        raise ValueError("Litmus requires explicit float/bool/string answer_type")
    kind = target["answer_type"]
    expected = scalar(target["expected_answer"], kind)
    pattern = target.get("output_regex")
    if not isinstance(pattern, str) or len(pattern) > 4096:
        raise ValueError("Litmus requires explicit bounded output_regex")
    pattern = regex.compile(pattern, regex.S)
    if pattern.groups != 1:
        raise ValueError("Litmus output_regex must contain exactly one capture")
    match = target.get("match") or {"rule": DEFAULT_RULES[kind]}
    if not isinstance(match, dict) or match.get("rule") not in RULES or set(match) - {"rule", "abs_tol", "rel_tol"}:
        raise ValueError("Unsupported Litmus matching contract")
    rule = match["rule"]
    if (
        (kind == "bool" and rule != "bool_eq")
        or (kind == "string" and rule != "string_eq")
        or (kind == "float" and rule in {"bool_eq", "string_eq"})
    ):
        raise ValueError("Litmus answer type and rule disagree")
    tolerances = {key: match.get(key, 1e-6) for key in ("abs_tol", "rel_tol")}
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in tolerances.values()):
        raise ValueError("Litmus tolerances must be finite and nonnegative")
    return pattern, expected, rule, tolerances


class LitmusVerifier:
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        pass

    async def async_call(self, tokens, prediction, label, **kwargs):
        pattern, expected, rule, tolerance = validate_target(label)
        try:
            # Preserve the row's final-match convention with one bounded search.
            matches = pattern.findall(common.answer_text(prediction), timeout=0.1)
        except TimeoutError:
            policy = os.environ.get(
                "OI_MILES_FORMAT_TIMEOUT_POLICY", os.environ.get("OI_MILES_MATH_TIMEOUT_POLICY", "zero")
            )
            if policy == "raise":
                raise
            if policy != "zero":
                raise ValueError("Timeout policy must be zero or raise") from None
            return common.Result(0.0, diagnostics={"kind": "litmus_scalar_v1", "status": "timeout"})
        try:
            predicted = scalar(matches[-1], label["answer_type"]) if matches else None
        except (ValueError, TypeError):
            predicted = None
        if predicted is None:
            score = False
        elif rule in {"bool_eq", "string_eq"}:
            score = predicted == expected
        elif rule == "exact":
            score = round(predicted) == round(expected)
        elif rule == "isclose":
            score = math.isclose(predicted, expected, **tolerance)
        elif rule == "abs_window":
            score = abs(predicted - expected) <= tolerance["abs_tol"]
        else:
            score = abs(predicted - expected) <= abs(tolerance["rel_tol"] * expected)
        return common.Result(float(score), diagnostics={"kind": "litmus_scalar_v1", "status": "ok"})
