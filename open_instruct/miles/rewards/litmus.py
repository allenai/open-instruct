"""Litmus v0.2 row-carried extraction and scalar reward contracts.

This version requires an explicit one-capture regex and unambiguous scalar
captures; it never searches a malformed capture for an incidental number.
"""

import math
import time
from typing import Any

import regex

from open_instruct import logger_utils
from open_instruct.miles.rewards import deterministic_verifiers as common
from open_instruct.miles.rewards import task_verifiers

logger = logger_utils.setup_logger(__name__)
KIND = "litmus_scalar_v1"
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
    # str(None) or str([...]) must not become a reference answer.
    if type(target.get("expected_answer")) not in (str, int, float, bool):
        raise ValueError("Litmus requires a scalar expected_answer")
    expected = scalar(target["expected_answer"], kind)
    if expected == "":
        raise ValueError("Litmus requires a nonempty expected_answer")
    pattern = target.get("output_regex")
    if not isinstance(pattern, str) or len(pattern) > 4096:
        raise ValueError("Litmus requires explicit bounded output_regex")
    try:
        pattern = regex.compile(pattern, regex.S)
    except (regex.error, RecursionError) as error:
        raise ValueError(f"Invalid Litmus output_regex: {error}") from error
    if pattern.groups != 1:
        raise ValueError("Litmus output_regex must contain exactly one capture")
    default: dict[str, Any] = {"rule": DEFAULT_RULES[kind]}
    match = target.get("match") or default
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
        start = time.monotonic()
        try:
            # Preserve the row's final-match convention with one bounded search.
            matches = pattern.findall(common.answer_text(prediction), timeout=0.1)
        except TimeoutError as error:
            logger.warning("Litmus verifier timeout; assigning fallback zero reward: %s", error)
            return common.Result(
                0.0,
                diagnostics={
                    "kind": KIND,
                    "status": "timeout",
                    "error": str(error) or "Litmus output_regex exceeded its search budget",
                    "failure_policy": "zero",
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
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
        return common.Result(float(score), diagnostics={"kind": KIND, "status": "ok"})
