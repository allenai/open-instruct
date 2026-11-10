"""Deterministic reward contracts for prepared dataset sources.

These factories accept data, never Python supplied by a dataset. Preparation must
validate targets and positive/negative controls before publishing a source.
Formatting rewards measure formatting only, not factual answer correctness.
"""

import ast
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

import regex

from open_instruct import logger_utils
from open_instruct.miles.rewards import task_verifiers

_QUOTED_PAIR_PATTERN = r'".+":\s*".+"'
_FORMAT_SEARCH_TIMEOUT_SECONDS = 0.1
_FORMAT_RESPONSE_TIMEOUT_SECONDS = 1.0
# An explicit choice: a boxed letter, or "Answer: LETTER". Emphasis, parentheses
# and \\text{} around the letter are presentation, not content. A bare letter must
# not run into a word ("Answer: I think") or echo the prompt ("Answer: A/B/C/D").
_CHOICE_DECLARATION = re.compile(
    r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?([A-Z])\)?[\s}]*\}"
    r"|(?i:answer)\s*:[\s*]*(?:\\?\(([A-Z])\\?\)|([A-Z])(?![A-Za-z0-9/]|[ \t]+[a-z]))"
)
logger = logger_utils.setup_logger(__name__)


class FormattingVerifierTimeout(TimeoutError):
    """A valid formatting contract exceeded its response-matching budget."""


@dataclass
class Result:
    score: float
    cost: float = 0.0
    diagnostics: dict = field(default_factory=lambda: {"kind": "deterministic_local_v1", "status": "ok"})


def answer_text(text):
    """Use an explicit final region when present; support non-thinking policies."""
    text = text.rsplit("</think>", 1)[-1].strip()
    for suffix in ("<|im_end|>", "<|endoftext|>"):
        text = text.removesuffix(suffix).strip()
    return text


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Nonfinite JSON number: {value}")
        return result

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)


def typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def validate_target(target):
    if not isinstance(target, dict) or target.get("kind") not in {
        "mcqa",
        "mcqa_final_choice",
        "format",
        "countdown",
        "knights",
        "numeric",
        "exact_text",
        "json_exact",
    }:
        raise ValueError("Unsupported deterministic target")
    kind = target["kind"]
    value = target.get("value")
    if kind in {"mcqa", "mcqa_final_choice"}:
        choices = target.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) < 2
            or len(set(choices)) != len(choices)
            or any(not isinstance(c, str) or re.fullmatch("[A-Z]", c) is None for c in choices)
            or value not in choices
        ):
            raise ValueError("MCQA requires distinct displayed choice letters and one correct letter")
        pattern = target.get("pattern")
        if pattern is not None:
            if kind != "mcqa_final_choice" or not isinstance(pattern, str) or len(pattern) > 4096:
                raise ValueError("Invalid MCQA answer pattern")
            try:
                groups = regex.compile(pattern).groups
            except (regex.error, RecursionError) as error:
                raise ValueError(f"Invalid MCQA answer pattern: {error}") from error
            if groups != 1:
                raise ValueError("MCQA answer pattern must contain exactly one capture")
    elif kind == "format":
        if not isinstance(value, dict) or value.get("type") not in {"regex", "inline_prose", "string_match"}:
            raise ValueError("Unsupported formatting contract")
        if value["type"] == "string_match":
            markers = value.get("expected_markers")
            if not isinstance(markers, list) or not markers or any(not isinstance(x, str) or not x for x in markers):
                raise ValueError("Formatting target needs nonempty expected markers")
            patterns = value.get("patterns", [])
        else:
            patterns = value.get("verify_regex")
            minimum = value.get("verify_min_matches", 1)
            if type(minimum) is not int or minimum < 1:
                raise ValueError("Formatting minimum must be positive")
        if not isinstance(patterns, list) or (value["type"] != "string_match" and not patterns):
            raise ValueError("Invalid formatting patterns")
        for pattern in patterns:
            if not isinstance(pattern, str) or len(pattern) > 4096:
                raise ValueError("Invalid formatting regex")
            try:
                re.compile(pattern)
                regex.compile(pattern)
            except (re.error, regex.error, RecursionError) as error:
                raise ValueError(f"Invalid formatting regex: {error}") from error
    elif kind == "countdown":
        nums = target.get("nums")
        if (
            type(value) is not int
            or not isinstance(nums, list)
            or not 1 <= len(nums) <= 8
            or any(type(x) is not int or not 0 < x < 1000000 for x in nums)
        ):
            raise ValueError("Invalid countdown target")
    elif kind == "knights":
        if (
            not isinstance(value, dict)
            or not value
            or any(
                not isinstance(k, str) or re.fullmatch(r"[A-Za-z]+", k) is None or type(v) is not bool
                for k, v in value.items()
            )
        ):
            raise ValueError("Invalid named knight/knave assignment")
        if len({name.casefold() for name in value}) != len(value):
            raise ValueError("Knight/knave names must be distinct ignoring case")
    elif kind == "numeric":
        if not isinstance(value, str) or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value) is None:
            raise ValueError("Numeric target must be a finite decimal string")
    elif kind == "exact_text":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Empty exact answer")
    else:
        # Also disallow non-JSON objects and nonfinite target numbers.
        strict_json(json.dumps(value, allow_nan=False))
    return kind


def countdown_value(expression, nums):
    if len(expression) > 256 or re.fullmatch(r"[\d\s+*/().-]+", expression) is None:
        return None
    found = []

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            found.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("Unsupported expression")

    try:
        result = visit(ast.parse(expression, mode="eval").body)
        return result if Counter(found) == Counter(nums) else None
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return None


def _quoted_pair_line(line, *, timeout=None):
    r"""Boolean equivalent of `".+":\s*".+"` on one newline-free line.

    The original unanchored search can retry every opening quote against the
    same failing suffix. Search the key/value separator once instead. The first
    and last quotes maximize the available key and value lengths; each must
    contain at least one character. Quotes inside either value remain allowed.
    """
    first, last = line.find('"'), line.rfind('"')
    if first < 0 or last - first < 6:
        return False
    return any(
        match.start() >= first + 2 and last >= match.end() + 2
        for match in regex.finditer(r'":\s*(?=")', line, timeout=timeout)
    )


def _format_search_budget(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Formatting response matching deadline exceeded")
    return min(_FORMAT_SEARCH_TIMEOUT_SECONDS, remaining)


def _format_score(final, value):
    deadline = time.monotonic() + _FORMAT_RESPONSE_TIMEOUT_SECONDS
    try:
        if value["type"] == "string_match":
            expected = set(value["expected_markers"])
            for marker in expected:
                _format_search_budget(deadline)
                present = marker in final
                _format_search_budget(deadline)
                if not present:
                    return 0.0
            for pattern in value.get("patterns", []):
                for match in regex.finditer(pattern, final, timeout=_format_search_budget(deadline)):
                    _format_search_budget(deadline)
                    if match.group() not in expected:
                        return 0.0
                _format_search_budget(deadline)
            return 1.0
        # inline_prose uses the same minimum-matching-lines contract as regex.
        matches = 0
        for line in final.split("\n"):
            # Equivalence is established only for this exact pattern. Variants
            # retain regex semantics and the existing timeout guard.
            for pattern in value["verify_regex"]:
                budget = _format_search_budget(deadline)
                matched = (
                    _quoted_pair_line(line, timeout=budget)
                    if pattern == _QUOTED_PAIR_PATTERN
                    else regex.search(pattern, line, timeout=budget)
                )
                _format_search_budget(deadline)
                if matched:
                    matches += 1
                    break
            if matches >= value.get("verify_min_matches", 1):
                return 1.0
        _format_search_budget(deadline)
        return 0.0
    except TimeoutError as error:
        # A verifier failure is unknown, not a valid zero reward. Preserve the
        # exception, but identify the contract in native rollout error records.
        raise FormattingVerifierTimeout(
            f"Formatting verifier timed out: pattern_id={value.get('pattern_id')!r}, "
            f"response_chars={len(final)}; original_error={error}"
        ) from error


def _final_choice_score(final, target):
    """Score one explicit final choice without searching reasoning for a letter.

    Declarations are a boxed letter, "Answer: LETTER" and the row's own requested
    format (``pattern``, one capture). They may repeat but must agree, and nothing
    but the rest of the line may follow the last one.
    """
    value = target["value"]
    declared = [(m.end(), m[1] or m[2] or m[3]) for m in _CHOICE_DECLARATION.finditer(final)]
    if target.get("pattern"):
        try:
            matches = list(regex.finditer(target["pattern"], final, timeout=_FORMAT_SEARCH_TIMEOUT_SECONDS))
        except TimeoutError as error:
            raise FormattingVerifierTimeout("MCQA answer pattern exceeded its search budget") from error
        # Row patterns are loose ("Response:\\s*([A-Za-z0-9])"): keep only a displayed
        # uppercase choice that does not start a word or an HTML tag such as <b>.
        declared += [
            (m.end(), m[1])
            for m in matches
            if m[1] in target["choices"] and not final[m.end(1) : m.end(1) + 1].isalnum()
        ]
    if declared:
        if {letter for _, letter in declared} != {value}:
            return 0.0
        # The rest of that line may gloss the choice; later lines may only close markup.
        return float(re.fullmatch(r"[^\n]*[\s\\\])$*]*", final[max(end for end, _ in declared) :]) is not None)
    return float(re.fullmatch(r"\**\(?" + re.escape(value) + r"\)?\**[.\s]*", final) is not None)


def score(text, target):
    kind = validate_target(target)  # Broken targets must not become model failures.
    final = answer_text(text)
    value = target["value"]
    if kind == "mcqa_final_choice":
        return _final_choice_score(final, target)
    if kind == "mcqa":
        # Preparation appends this exact contract; never search reasoning for a letter.
        match = re.fullmatch(r"Answer:\s*([A-Z])\s*", final.split("\n")[-1])
        return float(match is not None and match[1] == value)
    if kind == "format":
        return _format_score(final, value)
    if kind == "countdown":
        boxes = re.findall(r"\\boxed\{([^{}]+)\}", final)
        expressions = boxes[-1:] or final.split("\n")[-1:]
        return float(any(countdown_value(e.strip(), target["nums"]) == value for e in expressions))
    if kind == "knights":
        answer = final.rsplit("<answer>", 1)[-1].split("</answer>", 1)[0]
        expected = {k.casefold(): v for k, v in value.items()}
        found = re.findall(r"\b([A-Za-z]+)\s+is\s+a\s+(knight|knave)\b", answer, re.I)
        if len(found) != len(expected):
            return 0.0
        assignments = {name.casefold(): role.casefold() == "knight" for name, role in found}
        return float(assignments == expected)
    if kind == "numeric":
        boxes = re.findall(r"\\boxed\{([^{}]+)\}", final)
        text = boxes[-1] if len(boxes) == 1 else final.split("\n")[-1]
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text.strip()) is None:
            return 0.0
        return float(Decimal(text.strip()) == Decimal(value))
    if kind == "exact_text":
        answers = re.findall(r"<answer>(.*?)</answer>", final, re.S)
        if len(answers) > 1:
            return 0.0
        final = answers[0].strip() if answers else final
        return float(final.casefold() == value.strip().casefold())
    try:
        return float(typed_equal(strict_json(final), value))
    except (ValueError, RecursionError):
        return 0.0


class DeterministicVerifier:
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        pass

    async def async_call(self, tokens, prediction, label, **kwargs):
        start = time.monotonic()
        try:
            result = score(prediction, label)
        except FormattingVerifierTimeout as error:
            logger.warning("Formatting verifier timeout; assigning fallback zero reward: %s", error)
            return Result(
                0.0,
                diagnostics={
                    "kind": label["kind"],
                    "status": "timeout",
                    "error": str(error),
                    "failure_policy": "zero",
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
        if not math.isfinite(result):
            raise ValueError("Nonfinite deterministic reward")
        return Result(result)
