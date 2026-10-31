"""Local contracts for explicitly prepared candidate slices.

These factories accept data, never Python supplied by a dataset. Preparation must
validate targets and positive/negative controls before publishing a source.
Formatting rewards measure formatting only, not factual answer correctness.
"""

import ast
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

import regex

from open_instruct.miles.rewards import task_verifiers


@dataclass
class Result:
    score: float
    cost: float = 0.0
    diagnostics: dict = field(default_factory=lambda: {"kind": "candidate_local_v1", "status": "ok"})


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

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


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
        "format",
        "countdown",
        "knights",
        "numeric",
        "exact_text",
        "json_exact",
    }:
        raise ValueError("Unsupported candidate target")
    kind = target["kind"]
    value = target.get("value")
    if kind == "mcqa":
        choices = target.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) < 2
            or len(set(choices)) != len(choices)
            or any(not isinstance(c, str) or re.fullmatch("[A-Z]", c) is None for c in choices)
            or value not in choices
        ):
            raise ValueError("MCQA requires distinct displayed choice letters and one correct letter")
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
            re.compile(pattern)
            regex.compile(pattern)
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


def score(text, target):
    kind = validate_target(target)  # Broken targets must not become model failures.
    final = answer_text(text)
    value = target["value"]
    if kind == "mcqa":
        # Preparation appends this exact contract; never search reasoning for a letter.
        match = re.fullmatch(r"Answer:\s*([A-Z])\s*", final.split("\n")[-1])
        return float(match is not None and match[1] == value)
    if kind == "format":
        if value["type"] == "string_match":
            expected = set(value["expected_markers"])
            return float(
                all(m in final for m in expected)
                and not any(
                    match.group() not in expected
                    for pattern in value.get("patterns", [])
                    for match in regex.finditer(pattern, final, timeout=0.1)
                )
            )
        matches = sum(
            any(regex.search(p, line, timeout=0.1) for p in value["verify_regex"]) for line in final.split("\n")
        )
        return float(matches >= value.get("verify_min_matches", 1))
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


class CandidateVerifier:
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        pass

    async def async_call(self, tokens, prediction, label, **kwargs):
        result = score(prediction, label)
        if not math.isfinite(result):
            raise ValueError("Nonfinite candidate reward")
        return Result(result)
