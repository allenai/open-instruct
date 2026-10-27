# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Allen Institute for Artificial Intelligence
# SPDX-License-Identifier: Apache-2.0
"""Offline ToolRL reward, adapted from qiancheng0/ToolRL at 8cee13ec0ca7.

Source: verl/utils/reward_score/rlla.py (Apache-2.0; see LICENSE).
Changes: dependency-free functions, explicit fixed default reward settings,
structured diagnostics, strict JSON validation, and scoring calls only after
the reasoning block. No tool execution, length bonus, or reward schedules.
"""

import dataclasses
import functools
import json
import re
from collections import Counter


@dataclasses.dataclass(frozen=True)
class ToolRLScore:
    score: float
    format_score: float
    correctness_score: float


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def parse_calls(body: str) -> list[dict]:
    """Read ToolRL's one-JSON-object-per-line call block, never execute it."""
    if body.count("<tool_call>") != 1 or body.count("</tool_call>") != 1:
        raise ValueError("Expected exactly one tool_call block")
    block = body.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    if not block:
        raise ValueError("Empty tool_call block")
    calls = []
    for line in block.splitlines():
        call = json.loads(line, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
        if (
            not isinstance(call, dict)
            or set(call) != {"name", "parameters"}
            or not isinstance(call["name"], str)
            or not call["name"]
            or not isinstance(call["parameters"], dict)
        ):
            raise ValueError("Calls require a nonempty name and a parameters object")
        calls.append(call)
    return calls


@functools.lru_cache(maxsize=8192)
def reference_parts(reference: str) -> tuple[str, list[dict]]:
    """Reject malformed targets as data errors, not model failures."""
    match = re.fullmatch(r"<think>.*?</think>\n(.*)", reference, re.DOTALL)
    if match is None:
        raise ValueError("ToolRL reference must contain a think block followed by a newline")
    body = match[1]
    if body.startswith("<tool_call>\n") and body.endswith("\n</tool_call>"):
        return "tool_call", parse_calls(body)
    if body.startswith("<response>") and body.endswith("</response>"):
        return "response", []
    raise ValueError("Unsupported ToolRL reference format")


def _overlap(expected, predicted):
    left, right = Counter(expected), Counter(predicted)
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    return intersection / union if union else 1.0


def _call_score(expected, predicted):
    # Preserve upstream's default multiset-name overlap and greedy per-call
    # parameter matching, including its argument comparison semantics.
    score = _overlap([call["name"] for call in expected], [call["name"] for call in predicted])
    possible = 1.0
    used = set()
    for target in expected:
        possible += 1 + len(target["parameters"])
        best_score, best_index = 0.0, None
        for index, call in enumerate(predicted):
            if index in used or call["name"] != target["name"]:
                continue
            parameters = target["parameters"]
            matched = _overlap(list(parameters), list(call["parameters"]))
            matched += sum(
                key in call["parameters"] and call["parameters"][key] == value for key, value in parameters.items()
            )
            if matched > best_score:
                best_score, best_index = matched, index
        if best_index is not None:
            used.add(best_index)
            score += best_score
    return -3.0 + 6.0 * score / possible


def score_response(prediction: str, reference: str, *, think_prefix_in_prompt: bool = False) -> ToolRLScore:
    """Return upstream's fixed default format (0..1) + call (-3..3) reward.

    No-call targets receive only a format reward; their answer text is not graded.
    Set think_prefix_in_prompt only when the generation prompt already ends with
    <think>. It is prepended literally, never inferred from a malformed response.
    """
    kind, expected = reference_parts(reference)
    if think_prefix_in_prompt:
        prediction = "<think>" + prediction
    match = re.fullmatch(r"<think>.*?</think>\n(.*)", prediction, re.DOTALL)
    body = match[1] if match else ""
    pattern = r"<tool_call>\n.*?\n</tool_call>" if kind == "tool_call" else r"<response>.*?</response>"
    formatted = (
        match is not None
        and prediction.count("<think>") == prediction.count("</think>") == 1
        and re.fullmatch(pattern, body, re.DOTALL) is not None
        and body.count(f"<{kind}>") == body.count(f"</{kind}>") == 1
    )
    format_score = float(formatted)
    correctness = 0.0
    if expected:
        # Missing reasoning tags need not erase a valid call's correctness,
        # but calls quoted inside reasoning must never count as actions.
        action = prediction.split("</think>", 1)[1] if "</think>" in prediction else prediction
        if "<think>" in action:
            action = ""
        try:
            correctness = _call_score(expected, parse_calls(action))
        except (ValueError, TypeError, RecursionError):
            correctness = -3.0
    return ToolRLScore(format_score + correctness, format_score, correctness)
