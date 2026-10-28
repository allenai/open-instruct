# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Allen Institute for Artificial Intelligence
# SPDX-License-Identifier: Apache-2.0
"""Offline ToolRL reward, adapted from qiancheng0/ToolRL at 8cee13ec0ca7.

Source: verl/utils/reward_score/rlla.py (Apache-2.0; see LICENSE).
Changes: dependency-free functions, explicit fixed default reward settings,
structured diagnostics, native Olmo XML call parsing, canonical targets, and
scoring calls only after the reasoning block. No tool execution, length bonus, or reward schedules.
"""

import dataclasses
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


def validate_target(reference):
    """Versioned canonical targets prevent silently grading old JSON-format data."""
    if not isinstance(reference, dict) or reference.get("format") != "olmo-toolrl-v1":
        raise ValueError("ToolRL reference requires olmo-toolrl-v1; regenerate the manifest")
    if not isinstance(reference.get("calls"), list) or not isinstance(reference.get("tools"), list):
        raise ValueError("ToolRL reference requires calls and tools")
    names = set()
    for tool in reference["tools"]:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Malformed ToolRL reference function schema")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or not isinstance(parameters.get("properties"), dict):
            raise ValueError("Malformed ToolRL reference parameters schema")
        if function["name"] in names:
            raise ValueError("Duplicate ToolRL reference function schema")
        names.add(function["name"])
    for call in reference["calls"]:
        if (
            not isinstance(call, dict)
            or set(call) != {"name", "parameters"}
            or not isinstance(call["name"], str)
            or call["name"] not in names
            or not isinstance(call["parameters"], dict)
        ):
            raise ValueError("Malformed canonical ToolRL reference call")
    return reference["calls"]


def parameter_value(value, schema):
    """Decode native unquoted strings and JSON scalars/containers using the tool schema."""
    value = value.strip()
    kind = schema.get("type", "string")
    if kind == "string":
        return value
    decoded = json.loads(value, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    json.dumps(decoded, allow_nan=False)  # Reject overflow such as 1e999, including nested values.
    kinds = kind if isinstance(kind, list) else [kind]
    matches = {
        "null": decoded is None,
        "boolean": type(decoded) is bool,
        "integer": type(decoded) is int,
        "number": type(decoded) in (int, float),
        "array": isinstance(decoded, list),
        "object": isinstance(decoded, dict),
        "string": isinstance(decoded, str),
    }
    if not any(matches.get(item, False) for item in kinds):
        raise ValueError("Parameter value does not match its declared type")
    return decoded


def parse_native_calls(body, tools):
    """Parse complete Olmo XML-style calls without a vLLM/SGLang dependency.

    This is the offline reward's strict parser, not a permissive serving parser.
    All non-whitespace action text must belong to a call. Parallel calls have
    separate wrappers, and duplicate parameters or malformed tails fail closed.
    """
    schemas = {tool["function"]["name"]: tool["function"]["parameters"]["properties"] for tool in tools}
    calls = []
    remaining = body.strip()
    while remaining:
        match = re.match(r"<tool_call>\s*<function=([^<>]+)>(.*?)</function>\s*</tool_call>", remaining, re.DOTALL)
        if not match:
            raise ValueError("Expected a complete native function call")
        name, parameters = match[1], {}
        content = match[2].strip()
        while content:
            parameter = re.match(r"<parameter=([^<>]+)>(.*?)</parameter>", content, re.DOTALL)
            if not parameter or parameter[1] in parameters:
                raise ValueError("Malformed or duplicate parameter")
            key, value = parameter[1], parameter[2]
            if re.search(r"</?(?:tool_call|function|parameter)(?:[=>])", value):
                raise ValueError("Ambiguous structural tag in parameter")
            parameters[key] = parameter_value(value, schemas.get(name, {}).get(key, {}))
            content = content[parameter.end() :].strip()
        calls.append({"name": name, "parameters": parameters})
        remaining = remaining[match.end() :].strip()
    if not calls:
        raise ValueError("Empty tool call")
    return calls


def score_response(prediction: str, reference: dict, *, think_prefix_in_prompt: bool = False) -> ToolRLScore:
    """Native format credit plus upstream's fixed semantic call reward.

    No-call targets grade plain answer format only, never factual correctness.
    Whitespace after reasoning and between calls is not a reward signal.
    Serving may retain one native end-of-turn/EOS token after the response.
    """
    expected = validate_target(reference)
    # Strip only the terminal boundary, preserving tokens inside arguments and
    # malformed tails (including repeated terminators) for the strict parser.
    prediction = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", prediction, count=1)
    if think_prefix_in_prompt:
        prediction = "<think>" + prediction
    match = re.fullmatch(r"<think>.*?</think>\s*(.*)", prediction, re.DOTALL)
    formatted = match is not None and prediction.count("<think>") == prediction.count("</think>") == 1
    action = prediction.split("</think>", 1)[1].strip() if "</think>" in prediction else prediction.strip()
    if "<think>" in action:
        action = ""
    if expected:
        try:
            calls = parse_native_calls(action, reference["tools"])
            correctness = _call_score(expected, calls)
        except (ValueError, TypeError, RecursionError):
            formatted, correctness = False, -3.0
    else:
        formatted = (
            formatted and bool(action) and not re.search(r"</?(?:tool_call|function|parameter|response)\b", action)
        )
        correctness = 0.0
    format_score = float(formatted)
    return ToolRLScore(format_score + correctness, format_score, correctness)
