# Copyright 2026 Allen Institute for Artificial Intelligence
# SPDX-License-Identifier: Apache-2.0
"""Native XML parser reused from the ToolRL integration (5732f2ef5).

Only parsing is shared; source-specific reward semantics belong to each verifier.
"""

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

from open_instruct.miles.rewards import structured_tool_calls


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


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
    schemas = {tool["function"]["name"]: tool["function"]["parameters"].get("properties", {}) for tool in tools}
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


def _boundary(text, delimiter):
    """Locate a delimiter outside literal strings and containers."""
    stack, quote, escaped = [], None, False
    pairs = {"[": "]", "{": "}", "(": ")"}
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == delimiter and not stack:
            return index
        elif char in ("'", '"'):
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "]})" and (not stack or stack.pop() != char):
            raise ValueError("Unbalanced call arguments")
    if stack or quote:
        raise ValueError("Unclosed call arguments")
    return len(text)


def parse_pythonic_calls(body: str, tools: Sequence[dict[str, Any]] = ()):
    """Read the template's literal call notation, including non-Python tool names.

    Names are matched against declarations before looking for parentheses: some
    ToolRL names themselves contain spaces, digits or parentheses. Values use a
    restricted literal parser, never Python evaluation or attribute resolution.
    """
    match = re.fullmatch(r"\s*<function_calls>(.*?)</function_calls>\s*", body, re.S)
    if match is None:
        raise ValueError("Expected a complete Pythonic function-call block")
    names: list[str] = [t["function"]["name"] for t in tools]
    names.sort(key=len, reverse=True)
    remaining, calls = match[1].strip(), []
    while remaining:
        name = next((n for n in names if remaining.startswith(n + "(")), None)
        if name is None:
            opening = remaining.find("(")
            if opening < 1 or not remaining[:opening].isidentifier():
                raise ValueError("Expected a declared tool name or function identifier")
            name = remaining[:opening]
        content = remaining[len(name) + 1 :]
        closing = _boundary(content, ")")
        if closing == len(content):
            raise ValueError("Unclosed function call")
        args, arguments = content[:closing].strip(), {}
        while args:
            end = _boundary(args, ",")
            argument = args[:end].strip()
            if argument.startswith("**"):
                values = structured_tool_calls._literal(ast.parse(argument[2:], mode="eval").body)
            else:
                key, separator, value = argument.partition("=")
                key = key.strip()
                if not separator or not key or any(c in key for c in "()[]{}<>\n"):
                    raise ValueError("Expected named literal arguments")
                values = {key: structured_tool_calls._literal(ast.parse(value.strip(), mode="eval").body)}
            if not isinstance(values, dict) or any(k in arguments for k in values):
                raise ValueError("Duplicate or malformed arguments")
            arguments.update(values)
            args = args[end + 1 :].strip()
        json.dumps(arguments, allow_nan=False)
        calls.append({"name": name, "parameters": arguments})
        remaining = content[closing + 1 :].strip()
        if remaining.startswith(";"):
            remaining = remaining[1:].strip()
    if not calls:
        raise ValueError("Empty function-call block")
    return calls
