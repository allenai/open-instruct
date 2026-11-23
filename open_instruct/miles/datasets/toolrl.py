"""Validate canonical ToolRL rows before rendering their next-action prompts."""

import copy
import json

from jsonschema import validators
from referencing import Registry
from referencing.exceptions import Unresolvable

from open_instruct import toolrl_utils


def adapt(raw, options):
    target = copy.deepcopy(raw["ground_truth"])
    target["tool_format"] = options.get("tool_format", "olmo_pythonic_v1")
    toolrl_utils.validate_target(target)
    if raw["tools"] != target["tools"]:
        raise ValueError("ToolRL prompt and target tools disagree")
    messages = copy.deepcopy(raw["messages"])
    if not isinstance(messages, list) or not messages:
        raise ValueError("ToolRL requires a nonempty conversation")
    checkers = {}
    for tool in target["tools"]:
        function = tool["function"]
        schema = function["parameters"]
        cls = validators.validator_for(schema)
        cls.check_schema(schema)
        # The default registry refuses retrieval with NoSuchResource; no network access.
        checkers[function["name"]] = cls(schema, registry=Registry())
    calls = list(target["calls"])
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) - {"role", "content", "reasoning_content", "tool_calls"}
            or message.get("role") not in {"system", "user", "assistant", "tool"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Unsupported ToolRL conversation fields")
        if "reasoning_content" in message and (
            message["role"] != "assistant" or not isinstance(message["reasoning_content"], str)
        ):
            raise ValueError("ToolRL reasoning must belong to an assistant")
        if "tool_calls" in message:
            if message["role"] != "assistant" or not isinstance(message["tool_calls"], list):
                raise ValueError("ToolRL calls must belong to an assistant")
            for call in message["tool_calls"]:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise ValueError("Malformed ToolRL history call")
                function = call["function"]
                calls.append({"name": function["name"], "parameters": function["arguments"]})
    if messages[-1]["role"] not in {"user", "tool"}:
        raise ValueError("ToolRL history must end with a user or tool observation")
    for call in calls:
        if call["name"] not in checkers or not isinstance(call["parameters"], dict):
            raise ValueError("Invalid ToolRL history or reference call")
        checker = checkers[call["name"]]
        if set(call["parameters"]) - set(checker.schema["properties"]):
            raise ValueError("ToolRL call uses undeclared arguments")
        try:
            error = next(checker.iter_errors(call["parameters"]), None)
        except Unresolvable as error:
            raise ValueError("ToolRL schema references an unavailable resource") from error
        if error is not None:
            raise ValueError("ToolRL call violates its schema: " + error.message)
    json.dumps(target, allow_nan=False)
    return messages, target, "toolrl", "toolrl_v1:" + ("calls" if target["calls"] else "format_only")
