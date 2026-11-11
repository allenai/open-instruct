"""Non-executing structured tool answers in declared Olmo call formats."""

import copy
import json
import keyword

from open_instruct.miles.rewards import tool_call_parsing

FORMATS = {"olmo3_pythonic_v1": "<function_calls>", "olmo_xml_v1": "<tool_call>"}


def validate_contract(target):
    if target.get("response_mode", "text") == "text":
        if any(key in target for key in ("tool_name", "tool_payload_key", "tool_call_format")):
            raise ValueError("Tool fields require response_mode=tool_call")
        return
    if target["response_mode"] != "tool_call" or target.get("format") != "json":
        raise ValueError("Structured tool calls require response_mode=tool_call and format=json")
    if target.get("tool_call_format") not in FORMATS:
        raise ValueError("Specify tool_call_format=olmo3_pythonic_v1 or olmo_xml_v1")
    name = target.get("tool_name")
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError("Expected an Olmo-compatible tool_name")
    key = target.get("tool_payload_key")
    if key is not None and (not isinstance(key, str) or not key):
        raise ValueError("tool_payload_key must be nonempty text or null")


def tools_for_template(raw):
    """Convert Responses function declarations to the chat template's tool shape."""
    params = raw.get("responses_create_params") or {}
    tools = params.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Structured tool answers require function declarations")
    result, names = [], set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Only non-executing function answer declarations are supported")
        function = copy.deepcopy({k: v for k, v in tool.items() if k != "type"})
        name = function.get("name")
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name in names:
            raise ValueError("Function names must be distinct Olmo-compatible identifiers")
        names.add(name)
        parameters = function.get("parameters")
        if isinstance(parameters, str):
            parameters = json.loads(parameters)
        if not isinstance(parameters, dict):
            raise ValueError("Function parameters must be a schema object")
        function["parameters"] = parameters
        result.append({"type": "function", "function": function})
    if raw.get("tool_name") not in names:
        raise ValueError("Expected tool_name is absent from the declarations")
    return result


def parse_payload(text, target):
    key = target.get("tool_payload_key")
    tools = [{"type": "function", "function": {"name": target["tool_name"], "parameters": {}}}]
    if target["tool_call_format"] == "olmo_xml_v1":
        # Parse the wrapper first, then decode each value in the original schema's
        # reference scope. Wrapping a payload schema would move its local $refs.
        calls = tool_call_parsing.parse_native_calls(text, tools)
        for call in calls:
            call["parameters"] = {
                name: tool_call_parsing.parameter_value(
                    value,
                    target["schema"]
                    if key is not None and name == key
                    else target["schema"].get("properties", {}).get(name, {}),
                    target["schema"],
                )
                for name, value in call["parameters"].items()
            }
    else:
        calls = tool_call_parsing.parse_pythonic_calls(text, tools)
    if len(calls) != 1 or calls[0]["name"] != target["tool_name"]:
        raise ValueError("Expected exactly one final call to the declared function")
    arguments = calls[0]["parameters"]
    if key is not None:
        if key not in arguments:
            raise ValueError("Missing tool payload key")
        arguments = arguments[key]
    json.dumps(arguments, allow_nan=False)
    return arguments
