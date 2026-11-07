"""Non-executing structured tool answers in the Olmo Pythonic call format."""

import ast
import copy
import json
import keyword
import re


def validate_contract(target):
    if target.get("response_mode", "text") == "text":
        if any(key in target for key in ("tool_name", "tool_payload_key", "tool_call_format")):
            raise ValueError("Tool fields require response_mode=tool_call")
        return
    if target["response_mode"] != "tool_call" or target.get("format") != "json":
        raise ValueError("Structured tool calls require response_mode=tool_call and format=json")
    if target.get("tool_call_format") != "olmo3_pythonic_v1":
        raise ValueError("Specify tool_call_format=olmo3_pythonic_v1")
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


def _literal(node):
    # The template emits JSON literals inside Pythonic keyword arguments. Accept
    # both JSON and Python spellings, without evaluating expressions or calls.
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return node.value
    if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, ast.List):
        return [_literal(value) for value in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values, strict=True):
            key = _literal(key)
            if not isinstance(key, str) or key in result:
                raise ValueError("Tool arguments require distinct string keys")
            result[key] = _literal(value)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        if type(value) in (int, float):
            return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("Tool arguments must contain JSON values only")


def parse_payload(text, target):
    match = re.fullmatch(r"\s*<function_calls>\s*(.*?)\s*</function_calls>\s*", text, re.S)
    if match is None:
        raise ValueError("Expected exactly one final function call")
    try:
        call = ast.parse(match[1], mode="eval").body
    except SyntaxError as error:
        raise ValueError("Malformed final function call") from error
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
        raise ValueError("Expected a function name with keyword arguments")
    if call.func.id != target["tool_name"]:
        raise ValueError("Wrong final function name")
    arguments = {}
    for arg in call.keywords:
        # Literal dictionary expansion represents schema fields such as 'class',
        # which cannot be written as Python keyword arguments. Nothing is executed.
        values = _literal(arg.value) if arg.arg is None else {arg.arg: _literal(arg.value)}
        if not isinstance(values, dict) or any(key in arguments for key in values):
            raise ValueError("Tool arguments must have distinct string keys")
        arguments.update(values)
    key = target.get("tool_payload_key")
    if key is not None:
        if key not in arguments:
            raise ValueError("Missing tool payload key")
        arguments = arguments[key]
    json.dumps(arguments, allow_nan=False)
    return arguments
