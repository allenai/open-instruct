"""Render prepared MILES tool prompts with the same rules used by dataset producers."""

import json

from open_instruct import toolrl_utils
from open_instruct.miles.datasets import tool_conversations
from open_instruct.miles.rewards import nemo_tool_calls, structured_tool_calls

OFFLINE_FORMATS = {"olmo_pythonic_v1": "<function_calls>", "olmo_xml_v1": "<tool_call>"}


def render_tool_prompt(messages, tools, target, tokenizer):
    """Render saved function declarations and validate the target's existing protocol."""
    template = tokenizer.chat_template
    if target.get("response_mode") == "tool_call":
        marker = structured_tool_calls.FORMATS.get(target.get("tool_call_format"))
        if marker is None or marker not in template:
            raise ValueError("Structured tool answer format does not match the selected chat template")
        validate_structured_tools(tools, target)
    else:
        marker = OFFLINE_FORMATS.get(target.get("tool_format"))
        if marker is None or marker not in template:
            raise ValueError("Tool target format does not match the selected chat template")
        if not isinstance(tools, list) or tools != target.get("tools"):
            raise ValueError("Offline tool declarations differ from the verifier target")
        reference_roundtrip(target, tokenizer)
        messages = tool_conversations.for_template(messages, template)

    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    if "calls" in target and "tool_format" in target:
        reasoning_prefix(text)
    return text


def validate_structured_tools(tools, target):
    if not isinstance(tools, list) or not tools:
        raise ValueError("Structured tool answers require saved function declarations")
    names = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Structured tool answers require function declarations")
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("parameters"), dict):
            raise ValueError("Structured tool answers require function schemas")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Structured answer functions require distinct nonempty names")
        names.append(name)
    if target.get("tool_name") not in names:
        raise ValueError("Structured answer function is absent from the template declarations")


def reasoning_prefix(text):
    if not isinstance(text, str) or not text:
        raise ValueError("ToolRL requires a rendered prompt")
    if text.rstrip().endswith("<think>") and not text.endswith("<think>"):
        raise ValueError("ToolRL requires an exact reasoning prefix without trailing whitespace")
    return text.endswith("<think>")


def reference_roundtrip(target, tokenizer):
    """A known reference must retain full reward through the selected template."""
    if "expected_action" in target:
        native = nemo_tool_calls
        _, expected, config = native.contract(json.dumps(target, allow_nan=False))
        calls = [
            {"name": call.name, "parameters": native.json_object(call.arguments)}
            for call in native.native.get_tool_calls(expected)
        ]
    else:
        calls = target.get("calls", [])
    if not calls:
        return
    messages = [{"role": "user", "content": "Reference serialization check."}]
    options = {"tools": target["tools"], "tokenize": False, "add_generation_prompt": False}
    prefix = tokenizer.apply_chat_template(messages, **options)
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": call["name"], "arguments": call["parameters"]}} for call in calls
        ],
    }
    rendered = tokenizer.apply_chat_template([*messages, assistant], **options)
    if not rendered.startswith(prefix):
        raise ValueError("Tool template does not preserve its conversation prefix")
    tail = rendered[len(prefix) :]
    marker = OFFLINE_FORMATS[target["tool_format"]]
    start = tail.find(marker)
    if start < 0:
        raise ValueError("Tool template omitted the reference call")
    action = tail[start:]
    try:
        if "expected_action" in target:
            actual = native.parse_action(action, target)
            valid = native.native.ActionComparator(config=config).compare_action(expected, actual).reward == 1
        else:
            valid = toolrl_utils.score_response("</think>" + action, target, think_prefix_in_prompt=True).score == 4
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError("Reference call cannot be parsed after template rendering") from error
    if not valid:
        raise ValueError("Reference call loses reward after template rendering")
