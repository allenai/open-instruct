"""Preserve Responses tool histories as chat-template messages for offline next-action tasks.

Uses the same assistant tool_calls / tool-observation representation as the
ToolRL preparation integration (5732f2ef5); no tools are executed during training.
"""

import copy
from typing import Any, cast

from open_instruct.miles.rewards import nemo_tool_calls


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(p, dict)
        and p.get("type") in {"text", "input_text", "output_text"}
        and isinstance(p.get("text"), str)
        for p in value
    ):
        return "\n".join(p["text"] for p in value)
    raise ValueError("Tool history requires text content")


def messages(raw):
    params = raw["responses_create_params"]
    value = params.get("input")
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError("Tool history requires a nonempty conversation")
    result, calls = [], {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Invalid tool history item")
        kind = item.get("type", "message")
        if kind == "reasoning":
            summary = item.get("summary", [])
            if not isinstance(summary, list) or any(
                not isinstance(part, dict)
                or part.get("type") != "summary_text"
                or not isinstance(part.get("text"), str)
                for part in summary
            ):
                raise ValueError("Unsupported reasoning history content")
            if item.get("encrypted_content"):
                raise ValueError("Encrypted reasoning history cannot be rendered")
            result.append(
                {"role": "assistant", "content": "", "reasoning_content": "\n".join(part["text"] for part in summary)}
            )
        elif kind == "function_call":
            identity = item.get("call_id")
            if not isinstance(identity, str) or not identity or identity in calls:
                raise ValueError("History calls require distinct call IDs")
            calls[identity] = item["name"]
            call = {
                "id": identity,
                "type": "function",
                "function": {"name": item["name"], "arguments": nemo_tool_calls.json_object(item["arguments"])},
            }
            if result and result[-1]["role"] == "assistant":
                result[-1].setdefault("tool_calls", []).append(call)
            else:
                result.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            identity = item.get("call_id")
            if identity not in calls:
                raise ValueError("Tool observation lacks a preceding call")
            result.append(
                {
                    "role": "tool",
                    "content": text_content(item["output"]),
                    "tool_call_id": identity,
                    "name": calls[identity],
                }
            )
        elif kind == "message":
            role = item.get("role")
            if role not in {"user", "assistant", "system", "developer", "tool"}:
                raise ValueError("Unsupported tool-history role")
            message: dict[str, Any] = {"role": role, "content": text_content(item.get("content") or "")}
            if item.get("tool_calls"):
                if role != "assistant":
                    raise ValueError("Only assistant history may contain calls")
                history_calls = cast(list[dict[str, Any]], copy.deepcopy(item["tool_calls"]))
                message["tool_calls"] = history_calls
                for call in history_calls:
                    function = call["function"]
                    function["arguments"] = nemo_tool_calls.json_object(function["arguments"])
                    if call.get("id"):
                        if call["id"] in calls:
                            raise ValueError("Repeated history call ID")
                        calls[call["id"]] = function["name"]
            if role == "tool":
                identity = item.get("tool_call_id")
                if identity not in calls:
                    raise ValueError("Tool observation lacks a preceding call")
                message.update(tool_call_id=identity, name=calls[identity])
            if (
                role == "assistant"
                and result
                and result[-1]["role"] == "assistant"
                and "reasoning_content" in result[-1]
                and not result[-1]["content"]
                and not result[-1].get("tool_calls")
            ):
                result[-1].update(message)
            else:
                result.append(message)
        else:
            raise ValueError(f"Unsupported tool history item: {kind}")
    if result[-1]["role"] not in {"user", "tool"}:
        raise ValueError("Tool history must end at a user or tool observation")
    context = []
    for text in (raw.get("system_prompt"), params.get("instructions")):
        if text is not None and not isinstance(text, str):
            raise ValueError("System instructions must be text")
        if text:
            context.append(text)
    while result and result[0]["role"] in {"system", "developer"}:
        context.append(result.pop(0)["content"])
    if any(m["role"] in {"system", "developer"} for m in result):
        raise ValueError("Tool history requires leading system/developer context")
    if context:
        result.insert(0, {"role": "system", "content": "\n\n".join(context)})
    return result


def for_template(messages, template):
    """Older Olmo templates consume reasoning as tagged assistant content."""
    rendered_messages = copy.deepcopy(messages)
    if "reasoning_content" not in template:
        for message in rendered_messages:
            reasoning = message.pop("reasoning_content", None)
            if reasoning:
                message["content"] = "<think>" + reasoning + "</think>\n" + message["content"]
    return rendered_messages
