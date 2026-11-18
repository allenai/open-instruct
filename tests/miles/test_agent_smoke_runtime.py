"""The live probe keeps model-sampled tool calls and observations in one session."""

import asyncio
import copy
import json
from types import SimpleNamespace

from scripts.miles import agent_smoke


class Message:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, **kwargs):
        result = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [
                {"id": "call1", "type": "function", "function": {"name": "read_value", "arguments": "{}"}}
            ]
        return result


def test_probe_preserves_tool_history_and_records_both_calls(monkeypatch, tmp_path):
    requests = []
    call = SimpleNamespace(id="call1", function=SimpleNamespace(name="read_value"))
    replies = [
        SimpleNamespace(message=Message("", [call]), finish_reason="tool_calls"),
        SimpleNamespace(message=Message("42"), finish_reason="stop"),
    ]

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["base_url"] == "http://session/v1"
            self.chat = SimpleNamespace(completions=self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, *, model, messages, tools, max_tokens, extra_body, tool_choice=None):
            requests.append(
                copy.deepcopy(
                    dict(
                        model=model,
                        messages=messages,
                        tools=tools,
                        max_tokens=max_tokens,
                        extra_body=extra_body,
                        tool_choice=tool_choice,
                    )
                )
            )
            return SimpleNamespace(choices=[replies[len(requests) - 1]])

    monkeypatch.setattr(agent_smoke, "AsyncOpenAI", Client)
    monkeypatch.setenv("OI_MILES_AGENT_SMOKE_TRACE_DIR", str(tmp_path))
    result = asyncio.run(
        agent_smoke.run(
            "http://session",
            [{"role": "user", "content": "Read the value"}],
            {"max_tokens": 256, "temperature": 1.0, "no_stop_trim": True, "skip_special_tokens": False},
            {"task_id": "task-1", "value": 42},
        )
    )
    assert requests[1]["messages"][-1] == {"role": "tool", "tool_call_id": "call1", "content": "42"}
    assert requests[1]["messages"][-2]["tool_calls"][0]["id"] == "call1"
    assert requests[1]["tool_choice"] == "none"
    assert requests[1]["max_tokens"] == 256
    assert requests[1]["extra_body"]["no_stop_trim"] is True
    assert requests[1]["extra_body"]["skip_special_tokens"] is False
    assert result["reward"] == 1
    assert result["agent_metrics"]["turns"] == 2
    assert result["agent_metrics"]["tool_calls"] == 1
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["task_id"] == "task-1"
