"""The live probe keeps model-sampled tool calls and observations in one session."""

import asyncio
import copy
import json
import sys
from types import SimpleNamespace

from miles.utils import arguments as native_arguments
from miles.utils.data import Dataset
from scripts.miles import agent_smoke

from open_instruct.miles.configuration import arguments as core_arguments
from open_instruct.miles.configuration import constraints
from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec


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


def test_compiled_agent_data_loads_without_a_label_column(tmp_path):
    path = tmp_path / "episodes.jsonl"
    messages = [{"role": "user", "content": "Read the value"}]
    path.write_text(json.dumps({"input": messages, "metadata": {"task_id": "task-1"}}) + "\n")
    spec = RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "native-agent-data",
            "model": {"source": "tiny"},
            "output": {"root": str(tmp_path / "run")},
            "data": {"agentic": True, "prompt_data": str(path)},
            "inference": {"placement_mode": "disaggregated", "gpus": 1},
            "async": {"fully_async": True},
            "core": {"publication_mode": "refresh"},
            "miles": {
                "custom_generate_function_path": constraints.AGENT_GENERATOR,
                "custom_agent_function_path": "scripts.miles.agent_smoke.run",
            },
        },
        config_path=tmp_path / "run.json",
    )
    options = spec.compile().miles
    data = Dataset(
        options["prompt_data"],
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key=options["input_key"],
        label_key=options.get("label_key"),
        metadata_key=options["metadata_key"],
    )
    assert data[0].prompt == messages
    assert data[0].label is None


def test_real_native_parser_round_trip_preserves_core_agent_contract(monkeypatch):
    config = RunConfig(
        CoreConfig(publication_mode="refresh", max_policy_lag=2),
        {
            "hf_checkpoint": "/fixture/hf",
            "global_batch_size": 4,
            "rollout_batch_size": 2,
            "n_samples_per_prompt": 2,
            "fully_async": True,
            "use_tis": True,
            "sglang_cuda_graph_backend_decode": "disabled",
            "sglang_cuda_graph_backend_prefill": "disabled",
            "pause_generation_mode": "in_place",
            "sglang_disable_radix_cache": True,
            "custom_generate_function_path": constraints.AGENT_GENERATOR,
            "custom_agent_function_path": "scripts.miles.agent_smoke.run",
            "use_session_server": "v2",
            "tito_model": "olmo35",
            "max_seq_len": 4096,
        },
    )
    monkeypatch.setattr(sys, "argv", ["train", *config.arguments()])
    args = core_arguments.load_core_args(native_arguments.get_miles_extra_args_provider())
    assert args.pause_generation_mode == "in_place"
    assert args.use_session_server == "v2"
    assert args.tito_model == "olmo35"
    assert args.max_seq_len == 4096
    assert args.custom_agent_function_path == "scripts.miles.agent_smoke.run"
    assert args.custom_async_data_buffer_path.endswith("RefreshPolicyDataBuffer")
