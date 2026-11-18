"""Agent runs preserve raw prompts and reject unsupported refresh contracts."""

import asyncio
import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import constraints
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.datasets import run_data
from open_instruct.miles.rollout import agent
from open_instruct.test_miles_policy_refresh import configured


def agent_config():
    config = configured()
    return dataclasses.replace(
        config,
        miles=config.miles
        | dict(
            custom_generate_function_path=constraints.AGENT_GENERATOR,
            custom_agent_function_path="tmax_agent_function.run",
            use_session_server="v2",
            tito_model="olmo35",
            pause_generation_mode="in_place",
            sglang_disable_radix_cache=True,
            max_seq_len=config.core.max_sequence_length,
        ),
    )


def test_agent_refresh_compiles_native_generator_arguments():
    argv = agent_config().arguments()
    assert argv[argv.index("--custom-agent-function-path") + 1] == "tmax_agent_function.run"
    assert argv[argv.index("--tito-model") + 1] == "olmo35"
    assert "--max-seq-len" in argv


@pytest.mark.parametrize(
    "change,match",
    [
        ({"custom_generate_function_path": "arbitrary.generate"}, "single-turn"),
        ({"use_session_server": "v1"}, "use_session_server=v2"),
        ({"tito_model": "qwen35"}, "tito_model=olmo35"),
        ({"custom_agent_function_path": ""}, "custom_agent_function_path"),
        ({"apply_chat_template": True}, "unrendered"),
        ({"max_seq_len": 10**8}, "max_seq_len"),
        ({"session_sample_picker_path": "custom.pick"}, "session_sample_picker_path"),
        ({"use_rollout_logprobs": True}, "use_rollout_logprobs"),
    ],
)
def test_agent_refresh_rejects_unsupported_contracts(change, match):
    config = agent_config()
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(config, miles=config.miles | change).validate()


def test_engine_drain_still_rejects_agent_generators():
    config = agent_config()
    with pytest.raises(ValueError, match="single-turn"):
        dataclasses.replace(config, core=dataclasses.replace(config.core, publication_mode="engine_drain")).validate()


def test_structured_agent_run_selects_native_sessions_and_episode_rewards(tmp_path):
    spec = RunSpec.from_dict(
        dict(
            schema_version=1,
            name="agent",
            model={"source": "tiny"},
            output={"root": str(tmp_path / "run")},
            data={"agentic": True, "prompt_data": "episodes.jsonl"},
            inference={"placement_mode": "disaggregated", "gpus": 1},
            **{"async": {"fully_async": True}},
            core={"publication_mode": "refresh"},
            miles={
                "custom_generate_function_path": constraints.AGENT_GENERATOR,
                "custom_agent_function_path": "tmax_agent_function.run",
            },
        ),
        config_path=tmp_path / "agent.toml",
    )
    config = spec.compile()
    assert config.miles["use_session_server"] == "v2"
    assert config.miles["tito_model"] == "olmo35"
    assert config.miles["max_seq_len"] == config.core.max_sequence_length
    assert config.miles["custom_rm_path"] == "open_instruct.miles.rollout.agent.episode_reward"
    assert not config.miles.get("apply_chat_template")
    assert "label_key" not in config.miles
    assert "--label-key" not in config.arguments()


class AgentTokenizer:
    chat_template = "native Olmo template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and return_dict is False
        return list(range(len(messages[-1]["content"])))


def test_agent_preparation_preserves_messages_metadata_and_resume(tmp_path, monkeypatch):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr(run_data, "_tokenizer", lambda path: AgentTokenizer())
    row = dict(
        input=[{"role": "user", "content": "inspect sandbox"}],
        metadata={"task_id": "task-1", "task_dir": "/tasks/task-1"},
    )
    original = copy.deepcopy(row)
    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps(row) + "\n")
    data = dict(prompt_data=str(source), agentic=True)
    output = tmp_path / "prepared"
    result = run_data.prepare_data(data, hf, output, max_prompt_length=100, seed=17)
    prepared = json.loads((output / "train.jsonl").read_text())
    assert prepared["input"] == original["input"]
    assert prepared["metadata"]["task_id"] == "task-1"
    assert "verifiers" not in prepared["metadata"]
    assert prepared["metadata"]["run_prompt_tokens"] == len("inspect sandbox")
    assert run_data.prepare_data(data, hf, output, max_prompt_length=100, seed=17) == result
    source.write_text(json.dumps(row | {"metadata": {"task_id": "different"}}) + "\n")
    with pytest.raises(ValueError, match="source changed"):
        run_data.prepare_data(data, hf, output, max_prompt_length=100, seed=17)


def test_missing_episode_verdict_is_not_a_zero_reward():
    assert asyncio.run(agent.episode_reward(None, SimpleNamespace(metadata={}))) is None
    assert asyncio.run(agent.episode_reward(None, SimpleNamespace(metadata={"reward": 0.0}))) == 0.0
