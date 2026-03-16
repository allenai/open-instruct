import pytest

from open_instruct.grpo_fast import create_generation_configs
from open_instruct.grpo_utils import ExperimentConfig, GRPOLossType
from open_instruct.vllm_utils import SamplingConfig


def test_local_eval_every_accepts_minus_one():
    cfg = ExperimentConfig(local_eval_every=-1)
    assert cfg.local_eval_every == -1


@pytest.mark.parametrize("value", [0, -2])
def test_local_eval_every_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="local_eval_every"):
        ExperimentConfig(local_eval_every=value)


@pytest.mark.parametrize("value", [0, -1, 3, 6])
def test_eval_pass_at_k_rejects_non_power_of_two_values(value):
    with pytest.raises(ValueError, match="eval_pass_at_k"):
        ExperimentConfig(eval_pass_at_k=value)


@pytest.mark.parametrize("value", [1, 2, 4, 8])
def test_eval_pass_at_k_accepts_power_of_two_values(value):
    cfg = ExperimentConfig(eval_pass_at_k=value)
    assert cfg.eval_pass_at_k == value


def test_tvpo_rejects_use_vllm_logprobs():
    with pytest.raises(ValueError, match="loss_fn=tvpo"):
        ExperimentConfig(loss_fn=GRPOLossType.tvpo, use_vllm_logprobs=True)


@pytest.mark.parametrize("value", [0, -2])
def test_eval_top_k_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="eval_top_k"):
        ExperimentConfig(eval_top_k=value)


@pytest.mark.parametrize("value", [-1, 1, 32])
def test_eval_top_k_accepts_valid_values(value):
    cfg = ExperimentConfig(eval_top_k=value)
    assert cfg.eval_top_k == value


def test_create_generation_configs_applies_eval_top_k_override(monkeypatch):
    args = ExperimentConfig(seed=123, eval_pass_at_k=2, eval_top_k=17)
    streaming_config = type(
        "StreamingConfigStub",
        (),
        {"temperature": 0.8, "response_length": 64, "num_samples_per_prompt_rollout": 4, "stop_strings": ["</s>"]},
    )()
    vllm_config = type("VLLMConfigStub", (), {"vllm_top_p": 0.9})()
    monkeypatch.setattr("open_instruct.grpo_fast.vllm_config", vllm_config, raising=False)

    generation_configs = create_generation_configs(args, streaming_config)

    assert generation_configs["train"] == SamplingConfig(
        temperature=0.8, top_p=0.9, top_k=-1, max_tokens=64, n=4, stop=["</s>"], seed=123, logprobs=1
    )
    assert generation_configs["eval"] == SamplingConfig(
        temperature=0.8, top_p=0.9, top_k=17, max_tokens=64, n=2, stop=["</s>"], seed=123, logprobs=1
    )
