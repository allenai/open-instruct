"""Deepscaler preparation and the standard 16x32 recipe."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.data.miles import prepare_deepscaler

from open_instruct.miles.configuration.run_spec import RunSpec


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return " ".join(message["content"] for message in messages)

    def __call__(self, prompt, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return SimpleNamespace(input_ids=prompt.split())


def test_preparation_excludes_reference_answers_and_filters_long_prompts(monkeypatch):
    data = [
        {
            "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "secret answer"}],
            "ground_truth": ["42"],
        },
        {"messages": [{"role": "user", "content": "too many words"}], "ground_truth": "0"},
    ]
    monkeypatch.setattr(prepare_deepscaler.datasets, "load_dataset", lambda *args, **kwargs: data)
    records = list(prepare_deepscaler.rows("train", "dataset", Tokenizer(), 2))
    assert len(records) == 1
    assert records[0]["input"] == "question"
    assert records[0]["label"] == "42"
    assert records[0]["metadata"]["prepared_sample_id"] == "train:0"
    assert records[0]["metadata"]["verifiers"] == [{"name": "math", "target": "42", "weight": 1.0}]


def test_preparation_rejects_ambiguous_targets(monkeypatch):
    data = [{"messages": [{"role": "user", "content": "question"}], "ground_truth": ["1", "2"]}]
    monkeypatch.setattr(prepare_deepscaler.datasets, "load_dataset", lambda *args, **kwargs: data)
    with pytest.raises(ValueError, match="expected one ground truth"):
        list(prepare_deepscaler.rows("train", "dataset", Tokenizer(), 2))


def test_deepscaler_recipe_uses_16_prompts_and_32_samples():
    path = Path(__file__).resolve().parents[1] / "configs/miles/benchmarks/qwen3_4b_deepscaler.toml"
    config = RunSpec.load(path).compile()
    assert config.miles["rollout_batch_size"] == 16
    assert config.miles["n_samples_per_prompt"] == 32
    assert config.miles["global_batch_size"] == 512
    assert config.miles["async_unused_samples_handler"] == "retry"
