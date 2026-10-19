"""Deepscaler preparation keeps targets out of prompts and recipes independent of NGU."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.miles import prepare_deepscaler

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


@pytest.mark.parametrize(
    "recipe", ["qwen3_4b_base_deepscaler_baseline", "qwen3_4b_base_deepscaler_baseline_k32n16", "smoke_saturn_fa2"]
)
def test_baseline_recipes_do_not_require_ngu(recipe):
    path = Path(__file__).resolve().parents[1] / "configs/ngu" / f"{recipe}.toml"
    config = RunSpec.load(path).compile()
    assert config.miles["async_unused_samples_handler"] == "retry"
    assert not any(key.startswith("ngu_") for key in config.miles)
