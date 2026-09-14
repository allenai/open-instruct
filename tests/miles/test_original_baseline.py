"""Frozen-input and placement contracts for the historical framework control.

Run in its original image, with /stage before this checkout on PYTHONPATH.
"""

import pytest
from scripts.miles import launch_original_baseline, original_baseline


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "already rendered prompt"}]
        return [10, 11]


def row():
    return {
        "input": "already rendered prompt",
        "metadata": {
            "prepared_sample_id": "gsm8k:train:42",
            "run_prompt_token_ids_sha256": original_baseline.sha(original_baseline.encoded([10, 11])),
            "verifiers": [{"name": "gsm8k", "target": "-12", "weight": 1.0}],
        },
    }


def test_original_transform_preserves_frozen_tokens_labels_and_identity():
    sample = row()
    converted = original_baseline.convert_row(sample, Tokenizer())
    assert converted["ground_truth"] == "-12"
    assert converted["dataset"] == "gsm8k"
    assert converted["prepared_sample_id"] == "gsm8k:train:42"
    assert sample["metadata"]["verifiers"][0]["target"] == "-12"


def test_changed_token_ids_fail_before_spending_gpus():
    sample = row()
    sample["metadata"]["run_prompt_token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed frozen prompt tokens"):
        original_baseline.convert_row(sample, Tokenizer())


@pytest.mark.parametrize("change", [{"name": "math"}, {"weight": 10}])
def test_reward_contract_changes_are_not_silently_adopted(change):
    sample = row()
    sample["metadata"]["verifiers"][0].update(change)
    with pytest.raises(ValueError, match="unit-weight GSM8K"):
        original_baseline.convert_row(sample, Tokenizer())


def test_cpu_weka_preparation_uses_saturn_and_training_is_the_original_backend():
    cpu = launch_original_baseline.specification("image", "source", "prepare", "test")["tasks"][0]
    assert cpu["resources"]["gpuCount"] == 0
    assert cpu["constraints"] == {"cluster": ["ai2/saturn"]}
    assert not any(e["name"] == "WANDB_API_KEY" for e in cpu["envVars"])
    gpu = launch_original_baseline.specification("image", "source", "smoke", "test")["tasks"][0]
    assert gpu["resources"]["gpuCount"] == 6
    assert gpu["constraints"] == {"cluster": ["ai2/holmes"]}
    assert "original_baseline.py smoke" in gpu["arguments"][0]
    assert "open_instruct.miles train" not in gpu["arguments"][0]
