"""Frozen-input and placement contracts for the historical framework control.

Run in its original image, with /stage before this checkout on PYTHONPATH.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
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
    assert launch_original_baseline.specification("image", "source", "prepare", "test")["budget"] == "ai2/oe-other"
    cpu = launch_original_baseline.specification("image", "source", "prepare", "test")["tasks"][0]
    assert cpu["resources"]["gpuCount"] == 0
    assert cpu["constraints"] == {"cluster": ["ai2/saturn"]}
    assert not any(e["name"] == "WANDB_API_KEY" for e in cpu["envVars"])
    gpu = launch_original_baseline.specification("image", "source", "smoke", "test")["tasks"][0]
    assert gpu["resources"]["gpuCount"] == 6
    assert gpu["constraints"] == {"cluster": ["ai2/holmes"]}
    assert "original_baseline.py smoke" in gpu["arguments"][0]
    assert "open_instruct.miles train" not in gpu["arguments"][0]


def test_jsonl_preserves_unicode_line_separators_inside_prompts(tmp_path):
    samples = [{"input": "question\u2028continued\u0085next\u2029paragraph"}, {"input": "second"}]
    path = tmp_path / "train.jsonl"
    path.write_bytes(b"".join(original_baseline.encoded(sample) for sample in samples))
    assert original_baseline.read_jsonl(path) == samples


def test_original_image_adjustments_preserve_loop_and_schedule_initial_eval():
    source = Path("/stage/open_instruct/grpo_fast.py").read_text()
    assert original_baseline.sha(source.encode()) == original_baseline.ORIGINAL_TRAINER_SHA256
    patched, changes = original_baseline.patch_trainer(source)
    tree = ast.parse(patched)
    loop = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_training")
    condition = next(
        n.test for n in ast.walk(loop) if isinstance(n, ast.If) and "local_eval_every" in ast.unparse(n.test)
    )
    expression = compile(ast.Expression(condition), "eval_schedule", "eval")
    for interval, expected in [(50, [1, 50, 100, 150, 200]), (1, [1, 2, 3])]:
        args = SimpleNamespace(local_eval_every=interval, eval_on_step_0=True)
        scheduled = [
            step
            for step in range(1, expected[-1] + 1)
            if eval(expression, {"args": args, "training_step": step, "eval_batch": object()})
        ]
        assert scheduled == expected
    for change in reversed(list(changes.values())):
        patched = patched.replace(change["after"], change["before"])
    assert patched == source


def test_model_alias_changes_only_metadata_and_links_unchanged_tensors(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "olmo3",
        "architectures": ["Olmo3ForCausalLM"],
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "sliding_window": 5,
        "rope_scaling": {"rope_type": "yarn", "factor": 2},
    }
    (source / "config.json").write_text(json.dumps(config))
    tensors = {
        "model.layers.0.self_attn.q_norm.weight": torch.ones(16),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(8),
    }
    save_file(tensors, source / "model.safetensors")
    alias = tmp_path / "alias"
    receipt = original_baseline.legacy_model(source, alias)
    assert (alias / "model.safetensors").resolve() == source / "model.safetensors"
    changed = json.loads((alias / "config.json").read_text())
    assert {key for key in changed if changed[key] != config[key]} == {"model_type", "architectures"}
    assert receipt["norm_shapes"] == {key: list(value.shape) for key, value in tensors.items()}
    assert json.loads((source / "config.json").read_text()) == config
    tensors["model.layers.0.self_attn.q_norm.weight"] = torch.ones(4)
    save_file(tensors, source / "model.safetensors")
    with pytest.raises(ValueError, match="global-normalization"):
        original_baseline.legacy_model(source, tmp_path / "bad-alias")


def test_public_export_restores_model_class_and_original_chat_template(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "chat_template.jinja").write_text("public chat template")
    output = tmp_path / "run"
    exported = output / "model" / "final"
    exported.mkdir(parents=True)
    (exported / "config.json").write_text(
        json.dumps({"model_type": "olmo2-retrofit", "architectures": ["Olmo2RetrofitForCausalLM"]})
    )
    save_file({"weight": torch.ones(4)}, exported / "model.safetensors")
    before = original_baseline.sha((exported / "model.safetensors").read_bytes())
    assert original_baseline.restore_public_exports(output, source) == [str(exported)]
    assert json.loads((exported / "config.json").read_text())["model_type"] == "olmo3"
    assert (exported / "chat_template.jinja").read_text() == "public chat template"
    assert original_baseline.sha((exported / "model.safetensors").read_bytes()) == before
