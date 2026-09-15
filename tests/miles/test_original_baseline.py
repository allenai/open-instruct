"""Frozen-input and placement contracts for the historical framework control.

Run in its original image, with /stage before this checkout on PYTHONPATH.
"""

import __future__

import ast
import json
import os
import queue
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from scripts.miles import launch_original_baseline, original_baseline
from transformers import GenerationConfig


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
    assert gpu["resources"]["gpuCount"] == 8
    assert gpu["constraints"] == {"cluster": ["ai2/jupiter"]}
    assert {e["name"]: e.get("value") for e in gpu["envVars"]}["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
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


def test_completion_does_not_confuse_driver_iterations_with_optimizer_updates():
    updates = [{"driver_step": 2}, {"driver_step": 5}]
    result = original_baseline.completion_record(5, updates, ["model"], {})
    assert result["driver_steps"] == 5
    assert result["completed_updates"] == 2
    with pytest.raises(RuntimeError, match="no optimizer updates"):
        original_baseline.completion_record(5, [], ["model"], {})


@pytest.mark.parametrize("steps", [[2, 2], [3, 2], [0], [6], [True]])
def test_invalid_optimizer_ledger_cannot_pass_the_completion_gate(steps):
    with pytest.raises(ValueError, match="Optimizer update ledger"):
        original_baseline.completion_record(5, [{"driver_step": step} for step in steps], ["model"], {})


def test_export_normalizes_the_unwrapped_models_generation_metadata(tmp_path):
    source, _ = original_baseline.patch_trainer(Path("/stage/open_instruct/grpo_fast.py").read_text())
    method = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "save_model")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.config = SimpleNamespace(tie_word_embeddings=False)
            self.generation_config = GenerationConfig(do_sample=False, temperature=0.6, top_p=0.95)

        def save_pretrained(self, output_dir, state_dict):
            self.generation_config.save_pretrained(output_dir)
            torch.save(state_dict, Path(output_dir) / "weights.pt")

    model = Model()
    actor = SimpleNamespace(
        model=SimpleNamespace(module=model),
        rank=0,
        stage=3,
        tokenizer=SimpleNamespace(save_pretrained=lambda output: None),
    )
    namespace = {
        "PreTrainedTokenizer": object,
        "os": os,
        "torch": torch,
        "get_olmo3_generation_config": lambda tokenizer: GenerationConfig(),
        "_z3_params_to_fetch": lambda params: [],
        "deepspeed": SimpleNamespace(zero=SimpleNamespace(GatheredParameters=lambda *args, **kwargs: nullcontext())),
        "PeftModel": type("UnusedPeftModel", (), {}),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "historical_save_model", "exec"), namespace)
    namespace["save_model"](actor, str(tmp_path), "olmo", object())
    assert GenerationConfig.from_pretrained(tmp_path).do_sample
    assert torch.equal(torch.load(tmp_path / "weights.pt", weights_only=True)["weight"], model.weight)


def test_optimizer_ledger_path_survives_historical_output_directory_rewriting(tmp_path, monkeypatch):
    source, changes = original_baseline.patch_trainer(Path("/stage/open_instruct/grpo_fast.py").read_text())
    ledger = tmp_path / "optimizer-updates.jsonl"
    monkeypatch.setenv("OI_ORIGINAL_BASELINE_UPDATE_LEDGER", str(ledger))
    block = ast.parse("def record():\n" + changes["update_accounting"]["after"].split("        if (")[0])
    namespace = {
        "os": os,
        "json": json,
        "training_step": 2,
        "args": SimpleNamespace(output_dir=str(tmp_path / "model" / "rewritten-name")),
    }
    exec(compile(block, "update_ledger", "exec"), namespace)
    namespace["record"]()
    assert original_baseline.read_jsonl(ledger) == [{"driver_step": 2}]


@pytest.mark.parametrize("scores", [[1, 1, 0, 0], [1, 0, 0, 0]])
def test_opt_in_comparison_retains_groups_without_changing_their_advantages(scores):
    source = Path("/stage/open_instruct/grpo_fast.py").read_text()
    _, default_changes = original_baseline.patch_trainer(source)
    assert "retain_zero_advantage_groups" not in default_changes
    _, changes = original_baseline.patch_trainer(source, keep_zero_advantage_groups=True)
    scores = np.array(scores)
    grouped = scores.reshape(-1, 2)
    advantages = scores - np.repeat(grouped.mean(axis=-1), 2)
    expanded_mask = np.repeat(grouped.std(axis=-1) != 0, 2)
    namespace = {"np": np, "scores": scores, "expanded_mask": expanded_mask}
    exec(changes["retain_zero_advantage_groups"]["after"].strip(), namespace)
    index = namespace["non_zero_gradient_index"]
    assert index.tolist() == list(range(len(scores)))
    assert np.array_equal(advantages[index], advantages)


def test_comparison_retention_switch_is_explicit_in_submitted_command():
    spec = launch_original_baseline.specification("image", "source", "train", "test", keep_zero_advantage_groups=True)
    assert "--keep-zero-advantage-groups" in spec["tasks"][0]["arguments"][0]
    spec = launch_original_baseline.specification("image", "source", "train", "test")
    assert "--keep-zero-advantage-groups" not in spec["tasks"][0]["arguments"][0]


def test_partial_evaluation_timeout_does_not_remove_results_or_prompts():
    source = Path("/stage/open_instruct/grpo_fast.py").read_text()
    patched, _ = original_baseline.patch_trainer(source)
    tree = ast.parse(patched)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "maybe_evaluate")
    result_queue = queue.Queue()
    result_queue.put(SimpleNamespace(dataset_index=7))
    pending = {7: "first prompt", 8: "second prompt"}

    def consume(*args, **kwargs):
        pytest.fail("An incomplete evaluation round was consumed")

    namespace = {
        "time": time,
        "Empty": queue.Empty,
        "logger": SimpleNamespace(warning=lambda message: None),
        "accumulate_inference_batches": consume,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "historical_eval",
            "exec",
            __future__.annotations.compiler_flag,
        ),
        namespace,
    )
    namespace["maybe_evaluate"](
        SimpleNamespace(num_training_steps=200, local_eval_every=50),
        5,
        result_queue,
        None,
        None,
        0,
        pending,
        None,
        None,
        2,
        None,
    )
    assert result_queue.qsize() == 1
    assert pending == {7: "first prompt", 8: "second prompt"}


def test_checkpoint_export_is_pinned_and_keeps_source_unchanged(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"olmo3"}')
    save_file({"weight": torch.ones(3, dtype=torch.bfloat16)}, model / "model.safetensors")
    root = tmp_path / "checkpoint"
    checkpoint = root / "global_step101"
    checkpoint.mkdir(parents=True)
    torch.save({"training_step": 100}, checkpoint / "mp_rank_00_model_states.pt")
    before = (checkpoint / "mp_rank_00_model_states.pt").read_bytes()
    values = torch.tensor([1.001, 2.123, -3.0])

    def load(path, *, tag):
        assert path == str(root) and tag == "global_step101"
        return {"weight": values}

    monkeypatch.setattr(
        original_baseline,
        "load_zero_converter",
        lambda: SimpleNamespace(get_fp32_state_dict_from_zero_checkpoint=load),
    )
    output = tmp_path / "export"
    original_baseline.export_checkpoint(model, root, "global_step101", output)
    receipt = json.loads((output / "export.json").read_text())
    assert receipt["training_step"] == 100
    assert receipt["dtype"] == "bfloat16"
    index = json.loads((output / "hf/model.safetensors.index.json").read_text())
    with original_baseline.safe_open(output / "hf" / index["weight_map"]["weight"], framework="pt") as f:
        actual = f.get_tensor("weight")
    assert torch.equal(actual, values.bfloat16())
    assert actual.dtype == torch.bfloat16
    assert (checkpoint / "mp_rank_00_model_states.pt").read_bytes() == before


def test_checkpoint_export_launch_is_cpu_only_on_saturn():
    task = launch_original_baseline.specification(
        "image", "source", "export", "test", checkpoint_root="/weka/checkpoint", checkpoint_tag="global_step101"
    )["tasks"][0]
    assert task["resources"]["gpuCount"] == 0
    assert task["constraints"] == {"cluster": ["ai2/saturn"]}
    assert "--checkpoint-tag global_step101" in task["arguments"][0]


def test_historical_zero_converter_imports_on_cpu():
    converter = original_baseline.load_zero_converter()
    assert callable(converter.get_fp32_state_dict_from_zero_checkpoint)
