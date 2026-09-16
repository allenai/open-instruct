"""Frozen-input and placement contracts for the historical framework control.

Run in its original image, with /stage before this checkout on PYTHONPATH.
"""

import __future__

import ast
import importlib
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


def test_real_zero3_export_reconstructs_partitioned_weights(tmp_path, monkeypatch):
    original_baseline.load_zero_converter()
    config = importlib.import_module("deepspeed.runtime.zero.config")
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"olmo3"}')
    expected = torch.tensor([1.003, -2.333, 3.111])
    save_file({"weight": torch.zeros(3, dtype=torch.bfloat16)}, model / "model.safetensors")
    root = tmp_path / "checkpoints"
    tag = root / "global_step101"
    tag.mkdir(parents=True)
    for rank, values in enumerate((expected[:2], torch.tensor([expected[2], 0.0]))):
        torch.save(
            {
                "optimizer_state_dict": {
                    "zero_stage": config.ZeroStageEnum.weights,
                    "partition_count": 2,
                    "fp32_flat_groups": [values],
                    "optimizer_state_dict": {},
                }
            },
            tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        )
        torch.save(
            {
                "training_step": 100,
                "buffer_names": [],
                "module": {},
                "param_shapes": [{"weight": torch.Size([3])}],
                "shared_params": {},
                "ds_version": "test",
            },
            tag / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt",
        )
    original_baseline.export_checkpoint(model, root, tag.name, tmp_path / "export")
    with original_baseline.safe_open(tmp_path / "export/hf/model-00000.safetensors", framework="pt") as f:
        assert torch.equal(f.get_tensor("weight"), expected.bfloat16())


def test_evaluation_launch_is_separate_from_training():
    task = launch_original_baseline.specification(
        "image", "source", "evaluate", "test", evaluation_model="/weka/export/hf"
    )["tasks"][0]
    assert task["resources"]["gpuCount"] == 1
    assert task["constraints"] == {"cluster": ["ai2/jupiter"]}
    assert "evaluate --model /weka/export/hf" in task["arguments"][0]
    assert not any(e["name"] == "WANDB_API_KEY" for e in task["envVars"])


def test_eval_rejects_modified_frozen_source_before_starting_engine(tmp_path):
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    source.mkdir()
    prepared.mkdir()
    (source / "manifest.json").write_text("{}")
    (prepared / "preparation.json").write_text(json.dumps({"source_manifest_sha256": "bad"}))
    with pytest.raises(ValueError, match="source differs"):
        original_baseline.evaluate_checkpoint(tmp_path / "model", source, prepared, tmp_path / "eval")
    assert not (tmp_path / "eval").exists()


def test_independent_eval_keeps_frozen_tokens_and_all_outputs(tmp_path, monkeypatch):
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    source.mkdir()
    prepared.mkdir()
    rows = [row() for _ in range(512)]
    for index, sample in enumerate(rows):
        sample["metadata"]["prepared_sample_id"] = f"gsm8k:train:{index}"
    raw = b"".join(original_baseline.encoded(r) for r in rows)
    (source / "eval.jsonl").write_bytes(raw)
    manifest = original_baseline.encoded({"outputs": {"eval.jsonl": original_baseline.sha(raw)}})
    (source / "manifest.json").write_bytes(manifest)
    (prepared / "preparation.json").write_text(json.dumps({"source_manifest_sha256": original_baseline.sha(manifest)}))
    monkeypatch.setattr(
        original_baseline.dataset_transformation,
        "TokenizerConfig",
        lambda **kw: SimpleNamespace(tokenizer=Tokenizer()),
    )
    monkeypatch.setattr(original_baseline, "legacy_model", lambda *a: {})
    seen = []

    class Engine:
        def __init__(self, **options):
            assert options["max_model_len"] == 34816

        def generate(self, prompts, sampling, **options):
            assert sampling.temperature == 0.0 and sampling.max_tokens == 32768
            seen.extend(prompts)
            return [
                SimpleNamespace(
                    prompt_token_ids=p["prompt_token_ids"],
                    outputs=[SimpleNamespace(text="-12", token_ids=[5, 6], finish_reason="stop")],
                )
                for p in prompts
            ]

    modules = {
        "vllm": SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace),
        "open_instruct.ground_truth_utils": SimpleNamespace(
            GSM8KVerifier=lambda: lambda **kw: SimpleNamespace(score=1.0)
        ),
    }
    monkeypatch.setattr(original_baseline.importlib, "import_module", modules.__getitem__)
    output = tmp_path / "eval"
    original_baseline.evaluate_checkpoint(tmp_path / "model", source, prepared, output)
    assert len(seen) == 512 and all(p["prompt_token_ids"] == [10, 11] for p in seen)
    saved = original_baseline.read_jsonl(output / "generations.jsonl")
    assert [r["id"] for r in saved] == [r["metadata"]["prepared_sample_id"] for r in rows]
    summary = json.loads((output / "evaluation.json").read_text())
    assert summary["correct"] == 512 and summary["capped"] == 0


def resume_fixture(tmp_path):
    record = {
        "command": ["python", "train.py", "--learning_rate", "1e-6"],
        "original_source_sha256": "source",
        "model_alias": {"source": "model"},
    }
    (tmp_path / "invocation.json").write_text(json.dumps(record))
    checkpoint = tmp_path / "checkpoints/global_step126"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "latest").write_text(checkpoint.name)
    for rank in range(4):
        torch.save(
            {"training_step": 125, "rng_states": {}}, checkpoint / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        )
        (checkpoint / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(b"test")
    (tmp_path / "optimizer-updates.jsonl").write_bytes(
        b"".join(original_baseline.encoded({"driver_step": n}) for n in range(1, 146))
    )
    return record, checkpoint


def test_resume_archives_unsaved_updates_and_preserves_checkpoint(tmp_path):
    record, checkpoint = resume_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    result = original_baseline.prepare_resume(tmp_path, record)
    assert result["completed_steps"] == 125 and result["discarded_unsaved_updates"] == 20
    assert len(original_baseline.read_jsonl(tmp_path / "optimizer-updates.jsonl")) == 125
    archive = next(tmp_path.glob("optimizer-updates-before-resume-*.jsonl"))
    assert len(original_baseline.read_jsonl(archive)) == 145
    assert {p.name: p.read_bytes() for p in checkpoint.iterdir()} == before


@pytest.mark.parametrize("failure", ["recipe", "rank", "shard", "newer", "complete"])
def test_resume_fails_before_changing_ledger_on_invalid_state(tmp_path, failure):
    record, checkpoint = resume_fixture(tmp_path)
    if failure == "recipe":
        record["command"][-1] = "1e-5"
    elif failure == "rank":
        torch.save({"training_step": 124, "rng_states": {}}, checkpoint / "zero_pp_rank_3_mp_rank_00_model_states.pt")
    elif failure == "shard":
        (checkpoint / "bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt").unlink()
    elif failure == "newer":
        (checkpoint.parent / "global_step151").mkdir()
    else:
        (tmp_path / "completion.json").write_text("{}")
    ledger = (tmp_path / "optimizer-updates.jsonl").read_bytes()
    with pytest.raises(ValueError):
        original_baseline.prepare_resume(tmp_path, record)
    assert (tmp_path / "optimizer-updates.jsonl").read_bytes() == ledger


def test_explicit_resume_launch_reuses_run_and_enables_preemption_recovery():
    task = launch_original_baseline.specification(
        "image", "source", "resume", "existing-run", keep_zero_advantage_groups=True
    )["tasks"][0]
    assert task["context"]["autoResume"] is True
    assert task["context"]["minRuntime"] == "4h"
    assert task["resources"]["gpuCount"] == 8
    assert "original_baseline.py resume" in task["arguments"][0]
    assert "--keep-zero-advantage-groups" in task["arguments"][0]
    assert {e["name"]: e.get("value") for e in task["envVars"]}["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


# --- Four-domain basket profile -------------------------------------------------


def basket_row(name, target):
    sample = row()
    sample["metadata"]["prepared_sample_id"] = f"basket:{name}"
    sample["metadata"]["verifiers"] = [{"name": name, "target": target, "weight": 1.0}]
    return sample


@pytest.mark.parametrize(
    "name, target, expected",
    [
        ("math", "\\frac{1}{2}", "\\frac{1}{2}"),
        ("ifeval", {"func_name": "validate_lowercase", "N": None}, '{"N": null, "func_name": "validate_lowercase"}'),
        ("code", ["assert f(1) == 2", "assert f(2) == 3"], '["assert f(1) == 2", "assert f(2) == 3"]'),
        ("code_stdio", '[{"input": "1", "output": "2"}]', '[{"input": "1", "output": "2"}]'),
        ("general-quality", "reference answer", "reference answer"),
        ("general-quality_ref", "reference answer", "reference answer"),
    ],
)
def test_basket_rows_keep_the_original_verifier_name_and_serialize_labels(name, target, expected):
    converted = original_baseline.convert_row(basket_row(name, target), Tokenizer(), profile="basket")
    assert converted["dataset"] == name
    assert converted["ground_truth"] == expected
    assert isinstance(converted["ground_truth"], str)


@pytest.mark.parametrize("bad", [basket_row("gsm8k", "1"), basket_row("math", {"unexpected": 1})])
def test_basket_rows_reject_foreign_verifiers_and_unknown_label_types(bad):
    with pytest.raises(ValueError, match="basket rows require|Unsupported frozen target"):
        original_baseline.convert_row(bad, Tokenizer(), profile="basket")
    with pytest.raises(ValueError, match="unit-weight GSM8K"):
        original_baseline.convert_row(basket_row("math", "1"), Tokenizer())


def test_basket_frozen_splits_merge_the_four_held_out_domains(tmp_path):
    def write(name, count, label):
        (tmp_path / name).write_text("".join(json.dumps({"input": f"{label} {i}"}) + "\n" for i in range(count)))

    write("train.jsonl", 5, "train")
    for domain in ("math", "ifeval", "code", "general"):
        write(f"{domain}.jsonl", 128, domain)
    splits = original_baseline.frozen_splits(tmp_path, "basket")
    assert len(splits["train"]) == 5 and len(splits["eval"]) == 512
    assert [r["input"] for r in splits["eval"][:2]] == ["math 0", "math 1"]
    write("code.jsonl", 100, "code")
    with pytest.raises(ValueError, match="128 frozen rows"):
        original_baseline.frozen_splits(tmp_path, "basket")


def test_basket_training_options_follow_the_released_recipe_and_gsm8k_is_unchanged(tmp_path):
    basket = original_baseline.training_options(
        tmp_path / "prepared", tmp_path / "run", profile="basket", steps=100, smoke=False
    )
    assert (basket["num_unique_prompts_rollout"], basket["num_samples_per_prompt_rollout"]) == (64, 4)
    assert basket["total_episodes"] == 100 * 64 * 4
    assert (basket["num_learners_per_node"], basket["vllm_num_engines"]) == (4, 2)
    assert basket["llm_judge_model"] == "hosted_vllm/Qwen/Qwen3-32B"
    assert basket["llm_judge_max_context_length"] == 131072 and basket["llm_judge_max_tokens"] == 2048
    assert basket["code_api_url"] == original_baseline.CODE_API_URL
    assert basket["code_pass_rate_reward_threshold"] == 0.99
    assert basket["response_length"] == 32768 and basket["learning_rate"] == 1e-6 and basket["beta"] == 0.0
    assert basket["checkpoint_state_freq"] == 25 and basket["local_eval_every"] == 50
    gsm8k = original_baseline.training_options(
        tmp_path / "prepared", tmp_path / "run", profile="gsm8k", steps=200, smoke=False
    )
    assert (gsm8k["num_unique_prompts_rollout"], gsm8k["vllm_num_engines"]) == (16, 4)
    assert "llm_judge_model" not in gsm8k and "code_api_url" not in gsm8k
    smoke = original_baseline.training_options(
        tmp_path / "prepared", tmp_path / "run", profile="gsm8k", steps=3, smoke=True
    )
    assert smoke["num_unique_prompts_rollout"] == 128
    trainer_gpus, judge_gpus = original_baseline.TRAINER_GPUS.split(","), original_baseline.JUDGE_GPUS.split(",")
    assert set(trainer_gpus).isdisjoint(judge_gpus) and len(trainer_gpus) + len(judge_gpus) == 8
    assert len(trainer_gpus) == basket["num_learners_per_node"] + basket["vllm_num_engines"]
    assert len(judge_gpus) == original_baseline.JUDGE_TENSOR_PARALLEL


def test_judge_service_uses_the_prepared_miles_snapshot_and_template(tmp_path):
    template = tmp_path / "judge.jinja"
    template.write_text("{{ messages }}<|im_start|>assistant\n<think>\n\n</think>\n\n")
    (tmp_path / "snapshot").mkdir()
    (tmp_path / "snapshot" / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "max_position_embeddings": 40960})
    )
    prepared = {
        "verdict": "passed",
        "model": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "snapshot": str(tmp_path / "snapshot"),
        "template": str(template),
        "template_sha256": original_baseline.sha(template.read_bytes()),
        "rendered_canary": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    }
    (tmp_path / "prepared.json").write_text(json.dumps(prepared))
    service = original_baseline.judge_service(tmp_path)
    command = service["command"]
    assert command[command.index("--model") + 1] == str(tmp_path / "snapshot")
    assert command[command.index("--chat-template") + 1] == str(template)
    assert command[command.index("--served-model-name") + 1] == "Qwen/Qwen3-32B"
    assert command[command.index("--tensor-parallel-size") + 1] == "2" and "--enforce-eager" in command
    assert command[command.index("--max-model-len") + 1] == "131072"
    overrides = json.loads(command[command.index("--hf-overrides") + 1])
    assert overrides["rope_scaling"] == {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    assert overrides["max_position_embeddings"] == 131072
    assert service["api_base"] == "http://127.0.0.1:8001/v1"
    assert "command" in service and service["template_sha256"] == prepared["template_sha256"]
    template.write_text("changed")
    with pytest.raises(ValueError, match="template hash"):
        original_baseline.judge_service(tmp_path)
    template.write_text("{{ messages }}<|im_start|>assistant\n<think>\n\n</think>\n\n")
    prepared["rendered_canary"] = "<|im_start|>assistant\n"
    (tmp_path / "prepared.json").write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="thinking block"):
        original_baseline.judge_service(tmp_path)
    prepared["rendered_canary"] = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    (tmp_path / "prepared.json").write_text(json.dumps(prepared))
    (tmp_path / "snapshot" / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "rope_scaling": {"factor": 4.0}})
    )
    with pytest.raises(ValueError, match="unscaled Qwen3"):
        original_baseline.judge_service(tmp_path)


def test_basket_launch_passes_profile_steps_and_judge_only_to_gpu_stages():
    train = launch_original_baseline.specification("image", "source", "train", "test", profile="basket", steps=100)
    command = train["tasks"][0]["arguments"][0]
    assert "--profile basket" in command and "--steps 100" in command and "--judge-prepared" in command
    assert launch_original_baseline.PROFILES["basket"]["source"].endswith("g16-20260915/prepared/data")
    assert train["tasks"][0]["resources"]["gpuCount"] == 8
    assert train["tasks"][0]["constraints"] == {"cluster": ["ai2/jupiter"]}
    assert {e["name"]: e.get("value") for e in train["tasks"][0]["envVars"]}[
        "WANDB_RUN_GROUP"
    ] == "dolci-basket-32k-zero-20260914"
    smoke = launch_original_baseline.specification("image", "source", "smoke", "test", profile="basket")
    assert smoke["tasks"][0]["context"]["minRuntime"] == "2h"
    assert (
        launch_original_baseline.specification("image", "source", "smoke", "test")["tasks"][0]["context"]["minRuntime"]
        == "30m"
    )
    prepare = launch_original_baseline.specification("image", "source", "prepare", "test", profile="basket")
    assert "--judge-prepared" not in prepare["tasks"][0]["arguments"][0]
    assert "--profile basket" in prepare["tasks"][0]["arguments"][0]
    gsm8k = launch_original_baseline.specification("image", "source", "train", "test")
    assert (
        "--profile gsm8k" in gsm8k["tasks"][0]["arguments"][0]
        and "--judge-prepared" not in gsm8k["tasks"][0]["arguments"][0]
    )
