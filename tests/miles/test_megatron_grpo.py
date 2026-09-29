"""Teacher-free contract, native objective mapping and verifier reward centering."""

import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import torch as safetensors_torch

from open_instruct.miles import (
    launch,
    megatron_grpo_args,
    megatron_grpo_audit,
    megatron_grpo_config,
    megatron_grpo_convert,
    megatron_grpo_hooks,
    specs,
)
from open_instruct.miles.errors import InputError


def document():
    return {
        "schema_version": 1,
        "name": "qwen3-grpo-test",
        "model": {"source": "/original/qwen3"},
        "training": {"algorithm": "grpo"},
        "trainer": {"backend": "megatron", "gpus": 2, "tensor_parallel_size": 2},
        "inference": {"gpus": 2, "rollout_batch_size": 8, "samples_per_prompt": 2},
        "data": {
            "prompt_data": "/data/train.jsonl",
            "eval_prompt_data": ["aime", "/data/eval.jsonl"],
            "reward_config": "/data/verifiers.json",
        },
        "output": {"root": "/run/fresh", "assets": "/assets/fresh"},
        "launch": {"gpus_per_replica": 4},
    }


def test_teacher_free_dispatch_and_roundtrip():
    spec = specs.from_dict(document())
    assert isinstance(spec, megatron_grpo_config.MegatronGRPORunSpec)
    assert specs.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    assert spec.allocation()["roles"] == {"trainer": [0, 1], "student": [2, 3]}
    assert spec.plan()["algorithm"] == "grpo" and not spec.plan()["runtime_validated"]
    task = launch.specification("pinned-image", spec)["tasks"][0]
    assert task["resources"]["gpuCount"] == 4 and not task["context"]["autoResume"]
    assert "teacher" not in spec.to_dict() and "distillation" not in spec.to_dict()


@pytest.mark.parametrize(
    "field,value",
    [
        ("teacher", {"source": "/teacher"}),
        ("distillation", {"kl_coef": 1.0}),
        ("miles", {"opd_kl_coef": 1.0}),
        ("miles", {"use_opd": True}),
        ("miles", {"custom_rm_path": "other.reward"}),
        ("miles", {"use_rollout_logprobs": True}),
        ("miles", {"fully_async": True}),
        ("miles", {"normalize_advantages": True}),
        ("miles", {"use_kl_loss": True}),
        ("miles", {"reset_optimizer_states": True}),
        ("miles", {"skip_actor_forward_only": True}),
    ],
)
def test_reject_opd_and_silent_objective_changes(field, value):
    doc = document()
    doc[field] = value
    with pytest.raises(InputError):
        specs.from_dict(doc)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["data"].pop("reward_config"),
        lambda d: d["inference"].update(samples_per_prompt=1),
        lambda d: d["trainer"].update(gpus=3),
        lambda d: d["launch"].update(gpus_per_replica=8),
        lambda d: d["model"].update(architecture="qwen3.5-2B"),
        lambda d: d["output"].update(assets="/run/fresh/assets"),
    ],
)
def test_reject_invalid_input_and_placement(change):
    doc = document()
    change(doc)
    with pytest.raises(InputError):
        specs.from_dict(doc)


def test_native_mapping_has_real_grpo_no_teacher(tmp_path):
    doc = document()
    doc["output"]["root"] = str(tmp_path / "root")
    spec = specs.from_dict(doc)
    args = megatron_grpo_args.native_arguments(
        spec,
        {
            "model": "/prepared/model",
            "data": {"prompt_data": "/prepared/prompts", "eval_prompt_data": ["aime", "/prepared/eval"]},
        },
        "/converted/native",
        [],
    )

    def value(flag):
        return args[args.index(flag) + 1]

    assert value("--advantage-estimator") == "grpo" and value("--opd-kl-coef") == "0.0"
    assert value("--custom-rm-path") == "open_instruct.miles.megatron_grpo_hooks.reward"
    assert value("--global-batch-size") == "16" and value("--eps-clip-high") == "0.28"
    assert value("--adam-beta2") == "0.95" and value("--adam-eps") == "1e-08"
    assert "--use-opd" not in args and "--rm-url" not in args and "--opd-type" not in args
    assert "--use-rollout-logprobs" not in args and "--disable-grpo-std-normalization" in args
    assert "--use-kl-loss" not in args and "--normalize-advantages" not in args


def test_native_hook_uses_same_registered_verifier(monkeypatch):
    seen = []

    async def score(sample, path, args):
        seen.append((sample, path, args))
        return 1.0

    monkeypatch.setattr(megatron_grpo_hooks.rewards, "score", score)
    monkeypatch.setenv("OI_GRPO_REWARD_CONFIG", "/fixed/verifiers.json")
    args, sample = SimpleNamespace(), SimpleNamespace()
    assert asyncio.run(megatron_grpo_hooks.reward(args, sample)) == 1.0
    assert seen == [(sample, "/fixed/verifiers.json", args)]


def test_center_rewards_before_native_grpo_and_preserve_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("OI_GRPO_OUTPUT", str(tmp_path))
    samples = [
        SimpleNamespace(
            index=i,
            reward=value,
            tokens=[1, 2, 3 + i],
            response_length=1,
            response="answer",
            metadata={"verifiers": []},
            group_index=i // 2,
        )
        for i, value in enumerate([0.0, 1.0, 0.0, 0.0])
    ]
    args = SimpleNamespace(n_samples_per_prompt=2, grpo_std_normalization=False)
    raw, centered = megatron_grpo_hooks.post_process(args, samples)
    assert raw == [0.0, 1.0, 0.0, 0.0] and centered == [-0.5, 0.5, 0.0, 0.0]
    records = [json.loads(line) for line in (tmp_path / "verifier-rewards.jsonl").read_text().splitlines()]
    assert [row["reward"] for row in records] == raw
    args.grpo_std_normalization = True
    assert megatron_grpo_hooks.post_process(args, samples)[1] == pytest.approx([-0.70710578, 0.70710578, 0, 0])
    samples[1].tokens = [9, 2, 4]
    with pytest.raises(ValueError, match="different tokenized prompts"):
        megatron_grpo_hooks.post_process(args, samples)


@pytest.mark.parametrize("wrong_advantage", [False, True])
def test_audit_requires_centered_reward_signal_in_trainer_dumps(tmp_path, monkeypatch, wrong_advantage):
    doc = document()
    doc["output"] = {"root": str(tmp_path / "run"), "assets": str(tmp_path / "assets")}
    spec = specs.from_dict(doc)
    root = Path(spec.output["root"])
    debug = root / "debug/train_data"
    debug.mkdir(parents=True)
    records = []
    for step in range(2):
        batch = []
        for index in range(16):
            reward = float(index == 1)
            batch.append(
                {
                    "sample_index": 16 * step + index,
                    "tokens": [1, 2, 3],
                    "response_length": 1,
                    "response": "answer",
                    "reward": reward,
                    "metadata": {"query": "question", "expected_reward": reward},
                }
            )
        records += batch
        indices = [row["sample_index"] for row in batch]
        advantages = [torch.tensor([0.5 if index == 1 else -0.5 if index == 0 else 0.0]) for index in range(16)]
        if wrong_advantage:
            advantages[0] = torch.tensor([0.0])
        torch.save({"rollout_data": {"sample_indices": indices, "advantages": advantages}}, debug / f"{step}_0.pt")
    (root / "verifier-rewards.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    (root / "training.log").write_text(
        "step 0: {'train/step': 0, 'train/grad_norm': 1.0}\nstep 1: {'train/step': 1, 'train/grad_norm': 1.0}\n"
    )
    base, export = tmp_path / "base", root / "hf-1"
    base.mkdir()
    export.mkdir()
    safetensors_torch.save_file({"model.norm.weight": torch.tensor([1.0])}, base / "model.safetensors")
    safetensors_torch.save_file({"model.norm.weight": torch.tensor([2.0])}, export / "model.safetensors")
    (export / ".complete").touch()

    async def score(sample, registry):
        return sample.metadata["expected_reward"]

    monkeypatch.setattr(megatron_grpo_audit.rewards, "score", score)
    prepared = {"model": str(base), "data": {"reward_config": "/fixed/registry"}}
    if wrong_advantage:
        with pytest.raises(AssertionError):
            megatron_grpo_audit.audit(spec, prepared)
    else:
        result = megatron_grpo_audit.audit(spec, prepared)
        assert result["passed"] and result["teacher"] is None
        assert [row["mixed_groups"] for row in result["rollouts"]] == [1, 1]
        assert result["export"]["changed_tensors"] == 1


def test_conversion_tp_only_mesh_matches_trainer():
    command = megatron_grpo_args.conversion_command(
        "/usr/bin/python", "/native/miles", ["--num-layers", "28"], "/original/qwen3", "/converted/tp2", 2
    )
    assert "--nproc-per-node=2" in command
    assert command[command.index("--tensor-model-parallel-size") + 1] == "2"
    assert command[command.index("--pipeline-model-parallel-size") + 1] == "1"
    assert command[command.index("--hf-checkpoint") + 1] == "/original/qwen3"
    assert command[command.index("--save") + 1] == "/converted/tp2"


CONVERTER_ARGS = """
def get_args():
    args = parse_args(None)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        args.pipeline_model_parallel_size = world_size
        args.decoder_last_pipeline_num_layers = args.num_layers // world_size
    validate_args(args)
    return args
"""


def exercise_converter(source, tp, patched):
    tree = megatron_grpo_convert.converter_tree(source, "native-converter") if patched else ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args")
    namespace = {
        "parse_args": lambda _: SimpleNamespace(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            num_layers=28,
            decoder_last_pipeline_num_layers=None,
        ),
        "set_default_megatron_args": lambda args: args,
        "add_convertion_args": None,
        "os": SimpleNamespace(environ={"WORLD_SIZE": "2"}),
    }

    def validate(args):
        assert 2 % (args.tensor_model_parallel_size * args.pipeline_model_parallel_size) == 0

    namespace["validate_args"] = validate
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native-converter", "exec"), namespace)
    return namespace["get_args"]()


def test_converter_override_reproduced_and_guarded():
    with pytest.raises(AssertionError):
        exercise_converter(CONVERTER_ARGS, tp=2, patched=False)
    args = exercise_converter(CONVERTER_ARGS, tp=2, patched=True)
    assert args.pipeline_model_parallel_size == 1 and args.decoder_last_pipeline_num_layers is None
    args = exercise_converter(CONVERTER_ARGS, tp=1, patched=True)
    assert args.pipeline_model_parallel_size == 2 and args.decoder_last_pipeline_num_layers == 14
    with pytest.raises(ValueError, match="guard changed"):
        megatron_grpo_convert.converter_tree(CONVERTER_ARGS.replace("world_size > 1", "world_size > 2"), "changed")


def test_pinned_native_converter_tp_only():
    native = importlib.util.find_spec("miles")
    if native is None or native.origin is None:
        pytest.skip("Pinned native converter is supplied by the GPU runtime image")
    path = Path(native.origin).resolve().parents[1] / "tools/convert_hf_to_torch_dist.py"
    source = path.read_text()
    with pytest.raises(AssertionError):
        exercise_converter(source, tp=2, patched=False)
    args = exercise_converter(source, tp=2, patched=True)
    assert args.pipeline_model_parallel_size == 1 and args.decoder_last_pipeline_num_layers is None
