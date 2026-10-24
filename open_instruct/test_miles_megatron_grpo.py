"""Megatron verifier GRPO: teacher-free contract, native objective mapping, verifier reward
centering, mechanics audit, initial-checkpoint reuse and bounded conversion."""

import ast
import asyncio
import contextlib
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import torch as safetensors_torch

from open_instruct.miles.configuration import run_spec, specs
from open_instruct.miles.distillation import opd_config
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import launch
from open_instruct.miles.megatron_grpo import (
    checkpoint_reuse,
    conversion,
    grpo_args,
    grpo_audit,
    grpo_config,
    grpo_hooks,
)

PREPARED = {
    "model": "/prepared/model",
    "data": {"prompt_data": "/prepared/prompts", "eval_prompt_data": ["aime", "/prepared/eval"]},
}


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


def qwen35_async_dppo():
    """The Open Instruct teacher recipe's objective: DPPO-TV 0.1 on rollout log-probs, token mean, async."""
    doc = document()
    doc["model"]["architecture"] = "qwen3.5-2B"
    doc["training"]["loss_aggregation"] = "token"
    doc["objective"] = {"policy_loss": "dppo", "use_rollout_logprobs": True}
    doc["miles"] = {"fully_async": True, "max_weight_staleness": 5}
    return doc


def flag_value(args, flag):
    return args[args.index(flag) + 1]


def test_teacher_free_dispatch_and_roundtrip():
    spec = specs.from_dict(document())
    assert isinstance(spec, grpo_config.MegatronGRPORunSpec)
    assert isinstance(spec, specs.MEGATRON_SPECS)
    assert specs.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    assert spec.allocation()["roles"] == {"trainer": [0, 1], "student": [2, 3]}
    assert spec.plan()["algorithm"] == "grpo" and not spec.plan()["runtime_validated"]
    beaker = launch.specification("pinned-image", spec)
    task = beaker["tasks"][0]
    assert beaker["description"] == "Miles Megatron GRPO: qwen3-grpo-test"
    assert task["resources"]["gpuCount"] == 4 and not task["context"]["autoResume"]
    assert "teacher" not in spec.to_dict() and "distillation" not in spec.to_dict()
    assert spec.to_dict()["reward"] == {"zero_truncated": False, "final_answer_only": False}


@pytest.mark.parametrize(
    "algorithm,backend,kind",
    [
        ("grpo", "megatron", "megatron-grpo"),
        ("grpo", None, "core"),
        ("grpo", "olmo_core", "core"),
        ("opd", None, "opd"),
        ("opd", "megatron", "opd"),
        (None, None, "core"),
    ],
)
def test_dispatch_selects_megatron_grpo_only_for_grpo_on_megatron(monkeypatch, algorithm, backend, kind):
    for cls, name in (
        (grpo_config.MegatronGRPORunSpec, "megatron-grpo"),
        (opd_config.OPDRunSpec, "opd"),
        (run_spec.RunSpec, "core"),
    ):
        monkeypatch.setattr(cls, "from_dict", classmethod(lambda cls, document, config_path, name=name: name))
    doc = {"training": {} if algorithm is None else {"algorithm": algorithm}}
    if backend is not None:
        doc["trainer"] = {"backend": backend}
    assert specs.from_dict(doc) == kind


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
        ("reward", {"zero_truncated": "yes"}),
        ("reward", {"partial_credit": True}),
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
        lambda d: d["model"].update(architecture="qwen3.5-4B"),
        lambda d: d["model"].update(source="Qwen/Qwen3-1.7B"),
        lambda d: d["output"].update(assets="/run/fresh/assets"),
        lambda d: d["training"].update(keep_checkpoints=2),
        lambda d: d["inference"].update(max_prompt_length=4000),
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
    args = grpo_args.native_arguments(spec, PREPARED, "/converted/native", [])
    assert flag_value(args, "--advantage-estimator") == "grpo" and flag_value(args, "--opd-kl-coef") == "0.0"
    assert flag_value(args, "--custom-rm-path") == "open_instruct.miles.megatron_grpo.grpo_hooks.reward"
    assert (
        flag_value(args, "--custom-reward-post-process-path")
        == "open_instruct.miles.megatron_grpo.grpo_hooks.post_process"
    )
    assert flag_value(args, "--global-batch-size") == "16" and flag_value(args, "--eps-clip-high") == "0.28"
    assert flag_value(args, "--adam-beta2") == "0.95" and flag_value(args, "--adam-eps") == "1e-08"
    assert flag_value(args, "--rollout-max-prompt-len") == "2048"
    assert "--use-opd" not in args and "--rm-url" not in args and "--opd-type" not in args
    assert "--use-rollout-logprobs" not in args and "--disable-grpo-std-normalization" in args
    assert "--use-kl-loss" not in args and "--normalize-advantages" not in args
    assert flag_value(args, "--loss-type") == "policy_loss" and "--custom-loss-function-path" not in args
    assert "--data-source-path" not in args and spec.dppo_settings() is None
    assert "--eval-function-path" not in args and "--use-checkpoint-opt-param-scheduler" not in args
    assert "assert (a.fully_async, a.use_rollout_logprobs, a.loss_type) == (False, False, 'policy_loss')" in (
        grpo_args.preflight_code(spec)
    )


def test_resume_restores_checkpoint_without_restepping_the_scheduler(tmp_path):
    doc = document()
    doc["output"]["root"] = str(tmp_path / "root")
    doc["training"]["resume"] = True
    spec = specs.from_dict(doc)
    assert "--load" not in grpo_args.native_arguments(spec, PREPARED, "/converted/native", [])
    marker = tmp_path / "root/checkpoints/latest_checkpointed_iteration.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("3")
    args = grpo_args.native_arguments(spec, PREPARED, "/converted/native", [])
    assert flag_value(args, "--load") == str(tmp_path / "root/checkpoints")
    assert "--use-checkpoint-opt-param-scheduler" in args


def test_serving_passthrough_replaces_the_wrapper_default(tmp_path):
    doc = document()
    doc["output"]["root"] = str(tmp_path / "root")
    doc["miles"] = {"sglang_disable_cuda_graph": False, "sglang_router_request_timeout_secs": 600}
    args = grpo_args.native_arguments(specs.from_dict(doc), PREPARED, "/converted/native", [])
    assert "--sglang-disable-cuda-graph" not in args
    assert flag_value(args, "--sglang-router-request-timeout-secs") == "600"


def test_qwen35_async_dppo_maps_to_custom_loss_rollout_logprobs_and_async_ledger(tmp_path):
    doc = qwen35_async_dppo()
    doc["output"]["root"] = str(tmp_path / "root")
    spec = specs.from_dict(doc)
    assert specs.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    args = grpo_args.native_arguments(spec, PREPARED, "/converted/native", [])
    assert flag_value(args, "--loss-type") == "custom_loss"
    assert flag_value(args, "--custom-loss-function-path") == "open_instruct.miles.megatron_grpo.dppo_loss.policy_loss"
    assert flag_value(args, "--data-source-path") == opd_config.ASYNC_SOURCE
    assert flag_value(args, "--custom-async-data-buffer-path") == opd_config.ASYNC_BUFFER
    assert flag_value(args, "--max-weight-staleness") == "5" and "--fully-async" in args
    assert "--use-rollout-logprobs" in args and "--calculate-per-token-loss" in args
    assert "--disable-grpo-std-normalization" in args and "--use-tis" not in args
    assert spec.dppo_settings().environment() == {"OI_DPPO_DIVERGENCE_TYPE": "tv", "OI_DPPO_THRESHOLD": "0.1"}
    assert "== (True, True, 'custom_loss')" in grpo_args.preflight_code(spec)
    assert "span policy versions" in " ".join(spec.plan()["warnings"])


def test_keep_inflight_switch_is_rejected_with_the_native_behavior():
    for spelling in ("keep_inflight_requests_on_weight_update", "keep-inflight-requests-on-weight-update"):
        doc = qwen35_async_dppo()
        doc["miles"][spelling] = True
        with pytest.raises(InputError, match="never aborts in-flight requests"):
            specs.from_dict(doc)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda d: d["objective"].update(use_rollout_logprobs=False), "use_rollout_logprobs"),
        (lambda d: d["objective"].update(policy_loss="grpo"), "policy_loss"),
        (lambda d: d["objective"].update(dppo_threshold=0.0), "dppo_threshold"),
        (lambda d: d["objective"].update(dppo_divergence_type="js"), "divergence type"),
        (lambda d: d["inference"].update(top_p=0.9), "top_p"),
        (lambda d: d["miles"].pop("max_weight_staleness"), "max_weight_staleness"),
        (lambda d: d["miles"].update(max_weight_staleness=0), "max_weight_staleness"),
        (lambda d: d["miles"].update(fully_async=False), "only with miles.fully_async"),
        (lambda d: d["miles"].update(pause_generation_mode="abort"), "abort"),
        (lambda d: d["miles"].update(data_source_path="other.Source"), "is set by"),
        (lambda d: d["miles"].update(use_tis=True), "is set by"),
        (lambda d: d["miles"].update(loss_type="policy_loss"), "objective.policy_loss"),
        (lambda d: d["miles"].update(update_weights_interval=2), "one publication per optimizer update"),
        (lambda d: d["miles"].update(colocate=True), "colocate"),
        (lambda d: d["training"].update(optimizer_steps_per_rollout=2), "one optimizer step"),
    ],
)
def test_reject_unqualified_async_or_dppo_settings(change, message):
    doc = qwen35_async_dppo()
    change(doc)
    with pytest.raises(InputError, match=message):
        specs.from_dict(doc)


def test_async_requires_rollout_logprobs_even_with_ppo():
    doc = qwen35_async_dppo()
    doc["objective"] = {"use_rollout_logprobs": False}
    with pytest.raises(InputError, match="Async Megatron GRPO requires objective.use_rollout_logprobs"):
        specs.from_dict(doc)
    doc["objective"]["use_rollout_logprobs"] = True
    assert specs.from_dict(doc).dppo_settings() is None


@pytest.mark.parametrize("gates", [(False, False), (True, True)])
def test_reward_hook_uses_registered_verifier_with_run_gates(monkeypatch, gates):
    seen = []

    async def registered_reward(args, samples, **kwargs):
        seen.append((args, samples))
        return [1.0] * len(samples) if isinstance(samples, list) else 1.0

    monkeypatch.setattr(grpo_hooks.rewards, "registered_reward", registered_reward)
    environment = grpo_hooks.reward_environment(
        "/fixed/verifiers.json", {"zero_truncated": gates[0], "final_answer_only": gates[1]}
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    args, sample = SimpleNamespace(n_samples_per_prompt=2), SimpleNamespace()
    assert asyncio.run(grpo_hooks.reward(args, sample)) == 1.0
    assert asyncio.run(grpo_hooks.reward(args, [sample, sample])) == [1.0, 1.0]
    scoring, scored = seen[0]
    assert scored is sample and scoring is not args and not hasattr(args, "olmo_core")
    assert scoring.n_samples_per_prompt == 2
    assert scoring.olmo_core.reward_config == "/fixed/verifiers.json"
    assert (scoring.olmo_core.reward_zero_truncated, scoring.olmo_core.reward_final_answer_only) == gates
    assert seen[1][1] == [sample, sample]


def test_reward_hook_rejects_nonfinite_scores(monkeypatch):
    async def registered_reward(args, samples, **kwargs):
        return float("nan")

    monkeypatch.setattr(grpo_hooks.rewards, "registered_reward", registered_reward)
    monkeypatch.setenv(grpo_hooks.REWARD_CONFIG_ENV, "/fixed/verifiers.json")
    with pytest.raises(ValueError, match="Nonfinite"):
        asyncio.run(grpo_hooks.reward(SimpleNamespace(), SimpleNamespace()))


def test_registered_reward_scores_the_verifier_registry(tmp_path, monkeypatch):
    """The hook reaches the real registry adapter (no monkeypatch on the scoring path)."""
    registry = tmp_path / "verifiers.json"
    registry.write_text(json.dumps({"gsm8k": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"}}))
    environment = grpo_hooks.reward_environment(registry, {"zero_truncated": True, "final_answer_only": False})
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    def sample(status):
        return SimpleNamespace(
            prompt="What is 6 * 7?",
            response="So the answer is 42.",
            tokens=[1, 2],
            response_length=1,
            status=status,
            metadata={"verifiers": [{"name": "gsm8k", "target": "42"}]},
        )

    scored, truncated = sample("completed"), sample("truncated")
    assert asyncio.run(grpo_hooks.reward(SimpleNamespace(), [scored, truncated])) == [1.0, 0.0]
    assert "reward_gate" not in scored.metadata and truncated.metadata["reward_gate"] == "truncated"


def test_center_rewards_before_native_grpo_and_preserve_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv(grpo_hooks.OUTPUT_ENV, str(tmp_path))
    samples = [
        SimpleNamespace(
            index=i,
            reward=value,
            tokens=[1, 2, 3 + i],
            response_length=1,
            response="answer",
            status=SimpleNamespace(value="completed"),
            metadata={"verifiers": []},
            group_index=i // 2,
        )
        for i, value in enumerate([0.0, 1.0, 0.0, 0.0])
    ]
    args = SimpleNamespace(n_samples_per_prompt=2, grpo_std_normalization=False)
    raw, centered = grpo_hooks.post_process(args, samples)
    assert raw == [0.0, 1.0, 0.0, 0.0] and centered == [-0.5, 0.5, 0.0, 0.0]
    records = [json.loads(line) for line in (tmp_path / grpo_hooks.EVIDENCE).read_text().splitlines()]
    assert [row["reward"] for row in records] == raw
    assert [row["group_index"] for row in records] == [0, 0, 1, 1]
    assert {row["status"] for row in records} == {"completed"}
    args.grpo_std_normalization = True
    assert grpo_hooks.post_process(args, samples)[1] == pytest.approx([-0.70710578, 0.70710578, 0, 0])
    samples[1].tokens = [9, 2, 4]
    with pytest.raises(ValueError, match="different tokenized prompts"):
        grpo_hooks.post_process(args, samples)
    samples[1].tokens = [1, 2, 4]
    with pytest.raises(ValueError, match="Incomplete"):
        grpo_hooks.post_process(args, samples[:3])
    samples[1].teacher_log_probs = [0.0]
    with pytest.raises(ValueError, match="Teacher"):
        grpo_hooks.post_process(args, samples)


@pytest.mark.parametrize("case", ["mixed", "wrong_advantage", "zero_signal"])
def test_audit_checks_centered_rewards_in_trainer_dumps(tmp_path, monkeypatch, case):
    # zero_signal: every group in rollout 1 has equal rewards, so its advantages and gradient norm are zero.
    wrong_advantage = case == "wrong_advantage"
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
            reward = 0.0 if case == "zero_signal" and step == 1 else float(index == 1)
            batch.append(
                {
                    "sample_index": 16 * step + index,
                    "tokens": [1, 2, 3],
                    "response_length": 1,
                    "response": "answer",
                    "status": "completed",
                    "reward": reward,
                    "metadata": {"query": "question", "expected_reward": reward},
                }
            )
        records += batch
        indices = [row["sample_index"] for row in batch]
        advantages = [torch.tensor([0.5 if index == 1 else -0.5 if index == 0 else 0.0]) for index in range(16)]
        if case == "zero_signal" and step == 1:
            advantages = [torch.tensor([0.0]) for _ in range(16)]
        if wrong_advantage:
            advantages[0] = torch.tensor([0.0])
        # TP2/DP1: one dump per data-parallel rank, keyed by global rank.
        torch.save({"rollout_data": {"sample_indices": indices, "advantages": advantages}}, debug / f"{step}_0.pt")
    (root / grpo_hooks.EVIDENCE).write_text("\n".join(json.dumps(row) for row in records) + "\n")
    second_norm = 0.0 if case == "zero_signal" else 1.0
    (root / "training.log").write_text(
        f"step 0: {{'train/step': 0, 'train/grad_norm': 1.0}}\nstep 1: {{'train/step': 1, 'train/grad_norm': {second_norm}}}\n"
    )
    base, export = tmp_path / "base", root / "hf-1"
    base.mkdir()
    export.mkdir()
    safetensors_torch.save_file({"model.norm.weight": torch.tensor([1.0])}, base / "model.safetensors")
    safetensors_torch.save_file({"model.norm.weight": torch.tensor([2.0])}, export / "model.safetensors")
    (export / ".complete").touch()
    registries = set()

    async def registered_reward(args, sample, **kwargs):
        registries.add(args.olmo_core.reward_config)
        assert sample.status == "completed"
        return sample.metadata["expected_reward"]

    monkeypatch.setattr(grpo_audit.rewards, "registered_reward", registered_reward)
    prepared = {"model": str(base), "data": {"reward_config": "/fixed/registry"}}
    if wrong_advantage:
        with pytest.raises(AssertionError):
            grpo_audit.audit(spec, prepared)
    else:
        result = grpo_audit.audit(spec, prepared)
        assert result["passed"] and result["teacher"] is None
        zero = case == "zero_signal"
        assert [row["mixed_groups"] for row in result["rollouts"]] == [1, 0 if zero else 1]
        assert [row["zero_signal"] for row in result["rollouts"]] == [False, zero]
        assert result["export"]["changed_tensors"] == 1
        assert registries == {"/fixed/registry"}


def test_conversion_tp_only_mesh_matches_trainer():
    command = conversion.conversion_command(
        "/usr/bin/python", "/native/miles", ["--num-layers", "28"], "/original/qwen3", "/converted/tp2", 2
    )
    assert "--nproc-per-node=2" in command
    assert command[command.index("--tensor-model-parallel-size") + 1] == "2"
    assert command[command.index("--pipeline-model-parallel-size") + 1] == "1"
    assert command[command.index("--hf-checkpoint") + 1] == "/original/qwen3"
    assert command[command.index("--save") + 1] == "/converted/tp2"


# The converter before 4772a67: the guard is the ``if`` test itself and ignores TP.
CONVERTER_DIRECT_GUARD = """
def get_args():
    args = parse_args(None)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.pipeline_model_parallel_size == 1 and world_size > 1 and not os.environ.get("CONVERT_KEEP_PP1"):
        args.pipeline_model_parallel_size = world_size
        args.decoder_last_pipeline_num_layers = args.num_layers // world_size
    validate_args(args)
    return args
"""
# The shape of the pinned (4772a67) converter: a named guard that also requires TP=CP=EP=ETP=1.
CONVERTER_NAMED_GUARD = """
def get_args():
    args = parse_args(None)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    auto_pipeline_parallel = (
        args.pipeline_model_parallel_size == 1
        and args.tensor_model_parallel_size == 1
        and args.context_parallel_size == 1
        and args.expert_model_parallel_size == 1
        and args.expert_tensor_parallel_size == 1
        and world_size > 1
        and not os.environ.get("CONVERT_KEEP_PP1")
    )
    if auto_pipeline_parallel:
        args.pipeline_model_parallel_size = world_size
        args.decoder_last_pipeline_num_layers = args.num_layers // world_size
    validate_args(args)
    return args
"""


def exercise_converter(source, tp, keep_pp1):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args")
    namespace = {
        "parse_args": lambda _: SimpleNamespace(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            num_layers=28,
            decoder_last_pipeline_num_layers=None,
        ),
        "os": SimpleNamespace(environ={"WORLD_SIZE": "2", **({"CONVERT_KEEP_PP1": "1"} if keep_pp1 else {})}),
    }

    def validate(args):
        assert 2 % (args.tensor_model_parallel_size * args.pipeline_model_parallel_size) == 0

    namespace["validate_args"] = validate
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native-converter", "exec"), namespace)
    return namespace["get_args"]()


def test_direct_converter_guard_reproduced_and_accepted():
    with pytest.raises(AssertionError):
        exercise_converter(CONVERTER_DIRECT_GUARD, tp=2, keep_pp1=False)
    conversion.verify_keep_pp1(CONVERTER_DIRECT_GUARD, "native-converter")
    args = exercise_converter(CONVERTER_DIRECT_GUARD, tp=2, keep_pp1=True)
    assert args.pipeline_model_parallel_size == 1 and args.decoder_last_pipeline_num_layers is None
    args = exercise_converter(CONVERTER_DIRECT_GUARD, tp=1, keep_pp1=False)
    assert args.pipeline_model_parallel_size == 2 and args.decoder_last_pipeline_num_layers == 14


def test_named_converter_guard_keeps_tp2_on_pp1_and_honors_the_switch():
    assert conversion.verify_keep_pp1(CONVERTER_NAMED_GUARD, "native-converter") == (
        hashlib.sha256(CONVERTER_NAMED_GUARD.encode()).hexdigest()
    )
    for keep_pp1 in (False, True):
        args = exercise_converter(CONVERTER_NAMED_GUARD, tp=2, keep_pp1=keep_pp1)
        assert args.pipeline_model_parallel_size == 1 and args.decoder_last_pipeline_num_layers is None
    assert exercise_converter(CONVERTER_NAMED_GUARD, tp=1, keep_pp1=False).pipeline_model_parallel_size == 2
    assert exercise_converter(CONVERTER_NAMED_GUARD, tp=1, keep_pp1=True).pipeline_model_parallel_size == 1


@pytest.mark.parametrize(
    "source",
    [
        CONVERTER_DIRECT_GUARD.replace("world_size > 1", "world_size > 2"),
        CONVERTER_NAMED_GUARD.replace('\n        and not os.environ.get("CONVERT_KEEP_PP1")', ""),
        CONVERTER_NAMED_GUARD.replace("if auto_pipeline_parallel:", "if world_size > 1:"),
        CONVERTER_NAMED_GUARD.replace("def get_args", "def parse_converter_args"),
        CONVERTER_DIRECT_GUARD + "\n" + CONVERTER_NAMED_GUARD,
    ],
    ids=["threshold", "no-switch", "unused-guard", "renamed", "ambiguous"],
)
def test_changed_converter_guard_fails_closed(source):
    with pytest.raises(ValueError, match="guard differs"):
        conversion.verify_keep_pp1(source, "changed")


def test_conversion_success_and_failure(tmp_path):
    with (tmp_path / "log").open("w") as stream:
        conversion.run_conversion([sys.executable, "-c", "print('done')"], dict(os.environ), stream)
        with pytest.raises(subprocess.CalledProcessError) as exc:
            conversion.run_conversion([sys.executable, "-c", "raise SystemExit(7)"], dict(os.environ), stream)
        assert exc.value.returncode == 7
    assert "done" in (tmp_path / "log").read_text()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Distributed conversion targets POSIX")
def test_conversion_timeout_kills_worker_even_after_parent_exits(tmp_path):
    child_pid = tmp_path / "child.pid"
    survivor = tmp_path / "survived"
    code = (
        "import os,signal,time,pathlib; pid=os.fork(); "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN) if pid==0 else None; "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())) if pid==0 else None; "
        "time.sleep(2); "
        f"pathlib.Path({str(survivor)!r}).write_text('alive') if pid==0 else None; time.sleep(30)"
    )
    try:
        with (tmp_path / "log").open("w") as stream, pytest.raises(subprocess.TimeoutExpired):
            conversion.run_conversion([sys.executable, "-c", code], dict(os.environ), stream, timeout=0.5)
        assert child_pid.exists(), "Child was never started"
        time.sleep(2)
        assert not survivor.exists(), "Worker survived the terminated parent and wrote after timeout"
    finally:
        if child_pid.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(child_pid.read_text()), signal.SIGKILL)


ARCHITECTURE_ARGS = ["--num-layers", "28", "--hidden-size", "2048"]


def sha(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def reuse_inputs(tmp_path):
    checkpoint, source, prepared = (tmp_path / key for key in ("checkpoint", "original-" + "e" * 12, "prepared"))
    checkpoint.mkdir()
    source.mkdir()
    prepared.mkdir()
    config = b'{"model_type":"qwen3"}'
    (source / "config.json").write_bytes(config)
    (prepared / "config.json").write_bytes(config)
    contents = {
        "latest_checkpointed_iteration.txt": b"release",
        "release/metadata.json": b"{}",
        "release/.metadata": b"audited-metadata",
        "release/__0_0.distcp": b"rank0-shard",
        "release/__1_0.distcp": b"rank1-shard",
    }
    files = {}
    for name, content in contents.items():
        path = checkpoint / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        files[name] = [path.stat().st_size, path.stat().st_mtime_ns]
    provenance = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "hf_checkpoint_source": str(source),
        "original_model_revision": "e" * 40,
        "hf_config_sha256": sha(config),
        "architecture": "qwen3-1.7B",
        "architecture_args_sha256": sha(json.dumps(ARCHITECTURE_ARGS, sort_keys=True).encode()),
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "format": "torch_dist",
        "tracker": "release",
        "native_converter_sha256": "c" * 64,
        "metadata_sha256": sha(contents["release/.metadata"]),
        "checkpoint_files": files,
        "save_experiment": "01M3RZYHGQJ3A7RJ5XKWEZEVYD",
        "load_experiment": "01M3SNWSYHMK2MQ2KAMA705VFC",
        "load_latest_job": "01M3SNWT2CAS05P6HB75TKG06C",
        "load_verified_utc": "2026-09-30T17:33:18.886194Z",
        "all_model_parameters_finite": True,
        "sampled_hf_weights_equal": True,
        "training_updates": 0,
        "scope": "Sampled initial-model load only.",
    }
    model = {"source": str(source), "architecture": "qwen3-1.7B", "native_checkpoint": provenance}
    return (
        provenance,
        model,
        {"tensor_parallel_size": 2},
        {"root": str(tmp_path / "fresh-output"), "assets": str(tmp_path / "fresh-assets")},
        prepared,
    )


def reuse(inputs):
    return checkpoint_reuse.reuse(*inputs, "c" * 64, ARCHITECTURE_ARGS)


def test_reuse_is_read_only_and_rechecks_source(reuse_inputs):
    provenance, _, _, output, _ = reuse_inputs
    before = {
        name: (Path(provenance["checkpoint_path"]) / name).read_bytes() for name in provenance["checkpoint_files"]
    }
    checkpoint, audit = reuse(reuse_inputs)
    assert str(checkpoint) == provenance["checkpoint_path"] and audit["passed"] and audit["mode"] == "reuse"
    checkpoint_reuse.verify_checkpoint(provenance, output)
    assert before == {name: (checkpoint / name).read_bytes() for name in before}
    assert not Path(output["root"]).exists() and not Path(output["assets"]).exists()
    path = checkpoint / "release/__0_0.distcp"
    path.write_bytes(b"changed")
    with pytest.raises(InputError, match="inventory"):
        checkpoint_reuse.verify_checkpoint(provenance, output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("hf_checkpoint_source", "/different/original"),
        ("checkpoint_path", "relative/asset"),
        ("checkpoint_path", "/asset/../partial"),
        ("architecture", "qwen3.5-2B"),
        ("tensor_parallel_size", 1),
        ("pipeline_parallel_size", 2),
        ("training_updates", 1),
        ("training_updates", False),
        ("format", "torch"),
        ("tracker", "1"),
        ("hf_config_sha256", "invalid"),
        ("native_converter_sha256", "a" * 63),
        ("metadata_sha256", "b" * 63),
        ("architecture_args_sha256", "bad"),
        ("original_model_revision", "main"),
        ("original_model_revision", "f" * 40),
        ("load_experiment", "unverified"),
        ("load_verified_utc", "2026-09-30T17:33:18"),
        ("all_model_parameters_finite", False),
        ("sampled_hf_weights_equal", False),
        ("checkpoint_files", {}),
    ],
)
def test_reuse_rejects_mismatched_declarations(reuse_inputs, key, value):
    reuse_inputs[0][key] = value
    with pytest.raises(InputError):
        reuse(reuse_inputs)


def test_reuse_rejects_missing_unknown_and_invalid_stats(reuse_inputs):
    provenance, model, trainer, output, _ = reuse_inputs
    for candidate in (
        {k: v for k, v in provenance.items() if k != "load_latest_job"},
        provenance | {"ignore_mismatch": True},
        provenance | {"checkpoint_files": provenance["checkpoint_files"] | {"release/__0_0.distcp": [True, 1]}},
    ):
        with pytest.raises(InputError):
            checkpoint_reuse.validate(candidate, model, trainer, output)


@pytest.mark.parametrize("field", ["root", "assets"])
def test_reuse_rejects_nested_output(reuse_inputs, field):
    provenance, _, _, output, _ = reuse_inputs
    output[field] = str(Path(provenance["checkpoint_path"]) / "would-mutate-input")
    with pytest.raises(InputError, match="separate"):
        reuse(reuse_inputs)


def test_reuse_rejects_resolved_output_alias(reuse_inputs, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(reuse_inputs[0]["checkpoint_path"], target_is_directory=True)
    reuse_inputs[3]["root"] = str(alias / "nested")
    with pytest.raises(InputError, match="separate"):
        reuse(reuse_inputs)


@pytest.mark.parametrize("change", ["missing", "extra", "mtime", "metadata", "tracker", "symlink"])
def test_reuse_rejects_partial_or_changed_runtime_asset(reuse_inputs, change):
    provenance = reuse_inputs[0]
    checkpoint = Path(provenance["checkpoint_path"])
    path = checkpoint / "release/__0_0.distcp"
    if change == "missing":
        path.unlink()
    elif change == "extra":
        (checkpoint / "unrecorded").write_text("extra")
    elif change == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    elif change in ("metadata", "tracker"):
        path = checkpoint / ("release/.metadata" if change == "metadata" else "latest_checkpointed_iteration.txt")
        path.write_bytes(b"changed-metadata" if change == "metadata" else b"partial")
        provenance["checkpoint_files"][path.relative_to(checkpoint).as_posix()] = [
            path.stat().st_size,
            path.stat().st_mtime_ns,
        ]
    else:
        path.unlink()
        path.symlink_to(checkpoint / "release/__1_0.distcp")
    snapshot = {p.name: p.lstat().st_mtime_ns for p in checkpoint.rglob("*")}
    with pytest.raises(InputError):
        reuse(reuse_inputs)
    assert snapshot == {p.name: p.lstat().st_mtime_ns for p in checkpoint.rglob("*")}
    assert not Path(reuse_inputs[3]["assets"]).exists()


def test_reuse_rejects_runtime_config_converter_and_architecture(reuse_inputs):
    with pytest.raises(InputError, match="converter hash"):
        checkpoint_reuse.reuse(*reuse_inputs, "d" * 64, ARCHITECTURE_ARGS)
    with pytest.raises(InputError, match="architecture arguments"):
        checkpoint_reuse.reuse(*reuse_inputs, "c" * 64, ["--different-profile"])
    (reuse_inputs[4] / "config.json").write_text('{"model_type":"changed"}')
    with pytest.raises(InputError, match="config hash"):
        reuse(reuse_inputs)


def test_config_and_submitted_spec_preserve_inline_reuse_bindings(reuse_inputs):
    provenance, model, _, output, _ = reuse_inputs
    document = {
        "schema_version": 1,
        "name": "reuse-test",
        "model": model,
        "output": output,
        "training": {"algorithm": "grpo"},
        "trainer": {"backend": "megatron", "gpus": 2, "tensor_parallel_size": 2},
        "inference": {"gpus": 2},
        "data": {
            "prompt_data": "/data/train",
            "eval_prompt_data": ["aime", "/data/eval"],
            "reward_config": "/data/reward",
        },
        "launch": {"gpus_per_replica": 4},
    }
    spec = grpo_config.MegatronGRPORunSpec.from_dict(document)
    assert spec.to_dict()["model"]["native_checkpoint"] == provenance
    assert grpo_config.MegatronGRPORunSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    assert launch.specification("native-image", spec)["tasks"][0]["resources"]["gpuCount"] == 4
    old = copy.deepcopy(document)
    old["model"].pop("native_checkpoint")
    assert grpo_config.MegatronGRPORunSpec.from_dict(old).model["native_checkpoint"] == {}
    document["trainer"]["tensor_parallel_size"] = 1
    with pytest.raises(InputError, match="parallelism"):
        grpo_config.MegatronGRPORunSpec.from_dict(document)


EXAMPLE = Path(__file__).resolve().parents[1] / "tests/miles/fixtures/megatron_grpo"


@pytest.mark.parametrize("path", sorted(EXAMPLE.glob("*.toml")), ids=lambda path: path.name)
def test_checked_in_examples_validate_and_render(path):
    spec = specs.load(path)
    assert isinstance(spec, grpo_config.MegatronGRPORunSpec)
    task = launch.specification("pending-build", spec)["tasks"][0]
    assert task["resources"]["gpuCount"] == spec.allocation()["gpus_per_replica"] == 8
