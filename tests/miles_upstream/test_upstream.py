"""CPU contracts for upstream source identity, TOML compatibility and submission."""

import argparse
import copy
import json
import os
import subprocess
from enum import Enum
from pathlib import Path

import pytest
from scripts.miles_upstream import build_image, config, execute, launch

from open_instruct.miles.configuration import options
from open_instruct.miles.errors import InputError


@pytest.fixture
def schema(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-backend", choices=config.BACKENDS)
    for name in ("hf-checkpoint", "save", "load", "prompt-data", "wandb-mode", "rm-type", "future-upstream-option"):
        parser.add_argument(f"--{name}", type=str)
    for name in (
        "actor-num-nodes",
        "actor-num-gpus-per-node",
        "num-gpus-per-node",
        "rollout-num-gpus",
        "rollout-num-gpus-per-engine",
        "num-rollout",
        "save-interval",
        "eval-num-gpus",
    ):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--eval-prompt-data", nargs="+", type=str)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sglang-disable-radix-cache", action="store_true")
    parser.add_argument("--fully-async", action="store_true")
    parser.add_argument("--colocate", action="store_true")
    parser.add_argument("--use-critic", action="store_true")
    parser.add_argument("--config", type=str)
    path = tmp_path / "fsdp.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "fsdp",
                "build_id": "build-123",
                "source": {"repository": build_image.REPOSITORY, "revision": "a" * 40},
                "options": options.compact_options(options.describe_parser(parser)),
            }
        )
    )
    return path


@pytest.fixture
def document():
    return {
        "schema_version": 1,
        "name": "upstream-test",
        "model": {"source": "model", "format": "hf"},
        "output": {"root": "output"},
        "trainer": {"backend": "fsdp", "gpus": 1, "activation_recompute": True},
        "inference": {"gpus": 1, "rollout_gpus_per_node": 2, "placement_mode": "disaggregated", "radix_cache": False},
        "data": {"prompt_data": "prompts.jsonl"},
        "training": {"num_rollouts": 4, "save_interval": 2},
        "optimizer": {"learning_rate": 1e-6},
    }


def compile_plan(document, schema):
    return config.compile_config(document, schema.parent / "run.toml", schema)


def test_shared_toml_maps_to_native_parser(document, schema):
    plan = compile_plan(document, schema)
    assert plan["native"]["train_backend"] == "fsdp"
    assert plan["native"]["lr"] == 1e-6
    assert plan["native"]["hf_checkpoint"] == str(schema.parent / "model")
    assert plan["native"]["prompt_data"] == str(schema.parent / "prompts.jsonl")
    assert plan["native"]["save"] == str(schema.parent / "output/checkpoints")
    assert "--gradient-checkpointing" in plan["argv"]
    assert "--sglang-disable-radix-cache" in plan["argv"]
    assert plan["entrypoint"] == "train.py"
    assert launch.launch_settings(plan)["gpus_per_replica"] == 2


def test_enum_choices_round_trip_through_json_schema(tmp_path):
    class Backend(Enum):
        auto = 1
        flash = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-backend", type=lambda value: Backend[value], choices=list(Backend))
    schema = tmp_path / "enum.json"
    schema.write_text(json.dumps({"schema_version": 1, "options": options.describe_parser(parser)}))
    argv = options.encode_options({"attention_backend": "flash"}, schema_path=schema)
    assert parser.parse_args(argv).attention_backend == Backend.flash
    with pytest.raises(InputError):
        options.encode_options({"attention_backend": 2}, schema_path=schema)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("core", "attention_backend", "torch"),
        ("runtime", "row_specialization", "dynamic"),
        ("compiler_cache", "enabled", True),
        ("data", "tasks", []),
        ("trainer", "fp32_lm_head", True),
        ("miles", "novel_fork_flag", 1),
        ("miles", "config", "overrides.yaml"),
        ("model", "format", "core"),
        ("output", "export_hf", True),
    ],
)
def test_unsupported_fields_are_not_dropped(document, schema, section, key, value):
    document.setdefault(section, {})[key] = value
    with pytest.raises(InputError):
        compile_plan(document, schema)


def test_new_upstream_flags_do_not_use_core_snapshot(document, schema):
    document["miles"] = {"future_upstream_option": "--literal-with-spaces a b"}
    plan = compile_plan(document, schema)
    assert "--future-upstream-option=--literal-with-spaces a b" in plan["argv"]
    with pytest.raises(InputError, match="Unknown MILES option"):
        options.encode_options(document["miles"])


def test_conflicting_native_and_shared_options_fail(document, schema):
    document["miles"] = {"lr": 2e-6}
    with pytest.raises(InputError, match="Conflicting.*optimizer.learning_rate.*miles.lr"):
        compile_plan(document, schema)


def test_invalid_types_and_backend_schema(document, schema):
    document["trainer"]["gpus"] = True
    with pytest.raises(InputError, match="expects int"):
        compile_plan(document, schema)
    document["trainer"]["backend"] = "megatron"
    with pytest.raises(InputError, match="upstream megatron schema"):
        compile_plan(document, schema)


def test_async_and_boolean_false(document, schema):
    document["async"] = {"fully_async": True}
    document["trainer"]["activation_recompute"] = False
    plan = compile_plan(document, schema)
    assert plan["entrypoint"] == "train_async.py"
    assert "--no-gradient-checkpointing" in plan["argv"]


def test_eval_paths_resolve_against_original_config(document, schema):
    document["data"]["eval_prompt_data"] = ["eval", "eval.jsonl"]
    assert compile_plan(document, schema)["native"]["eval_prompt_data"] == ["eval", str(schema.parent / "eval.jsonl")]


def test_overrides(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[trainer]\nbackend = "fsdp"\n')
    document, resolved = config.read(path, ["trainer.gpus=2", 'launch.env.WANDB_MODE="offline"'])
    assert document["trainer"]["gpus"] == 2
    assert document["launch"]["env"]["WANDB_MODE"] == "offline"
    assert resolved == path


@pytest.mark.parametrize(
    "native",
    [{"actor_num_nodes": 2}, {"use_critic": True}, {"eval_num_gpus": 1}, {"num_gpus_per_node": 8}, {"colocate": True}],
)
def test_reject_unaccounted_resources(document, schema, native):
    plan = compile_plan(document, schema)
    plan["native"].update(native)
    with pytest.raises(InputError):
        launch.launch_settings(plan)


def test_beaker_spec_preserves_scheduling_and_secret_references(document, schema):
    document["launch"] = {
        "priority": "urgent",
        "min_runtime": 0,
        "timeout": "20m",
        "secrets": {"HF_TOKEN": "hf-secret"},
    }
    plan = compile_plan(document, schema)
    spec = launch.specification("immutable-image", plan)
    task = spec["tasks"][0]
    assert task["context"] == {"priority": "urgent", "minRuntime": "0s", "autoResume": False}
    assert task["timeout"] == "20m"
    assert task["resources"]["gpuCount"] == 2
    assert {"name": "HF_TOKEN", "secret": "hf-secret"} in task["envVars"]
    assert "olmo" not in task["arguments"][0]
    subprocess.run(["bash", "-n", "-c", task["arguments"][0]], check=True)


def test_mount_and_resume_checks(document, schema):
    document["output"]["root"] = "/weka/test/run"
    document["miles"] = {"load": "/weka/test/run/checkpoints"}
    document["launch"] = {"auto_resume": True}
    plan = compile_plan(document, schema)
    with pytest.raises(InputError, match="Missing WEKA mount"):
        launch.launch_settings(plan)
    plan["launch"]["weka_mounts"] = [{"weka": "test", "mount_path": "/weka/test"}]
    assert launch.launch_settings(plan)["auto_resume"]
    plan["native"]["load"] = "/weka/test/other"
    with pytest.raises(InputError, match="auto_resume requires"):
        launch.launch_settings(plan)


@pytest.mark.parametrize("env", [{"PYTHONPATH": "x"}, {"HF_TOKEN": "secret"}])
def test_env_cannot_override_runtime_or_embed_secrets(document, schema, env):
    plan = compile_plan(document, schema)
    plan["launch"]["env"] = env
    with pytest.raises(InputError):
        launch.launch_settings(plan)


def test_execution_rechecks_identity_and_argv(document, schema, monkeypatch):
    plan = compile_plan(document, schema)
    monkeypatch.setattr(execute.subprocess, "check_output", lambda *a, **k: "a" * 40)
    cmd = execute.verify(plan, root=Path("/root/miles"), schemas=schema.parent)
    assert cmd[1:] == ["/root/miles/train.py", *plan["argv"]]
    bad = copy.deepcopy(plan)
    bad["build_id"] = "wrong-image"
    with pytest.raises(InputError, match="different image"):
        execute.verify(bad, schemas=schema.parent)
    bad = copy.deepcopy(plan)
    bad["argv"].append("--surprise")
    with pytest.raises(InputError, match="native arguments"):
        execute.verify(bad, schemas=schema.parent)


def test_build_has_only_pinned_public_inputs():
    lock = build_image.read_lock(build_image.DEFAULT_LOCK)
    files = build_image.build_inputs(lock)
    identity = build_image.fingerprint(files)
    cmd = build_image.command(lock, identity, "test-image", "application", "/tmp/context")
    assert f"MILES_REVISION={lock['miles']['revision']}" in cmd
    assert f"BASE_IMAGE={lock['base_image']}" in cmd
    assert "--secret" not in cmd
    assert not any("AGENTS" in name or "options.json" in name or "olmo-core" in name for name in files)
    changed = files | {"runtime.lock.json": b"changed"}
    assert build_image.fingerprint(changed) != identity
    assert build_image.fingerprint(files | {"adapter/change.py": b"change"}) != identity


@pytest.mark.parametrize(
    "key,value", [("base_image", "radixark/miles:latest"), ("image_inputs", ""), ("platform", "linux/arm64")]
)
def test_invalid_locks_fail_before_build(tmp_path, key, value):
    lock = build_image.read_lock(build_image.DEFAULT_LOCK)
    lock[key] = value
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        build_image.read_lock(path)


@pytest.mark.parametrize("matches", [False, True])
def test_refresh_checks_dependencies_before_replacing_lock(tmp_path, monkeypatch, matches):
    path = tmp_path / "lock.json"
    path.write_text("original")
    digest = "sha256:" + "b" * 64
    expected = "c" * 64

    def inspect(image, field):
        if field == "Manifest":
            return {"manifests": [{"digest": digest, "platform": {"os": "linux", "architecture": "amd64"}}]}
        assert image == f"radixark/miles@{digest}"
        return {"os": "linux", "architecture": "amd64", "config": {"Labels": {"miles.image-inputs": expected}}}

    def output(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return "a" * 40
        return expected if matches else "d" * 64

    monkeypatch.setattr(build_image, "inspect_registry", inspect)
    monkeypatch.setattr(build_image, "output", output)
    monkeypatch.setattr(build_image.subprocess, "run", lambda *a, **k: None)
    if matches:
        build_image.refresh(path, "main", "radixark/miles:latest")
        lock = build_image.read_lock(path)
        assert lock["base_image"] == f"radixark/miles@{digest}"
        assert lock["miles"]["revision"] == "a" * 40
    else:
        with pytest.raises(ValueError, match="dependency inputs differ"):
            build_image.refresh(path, "main", "radixark/miles:latest")
        assert path.read_text() == "original"


def test_submit_rejects_mismatched_image_before_allocating(document, schema, monkeypatch, tmp_path):
    image = "01M24E7MSDGN2QFW1T8Z31BCKS"
    calls = []

    def output(command, **kwargs):
        calls.append(command)
        assert command[:3] == ["beaker", "image", "get"]
        return json.dumps([{"id": image, "description": "different-build"}])

    monkeypatch.setattr(launch.subprocess, "check_output", output)
    with pytest.raises(InputError, match="does not match"):
        launch.submit(image, compile_plan(document, schema), tmp_path / "receipt.json")
    assert len(calls) == 1


@pytest.mark.parametrize("cuda", [["--cuda-version", "12"], ["--cuda-version=13"]])
@pytest.mark.parametrize("upstream_first", [True, False])
def test_mixed_cuda_modes_fail_before_external_commands(tmp_path, cuda, upstream_first):
    flags = ["--miles-upstream", *cuda] if upstream_first else [*cuda, "--miles-upstream"]
    result = subprocess.run(
        ["/bin/bash", str(build_image.ROOT / "scripts/train/build_image_and_launch.sh"), *flags, "run.toml"],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "--miles-upstream cannot be combined with --cuda-version" in result.stderr
