"""Reject partial, mismatched or mutable initial assets before native training."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from open_instruct.miles import launch, megatron_grpo_assets, megatron_grpo_config
from open_instruct.miles.errors import InputError

ARCHITECTURE_ARGS = ["--num-layers", "28", "--hidden-size", "2048"]


def sha(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def inputs(tmp_path):
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
    return megatron_grpo_assets.reuse(*inputs, "c" * 64, ARCHITECTURE_ARGS)


def test_reuse_is_read_only_and_rechecks_source(inputs):
    provenance, _, _, output, _ = inputs
    before = {
        name: (Path(provenance["checkpoint_path"]) / name).read_bytes() for name in provenance["checkpoint_files"]
    }
    checkpoint, audit = reuse(inputs)
    assert str(checkpoint) == provenance["checkpoint_path"] and audit["passed"] and audit["mode"] == "reuse"
    megatron_grpo_assets.verify_checkpoint(provenance, output)
    assert before == {name: (checkpoint / name).read_bytes() for name in before}
    assert not Path(output["root"]).exists() and not Path(output["assets"]).exists()
    path = checkpoint / "release/__0_0.distcp"
    path.write_bytes(b"changed")
    with pytest.raises(InputError, match="inventory"):
        megatron_grpo_assets.verify_checkpoint(provenance, output)


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
def test_reject_mismatched_declarations(inputs, key, value):
    inputs[0][key] = value
    with pytest.raises(InputError):
        reuse(inputs)


def test_reject_missing_unknown_and_invalid_stats(inputs):
    provenance, model, trainer, output, _ = inputs
    for candidate in (
        {k: v for k, v in provenance.items() if k != "load_latest_job"},
        provenance | {"ignore_mismatch": True},
        provenance | {"checkpoint_files": provenance["checkpoint_files"] | {"release/__0_0.distcp": [True, 1]}},
    ):
        with pytest.raises(InputError):
            megatron_grpo_assets.validate(candidate, model, trainer, output)


@pytest.mark.parametrize("field", ["root", "assets"])
def test_reject_nested_output(inputs, field):
    provenance, _, _, output, _ = inputs
    output[field] = str(Path(provenance["checkpoint_path"]) / "would-mutate-input")
    with pytest.raises(InputError, match="separate"):
        reuse(inputs)


def test_reject_resolved_output_alias(inputs, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(inputs[0]["checkpoint_path"], target_is_directory=True)
    inputs[3]["root"] = str(alias / "nested")
    with pytest.raises(InputError, match="separate"):
        reuse(inputs)


@pytest.mark.parametrize("change", ["missing", "extra", "mtime", "metadata", "tracker", "symlink"])
def test_reject_partial_or_changed_runtime_asset(inputs, change):
    provenance = inputs[0]
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
        reuse(inputs)
    assert snapshot == {p.name: p.lstat().st_mtime_ns for p in checkpoint.rglob("*")}
    assert not Path(inputs[3]["assets"]).exists()


def test_reject_runtime_config_converter_and_architecture(inputs):
    with pytest.raises(InputError, match="converter hash"):
        megatron_grpo_assets.reuse(*inputs, "d" * 64, ARCHITECTURE_ARGS)
    with pytest.raises(InputError, match="architecture arguments"):
        megatron_grpo_assets.reuse(*inputs, "c" * 64, ["--different-profile"])
    (inputs[4] / "config.json").write_text('{"model_type":"changed"}')
    with pytest.raises(InputError, match="config hash"):
        reuse(inputs)


def test_config_and_submitted_spec_preserve_inline_bindings(inputs):
    provenance, model, _, output, _ = inputs
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
    spec = megatron_grpo_config.MegatronGRPORunSpec.from_dict(document)
    assert spec.to_dict()["model"]["native_checkpoint"] == provenance
    assert megatron_grpo_config.MegatronGRPORunSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    assert launch.specification("native-image", spec)["tasks"][0]["resources"]["gpuCount"] == 4
    old = copy.deepcopy(document)
    old["model"].pop("native_checkpoint")
    assert megatron_grpo_config.MegatronGRPORunSpec.from_dict(old).model["native_checkpoint"] == {}
    document["trainer"]["tensor_parallel_size"] = 1
    with pytest.raises(InputError, match="parallelism"):
        megatron_grpo_config.MegatronGRPORunSpec.from_dict(document)
