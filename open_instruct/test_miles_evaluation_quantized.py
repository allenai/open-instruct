"""Quantized evaluation preserves source provenance and resumes committed exports."""

import copy
import hashlib
import json

import pytest

from open_instruct.miles.evaluation import evaluation_runner, serving_artifact
from open_instruct.test_miles_evaluation_serving_artifact import ROOT


def test_quantized_snapshot_converts_once_and_keeps_original(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"olmo3moe"}')
    receipt = {"checkpoint": str(source), "evaluation": {"quantized_export": "fp8"}}
    calls = []

    def convert(args, **kwargs):
        calls.append(args)
        target = tmp_path / "serialized-fp8"
        target.mkdir()
        (target / "config.json").write_text(
            json.dumps({"model_type": "olmo3moe", "quantization_config": {"weight_block_size": [128, 128]}})
        )

    monkeypatch.setattr(evaluation_runner.subprocess, "run", convert)
    first = evaluation_runner.quantized_snapshot(receipt, tmp_path)
    second = evaluation_runner.quantized_snapshot(receipt, tmp_path)
    assert first == second
    assert len(calls) == 1
    assert first["checkpoint"] == str(source)
    assert first["serving_checkpoint"] == str(tmp_path / "serialized-fp8")
    assert "--device" in calls[0] and calls[0][-1] == "cuda"
    (source / "config.json").write_text('{"model_type":"changed"}')
    with pytest.raises(ValueError, match="provenance"):
        evaluation_runner.quantized_snapshot(receipt, tmp_path)


def test_quantized_snapshot_refuses_incomplete_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (tmp_path / "serialized-fp8").mkdir()
    with pytest.raises(ValueError, match="Uncommitted"):
        evaluation_runner.quantized_snapshot(
            {"checkpoint": str(source), "evaluation": {"quantized_export": "fp8"}}, tmp_path
        )


def test_quantized_snapshot_rebinds_policy_without_mutating_receipt(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "runtime_lock_sha256": hashlib.sha256(
                    (ROOT / "runtime/miles/runtime.lock.json").read_bytes()
                ).hexdigest(),
                "model_config_sha256": hashlib.sha256(b"{}").hexdigest(),
            }
        )
    )
    config = {
        "quantized_export": "fp8",
        "kda_policy": {
            "path": str(policy),
            "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "core_compat": "auto",
        },
    }
    config["serving_artifact"] = serving_artifact.build(config["kda_policy"])
    receipt = {"checkpoint": str(source), "evaluation": config}
    original = copy.deepcopy(receipt)

    def convert(*args, **kwargs):
        target = tmp_path / "serialized-fp8"
        target.mkdir()
        (target / "config.json").write_text('{"quantization_config":{"weight_block_size":[128,128]}}')

    monkeypatch.setattr(evaluation_runner.subprocess, "run", convert)
    result = evaluation_runner.quantized_snapshot(receipt, tmp_path)
    files = evaluation_runner.validate_serving_artifact(result["evaluation"])
    assert (
        json.loads(files["policy.json"])["model_config_sha256"]
        == hashlib.sha256((tmp_path / "serialized-fp8/config.json").read_bytes()).hexdigest()
    )
    assert receipt == original
