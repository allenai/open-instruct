"""Numerical audit comparisons and launch isolation."""

import json

import pytest
import torch

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.distillation import audit_trace, numerical_audit
from open_instruct.miles.execution import launch


def test_comparison_exposes_centered_noise():
    result = numerical_audit.compare([-1, -2], [-1.1, -1.9])
    assert result["signed_mean"] == pytest.approx(0)
    assert result["mae"] == pytest.approx(0.1)
    assert result["exact_fraction"] == 0
    assert numerical_audit.compare([-1], [-1])["exact_fraction"] == 1
    for left, right in [([], []), ([-1], [-1, -2]), ([float("nan")], [-1])]:
        with pytest.raises(ValueError):
            numerical_audit.compare(left, right)


def test_trace_copies_tensors_and_ignores_runtime_objects():
    original = torch.tensor([1.0])
    copied = audit_trace.tensors((original, object()))
    original.add_(1)
    assert len(copied) == 1
    assert copied[0].item() == 1


def test_capture_records_loaded_weights_and_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test CPU")
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_norm = torch.nn.Linear(2, 2)
    hook = audit_trace.factory({"root": str(tmp_path)})
    hook(model, (), None)
    (tmp_path / "capture").touch()
    expected = model.model.embed_norm(torch.ones(1, 2))
    hook(model, (), expected)
    saved = torch.load(tmp_path / "model.embed_norm.pt", weights_only=True)
    assert torch.equal(saved["outputs"], expected)
    weights = json.loads((tmp_path / "weights.json").read_text())
    assert weights["model.embed_norm.weight"]["shape"] == [2, 2]
    assert len(weights["model.embed_norm.weight"]["sha256"]) == 64


def test_audit_launch_does_not_train():
    spec = RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "numerical-audit",
            "model": {"source": "/model"},
            "output": {"root": "/output/audit"},
            "data": {"prompt_data": "/data", "reward_config": "/rewards"},
            "launch": {"gpus_per_replica": 2, "auto_resume": False, "env": {"OI_MILES_OPD_NUMERICAL_AUDIT": "1"}},
            "trainer": {"gpus": 1, "expert_parallel_size": 1},
            "inference": {"placement_mode": "disaggregated", "gpus": 1},
        }
    )
    task = launch.specification("image", spec)["tasks"][0]
    assert "python -m open_instruct.miles.distillation.numerical_audit" in task["arguments"][0]
    assert "python -m open_instruct.miles train" not in task["arguments"][0]
    assert task["resources"]["gpuCount"] == 2
