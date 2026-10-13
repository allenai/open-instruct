"""Numerical audit comparisons and launch isolation."""

import json
from types import SimpleNamespace

import pytest
import torch

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.distillation import audit_trace, audit_tuning, numerical_audit
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


def test_inner_model_trace_survives_outer_forward_bypass(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test CPU")

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_norm = torch.nn.Linear(2, 2)

        def forward(self, value):
            return self.embed_norm(value)

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, value):
            return self.model(value)

    model = Outer()
    model.model.register_forward_hook(audit_trace.factory({"root": str(tmp_path), "base": "model"}))
    model.forward(torch.ones(1, 2))
    (tmp_path / "capture").touch()
    expected = model.forward(torch.ones(1, 2))
    saved = torch.load(tmp_path / "model.embed_norm.pt", weights_only=True)
    assert torch.equal(saved["outputs"], expected)
    assert (tmp_path / "weights.json").exists()


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


def test_tuning_replay_checks_real_choices_and_scoring_use(tmp_path, monkeypatch):
    marker = tmp_path / "capture"
    tuner = SimpleNamespace(cache={}, configs=["config-a", "config-b"])
    module = SimpleNamespace(kernel=SimpleNamespace(fn=tuner))
    monkeypatch.setattr(audit_tuning.importlib, "import_module", lambda name: module)
    document = {"fla.ops.test.kernel": {"(8, 'torch.bfloat16')": "config-b"}}
    replay = audit_tuning.install(document, marker)
    key = (8, "torch.bfloat16")
    assert tuner.cache[key] == "config-b"
    with pytest.raises(RuntimeError, match="never used"):
        audit_tuning.report(replay)
    marker.touch()
    assert tuner.cache[key] == "config-b"
    assert audit_tuning.report(replay)["fla.ops.test.kernel"][repr(key)]["scoring_hits"] == 1
    with pytest.raises(RuntimeError, match="replace pinned"):
        tuner.cache[key] = "config-a"
    with pytest.raises(ValueError, match="unavailable"):
        audit_tuning.install({"fla.ops.test.kernel": {repr(key): "unavailable"}}, marker)
