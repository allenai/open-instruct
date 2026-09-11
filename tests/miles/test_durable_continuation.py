"""Adversarial checks for bounded hashing and whole-state continuation comparisons."""

import copy

import pytest
import torch
from scripts.miles import durable_continuation as gate
from scripts.miles import launch_durable_continuation as launcher


def state():
    return {
        "model": {"parameter": gate.tensor_record(torch.tensor([1, 2], dtype=torch.bfloat16))},
        "optimizer": {"parameter.main": gate.tensor_record(torch.tensor([1.001, 2.001]))},
        "scheduler": {"last_epoch": 2, "lr_decay_steps": 4},
        "clock": {"completed_steps": 2, "next_rollout_id": 2, "published_step": 1},
        "trainer_global_step": 2,
        "cursor": {"sample_offset": 2, "sample_index": 8},
        "rng": {"torch": "frozen"},
    }


def test_hash_chunking_preserves_exact_tensor_bytes(monkeypatch):
    value = torch.arange(128, dtype=torch.float32)
    before = gate.tensor_record(value)
    monkeypatch.setattr(gate, "CHUNK_BYTES", 16)
    assert gate.tensor_record(value) == before
    value[-1] += 1
    assert gate.tensor_record(value)["sha256"] != before["sha256"]


def test_hash_catches_master_changes_below_bf16_resolution():
    value = torch.tensor([1.0], dtype=torch.float32)
    updated = value + 1e-6
    assert torch.equal(value.bfloat16(), updated.bfloat16())
    assert gate.tensor_record(value) != gate.tensor_record(updated)


@pytest.mark.parametrize(
    "category", ["model", "optimizer", "scheduler", "clock", "trainer_global_step", "cursor", "rng"]
)
def test_comparison_rejects_each_contract_component(category):
    expected = state()
    assert gate.compare_states(expected, copy.deepcopy(expected)) == []
    actual = copy.deepcopy(expected)
    if category in ("model", "optimizer"):
        actual[category][next(iter(actual[category]))]["sha256"] = "changed"
    else:
        actual[category] = "changed"
    assert gate.compare_states(expected, actual)


def test_comparison_rejects_missing_optimizer_tensor():
    expected = state()
    actual = copy.deepcopy(expected)
    actual["optimizer"].clear()
    assert gate.compare_states(expected, actual) == ["optimizer: state keys differ"]


def test_launcher_preserves_bounded_two_gpu_placement():
    task = launcher.specification("qualified-image")["tasks"][0]
    assert task["resources"]["gpuCount"] == 2
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["context"] == {"priority": "urgent", "minRuntime": "60m", "autoResume": False}
    assert task["timeout"] == "90m"
    command = task["arguments"][0]
    assert "control split resumed" in command and "--world 2" in command
