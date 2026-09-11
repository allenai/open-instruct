"""Negative controls for read-only native optimizer-boundary diagnostics."""

from types import SimpleNamespace

import pytest
import torch
from scripts.miles import update_zero_gradient_capture as capture


class Optimizer:
    def __init__(self):
        self.intake = False
        self.updates = 0

    def _clip_grad(self):
        raise AssertionError("actual clip must not execute")

    def _step_foreach(self):
        self.updates += 1

    def step(self):
        self.intake = True
        self._clip_grad()
        self._step_foreach()


def test_core_capture_follows_intake_and_prohibits_updates(monkeypatch):
    optimizer = Optimizer()
    worker = SimpleNamespace(optimizer=optimizer, args=SimpleNamespace(train_backend="olmo_core"))
    original = optimizer._clip_grad

    def observe(_worker):
        assert optimizer.intake
        return {"router": {"gradient": torch.tensor([3.0])}}

    monkeypatch.setattr(capture, "router_gradients", observe)
    sink = {}
    with capture.stop_at_boundary(worker, sink) as calls:
        with pytest.raises(capture.CapturedBoundary):
            optimizer.step()
        with pytest.raises(RuntimeError, match="forbidden"):
            optimizer._step_foreach()
    assert optimizer._clip_grad == original
    assert optimizer.updates == 0 and len(calls) == 1
    assert sink["router"]["gradient"].item() == 3


def test_megatron_boundary_and_underlying_adam_both_guarded(monkeypatch):
    inner = SimpleNamespace(step=lambda: pytest.fail("Adam executed"))
    optimizer = SimpleNamespace(step=lambda: pytest.fail("optimizer executed"), optimizer=inner)
    worker = SimpleNamespace(optimizer=optimizer, args=SimpleNamespace(train_backend="megatron"))
    monkeypatch.setattr(capture, "router_gradients", lambda _worker: {"router": {"gradient": torch.ones(2)}})
    original = optimizer.step
    with capture.stop_at_boundary(worker, {}) as calls:
        with pytest.raises(RuntimeError, match="forbidden"):
            inner.step()
        with pytest.raises(capture.CapturedBoundary):
            optimizer.step()
    assert optimizer.step == original and len(calls) == 1


def test_decomposition_reports_opposition_and_nonzero_superposition_residual():
    def arm(values):
        return {"router": {"gradient": torch.tensor(values)}}

    arms = {"policy": arm([1.0, 0]), "auxiliary": arm([-2.0, 0]), "combined": arm([-1.0, 0.25])}
    result = capture.decomposition(arms)["router"]
    assert result["policy_auxiliary_cosine"] == -1
    assert result["superposition_residual_norm"] == 0.25
    arms["combined"] = {}
    with pytest.raises(ValueError, match="ownership"):
        capture.decomposition(arms)


def test_tensor_state_fingerprint_detects_bit_changes_and_dtype():
    original = torch.tensor([1.0, 2.0], dtype=torch.float32)
    before = capture.tree_hash({"main": original})
    assert capture.tree_hash({"main": original.clone()}) == before
    original[0] += 1
    assert capture.tree_hash({"main": original}) != before
    assert capture.tree_hash({"main": original.bfloat16()}) != capture.tree_hash({"main": original})


def test_explicit_advantage_and_anchor_axes_reject_malformed_payload():
    payload = {
        "schema_version": 1,
        "auxiliary": {"lb": 0.01, "z": 1e-5},
        "cases": [
            {
                "case_id": "a",
                "input_ids": [1, 2, 3],
                "response_length": 2,
                "loss_mask": [1, 1],
                "old_log_probs": [-2.0, -3.0],
                "advantages": [1.0, -1.0],
            }
        ],
    }
    assert capture.validate_payload(payload, 1, 4) == payload["cases"]
    payload["cases"][0]["advantages"] = [float("nan"), 1.0]
    with pytest.raises(ValueError, match="advantages"):
        capture.validate_payload(payload, 1, 4)


def test_objective_boundary_detects_changed_mask_and_token_order():
    rollout = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1.0, 0.0])],
        "advantages": [torch.tensor([1.0, -1.0])],
        "rollout_log_probs": [torch.tensor([-2.0, -3.0])],
    }
    batch = dict(rollout, unconcat_tokens=rollout["tokens"])
    capture.assert_objective_batch(batch, rollout, 0)
    batch = dict(batch, loss_masks=[torch.ones(2)])
    with pytest.raises(ValueError, match="loss_masks"):
        capture.assert_objective_batch(batch, rollout, 0)
    batch = dict(rollout, unconcat_tokens=[torch.tensor([3, 2, 1])])
    with pytest.raises(ValueError, match="token IDs"):
        capture.assert_objective_batch(batch, rollout, 0)
