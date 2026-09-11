"""Failure-sensitive replay checks against the actual Core router."""

from types import SimpleNamespace

import pytest
import torch
from olmo_core.nn.moe.v2 import replay
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from scripts.miles import exercise_controls, launch_control_exercise
from torch import nn
from torch.utils.checkpoint import checkpoint

from open_instruct.miles import data, replay_diagnostics


@pytest.mark.parametrize("corrupt", [False, True])
def test_observes_routes_in_forward_and_recomputation(monkeypatch, corrupt):
    router = MoERouterConfigV2(d_model=8, num_experts=4, top_k=2).build()
    nn.init.normal_(router.weight, std=0.1)
    model = nn.Module()
    model.blocks = nn.ModuleDict({"0": nn.Module()})
    model.blocks["0"].routed_experts_router = router
    module = SimpleNamespace(model=model)
    actor = SimpleNamespace(args=None, clock=SimpleNamespace(next_rollout_id=0))
    batch = {"tokens": torch.tensor([[1, 2, 3]]), "rollout_routed_experts": [torch.tensor([[[1, 3]], [[2, 3]]])]}
    rows = []
    monkeypatch.setattr(replay_diagnostics.contract, "record", lambda args, row: rows.append(row))
    context = replay.replay_routes(model, data.router_routes(model, batch))
    x = torch.randn(1, 3, 8, requires_grad=True)

    def forward(x):
        weights, _, _, _ = router(x, scores_only=False)
        return weights

    def run():
        with replay_diagnostics.checked_context(actor, module, batch, context):
            if corrupt:
                router.replay_expert_indices = router.replay_expert_indices.roll(1, dims=-1)
            weights = checkpoint(forward, x, use_reentrant=True)
            (weights * torch.tensor([1.0, 2.0])).sum().backward()

    if corrupt:
        with pytest.raises(ValueError, match="diverged"):
            run()
    else:
        run()
        assert rows[0]["mismatches"] == 0
        counts = rows[0]["layers"]["blocks.0.routed_experts_router"]
        assert counts == {"entered": 2, "returned": 2, "grad_enabled": 1}
        assert torch.isfinite(router.weight.grad).all() and router.weight.grad.abs().sum() > 0
    assert not router._forward_hooks and not router._forward_pre_hooks
    assert router.replay_expert_indices is None


def test_trial_uses_real_model_shape_and_bounded_allocation(tmp_path):
    config = exercise_controls.configuration(tmp_path, tmp_path, "replay-admission64", 8)
    assert config.core.replay_diagnostics and config.core.activation_checkpointing
    assert config.core.expert_parallel_size == 2
    assert config.miles["use_rollout_routing_replay"] and config.miles["use_miles_router"]
    assert config.miles["global_batch_size"] == 64
    (task,) = launch_control_exercise.specification("image", replay_only=True)["tasks"]
    assert task["resources"]["gpuCount"] == 3
    assert task["context"] == {"priority": "urgent", "minRuntime": "1h", "autoResume": False}


@pytest.mark.parametrize("fault", [None, "missing", "mismatch", "recompute"])
def test_audit_rejects_incomplete_replay(fault):
    rows = [
        dict(
            event="replay_routes",
            rollout_id=0,
            phase=phase,
            tokens=3,
            captured_tokens=2,
            mismatches=int(fault == "mismatch"),
            layers={
                "blocks.0.routed_experts_router": dict(
                    entered=2, returned=1 if fault == "recompute" else 2, grad_enabled=1
                )
            },
        )
        for phase in ("scoring", "training")
    ]
    if fault == "missing":
        rows.pop()
    if fault:
        with pytest.raises(ValueError):
            replay_diagnostics.audit_contracts({"0": rows}, 1, 1)
    else:
        assert replay_diagnostics.audit_contracts({"0": rows}, 1, 1)["passed"]
