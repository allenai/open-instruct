"""Failure-sensitive replay checks against the actual Core router."""

from types import SimpleNamespace

import pytest
import torch
from miles.backends.core_utils import data, replay_diagnostics
from olmo_core.nn.moe.v2 import replay
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from torch import nn
from torch.utils.checkpoint import checkpoint


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("detailed", [False, True])
def test_observes_routes_in_forward_and_recomputation(monkeypatch, corrupt, device, detailed):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for CPU-route/GPU-router regression")
    router = MoERouterConfigV2(d_model=8, num_experts=4, top_k=2).build()
    nn.init.normal_(router.weight, std=0.1)
    model = nn.Module()
    model.blocks = nn.ModuleDict({"0": nn.Module()})
    model.blocks["0"].routed_experts_router = router
    model.to(device)
    module = SimpleNamespace(model=model)
    actor = SimpleNamespace(
        args=SimpleNamespace(olmo_core=SimpleNamespace(router_diagnostics=detailed)),
        clock=SimpleNamespace(next_rollout_id=0),
    )
    batch = {
        "tokens": torch.tensor([[1, 2, 3]], device=device),
        "rollout_routed_experts": [torch.tensor([[[1, 3]], [[2, 3]]])],
    }
    rows = []
    monkeypatch.setattr(replay_diagnostics.contract, "record", lambda args, row: rows.append(row))
    context = replay.replay_routes(model, data.router_routes(model, batch))
    x = torch.randn(1, 3, 8, device=device, requires_grad=True)

    def forward(x):
        weights, _, _, _ = router(x, scores_only=False)
        return weights

    def run():
        with replay_diagnostics.checked_context(actor, module, batch, context):
            if corrupt:
                router.replay_expert_indices = router.replay_expert_indices.roll(1, dims=-1)
            weights = checkpoint(forward, x, use_reentrant=True)
            (weights * torch.tensor([1.0, 2.0], device=device)).sum().backward()

    if corrupt:
        with pytest.raises(ValueError, match="diverged"):
            run()
    else:
        run()
        assert rows[0]["mismatches"] == 0
        counts = rows[0]["layers"]["blocks.0.routed_experts_router"]
        assert counts == {"entered": 2, "returned": 2, "grad_enabled": 1}
        observations = [row for row in rows if row["event"] == "router_behavior"]
        assert len(observations) == (2 if detailed else 0)
        if detailed:
            assert [row["recomputation"] for row in observations] == [False, True]
            assert all(row["tokens"] == 2 for row in observations)
        assert torch.isfinite(router.weight.grad).all() and router.weight.grad.abs().sum() > 0
    assert not router._forward_hooks and not router._forward_pre_hooks
    assert router.replay_expert_indices is None


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
