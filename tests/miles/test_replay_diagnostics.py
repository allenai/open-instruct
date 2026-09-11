"""Failure-sensitive replay checks against the actual Core router."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from olmo_core.nn.moe.v2 import replay
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from scripts.miles import exercise_controls, launch_control_exercise, launch_control_reaudit
from torch import nn
from torch.utils.checkpoint import checkpoint

from open_instruct.miles import data, replay_diagnostics


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_observes_routes_in_forward_and_recomputation(monkeypatch, corrupt, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for CPU-route/GPU-router regression")
    router = MoERouterConfigV2(d_model=8, num_experts=4, top_k=2).build()
    nn.init.normal_(router.weight, std=0.1)
    model = nn.Module()
    model.blocks = nn.ModuleDict({"0": nn.Module()})
    model.blocks["0"].routed_experts_router = router
    model.to(device)
    module = SimpleNamespace(model=model)
    actor = SimpleNamespace(args=None, clock=SimpleNamespace(next_rollout_id=0))
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


@pytest.mark.parametrize("fault", [None, "missing", "extra", "wrong_version"])
def test_replay_publication_verification_roundtrips(tmp_path, fault):
    config = exercise_controls.configuration(tmp_path, tmp_path, "replay-admission64", 8)
    rows = [{"version": 0, "repeated_version": False}]
    for version in range(1, 9):
        rows.extend([{"version": version, "repeated_version": False}, {"version": version, "repeated_version": True}])
    if fault == "missing":
        rows.pop()
    elif fault == "extra":
        rows.append(rows[-1])
    elif fault == "wrong_version":
        rows[-1] = {"version": 7, "repeated_version": True}
    if fault:
        with pytest.raises(ValueError, match="publication"):
            exercise_controls.validate_publications(config, rows, 8)
    else:
        exercise_controls.validate_publications(config, rows, 8)


def test_replay_reaudit_is_cpu_only_on_saturn():
    (task,) = launch_control_reaudit.specification("image", "01M2931SC4WNX57GB3AXWKV03W", replay_only=True)["tasks"]
    assert task["constraints"]["cluster"] == ["ai2/saturn"]
    assert task["resources"].get("gpuCount", 0) == 0
    assert "for arm in replay-admission64; do" in task["arguments"][0]


def test_reaudit_shell_forwards_replay_flag(tmp_path):
    executable = tmp_path / "python"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    script = Path(__file__).resolve().parents[2] / "scripts/train/debug/miles_control_reaudit.sh"
    result = subprocess.run(
        ["bash", str(script), "image", "retained", "--replay-only"],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "-m",
        "scripts.miles.launch_control_reaudit",
        "image",
        "retained",
        "--replay-only",
    ]
