"""Audit must reject inert learning, malformed replay and mixed policy groups."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts.miles import audit_engine_drain_learning as audit


def fixture(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    saved = checkpoints / "core/rollout_0000001"
    saved.mkdir(parents=True)
    (saved / "complete.json").write_text(json.dumps({"clock": {"completed_steps": 2}, "hf_config": {}}))
    (tmp_path / "resolved-plan.json").write_text(
        json.dumps(
            {
                "core": {"max_policy_lag": 2, "publication_mode": "barrier"},
                "miles": {
                    "actor_num_nodes": 1,
                    "actor_num_gpus_per_node": 2,
                    "num_rollout": 2,
                    "global_batch_size": 2,
                    "n_samples_per_prompt": 2,
                    "hf_checkpoint": "initial",
                },
            }
        )
    )
    row = {"event": "optimizer", "step": 2, "optimizer_skipped": False, "local_behavior_versions": [1]}
    for rank in range(2):
        (checkpoints / f"training_contract_rank{rank}.jsonl").write_text(json.dumps(row) + "\n")
    (checkpoints / "driver_timing.jsonl").write_text("")
    samples = [
        {
            "weight_versions": ["1"],
            "response_length": 2,
            "rollout_log_probs": [-1, -1],
            "tokens": [1, 2, 3, 4],
            "rollout_routed_experts": np.zeros((3, 2, 1), dtype=np.int32),
            "group_index": 0,
            "reward": float(i),
        }
        for i in range(2)
    ]
    monkeypatch.setattr(audit.audit_workflow, "load_rollout", lambda path: {"samples": samples})
    masters = {"blocks.1.routed_experts_router.weight": torch.tensor([1.001], dtype=torch.float32)}
    monkeypatch.setattr(audit, "CoreCheckpointState", lambda *a, **k: masters)
    monkeypatch.setattr(audit, "Olmo3MoeConfig", lambda **kwargs: None)

    class Initial(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        audit,
        "SafeTensorState",
        lambda path: Initial({"model.layers.1.mlp.router.gate.weight": torch.tensor([1.0], dtype=torch.bfloat16)}),
    )
    return SimpleNamespace(samples=samples, masters=masters)


def test_completed_learning_and_native_master_change(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    report = audit.audit(tmp_path)
    assert report["completed_steps"] == [2]
    assert report["mixed_reward_groups"] == 1
    assert report["active_response_tokens"] == 4
    assert report["router_fp32_master_changes"]["model.layers.1.mlp.router.gate.weight"]["changed_elements"] == 1


@pytest.mark.parametrize(
    "fault", ["same_rewards", "mixed_policy", "future_policy", "bad_replay", "nan_probs", "no_update"]
)
def test_incomplete_or_misleading_evidence_fails(tmp_path, monkeypatch, fault):
    state = fixture(tmp_path, monkeypatch)
    if fault == "same_rewards":
        state.samples[1]["reward"] = 0
    elif fault == "mixed_policy":
        state.samples[1]["weight_versions"] = ["0"]
    elif fault == "future_policy":
        state.samples[1]["weight_versions"] = ["2"]
    elif fault == "bad_replay":
        state.samples[1]["rollout_routed_experts"] = np.zeros((2, 2, 1), dtype=np.int32)
    elif fault == "nan_probs":
        state.samples[1]["rollout_log_probs"] = [float("nan"), -1]
    else:
        state.masters["blocks.1.routed_experts_router.weight"].fill_(1)
    with pytest.raises(ValueError):
        audit.audit(tmp_path)


def test_import_rounding_is_not_mistaken_for_optimizer_change(tmp_path, monkeypatch):
    state = fixture(tmp_path, monkeypatch)
    state.masters["blocks.1.routed_experts_router.weight"].fill_(1.0)

    class Initial(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        audit,
        "SafeTensorState",
        lambda path: Initial({"model.layers.1.mlp.router.gate.weight": torch.tensor([1.001], dtype=torch.float32)}),
    )
    with pytest.raises(ValueError, match="No measured router master change"):
        audit.audit(tmp_path)
