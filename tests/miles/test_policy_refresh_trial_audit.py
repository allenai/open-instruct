"""Require mixed-policy training, returned replay routes and real resume evidence."""

import json

import numpy as np
import pytest
import torch
from scripts.miles import audit_policy_refresh_trial as audit


def fixture(root):
    (root / "rollouts").mkdir()
    (root / "metrics").mkdir()
    contracts = []
    for update in range(5):
        spans = (
            [dict(version=update, start=0, end=3)]
            if update == 0
            else [dict(version=update - 1, start=0, end=1), dict(version=update, start=1, end=3)]
        )
        provenance = dict(spans=spans, replay_version=update)
        samples = [
            dict(
                tokens=[1, 2, 3, 4],
                response_length=3,
                rollout_log_probs=[-0.5, -0.8, -0.2],
                weight_versions=[
                    [dict(version=str(s["version"]), abs_start=1 + s["start"], abs_end=1 + s["end"]) for s in spans]
                ],
                group_index=update * 4 + i // 4,
                train_metadata=dict(policy_refresh=provenance),
                metadata=dict(policy_refresh=provenance),
                loss_mask=[1, 1, 1],
                rollout_routed_experts=np.zeros((3, 1, 1), dtype=np.int32),
                reward=float(i % 2),
            )
            for i in range(16)
        ]
        torch.save(dict(samples=samples), root / f"rollouts/{update}.pt")
        contracts.append(
            dict(
                event="optimizer",
                step=update + 1,
                optimizer_skipped=False,
                local_behavior_versions=[s["version"] for s in spans],
            )
        )
        for phase in ("scoring", "training"):
            contracts.append(
                dict(
                    event="replay_routes",
                    rollout_id=update,
                    phase=phase,
                    samples=8,
                    tokens=32,
                    captured_tokens=24,
                    synthetic_tail_tokens=8,
                    mismatches=0,
                    layers={"router": dict(entered=2, returned=2, grad_enabled=int(phase == "training"))},
                )
            )
        path = root / f"metrics/core/rollout_{update:07d}"
        path.mkdir(parents=True)
        (path / "complete.json").write_text(json.dumps(dict(clock=dict(completed_steps=update + 1))))
    for rank in range(2):
        (root / f"metrics/training_contract_rank{rank}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in contracts)
        )
    (root / "metrics/driver_timing.jsonl").write_text(json.dumps(dict(passed=True)) + "\n")
    (root / "metrics/publication.jsonl").write_text(json.dumps(dict(version=5)) + "\n")
    (root / "initial-result.json").write_text(json.dumps(dict(completed_rollout_ids=[0, 1, 2, 3], wall_seconds=5)))
    (root / "resume-result.json").write_text(json.dumps(dict(completed_rollout_ids=[4], wall_seconds=2)))


def test_successful_mixed_training_and_resume(tmp_path):
    fixture(tmp_path)
    result = audit.audit(tmp_path)
    assert result["consumed_mixed_responses"] == 64
    assert result["groups_with_policy_advantage"] == 20
    assert result["replay"]["passed"]


@pytest.mark.parametrize("fault", ["stale", "no_routes", "masked_prefix", "no_resume", "bad_replay"])
def test_false_success_is_rejected(tmp_path, fault):
    fixture(tmp_path)
    path = tmp_path / "rollouts/4.pt"
    payload = torch.load(path, weights_only=False)
    if fault == "stale":
        sample = payload["samples"][0]
        sample["weight_versions"][0] = "0"
        sample["train_metadata"]["policy_refresh"]["spans"][0]["version"] = 0
    elif fault == "no_routes":
        payload["samples"][0]["rollout_routed_experts"] = np.zeros((2, 1, 1), dtype=np.int32)
    elif fault == "masked_prefix":
        payload["samples"][0]["loss_mask"][0] = 0
    elif fault == "no_resume":
        (tmp_path / "resume-result.json").write_text(json.dumps(dict(completed_rollout_ids=[0], wall_seconds=2)))
    else:
        path2 = tmp_path / "metrics/training_contract_rank0.jsonl"
        rows = audit.records(path2)
        rows[1]["mismatches"] = 1
        path2.write_text("".join(json.dumps(r) + "\n" for r in rows))
    torch.save(payload, path)
    with pytest.raises(ValueError):
        audit.audit(tmp_path)


def test_barrier_control_keeps_single_version_contract(tmp_path):
    fixture(tmp_path)
    for update in range(5):
        path = tmp_path / f"rollouts/{update}.pt"
        payload = torch.load(path, weights_only=False)
        for sample in payload["samples"]:
            sample["weight_versions"] = [str(update)]
            sample.pop("train_metadata")
            sample["metadata"] = {}
        torch.save(payload, path)
    result = audit.audit(tmp_path, mode="barrier")
    assert result["mode"] == "barrier"
    assert result["consumed_mixed_responses"] == 0
    assert result["replay"]["passed"]
