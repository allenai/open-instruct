import copy
import json

import numpy as np
import pytest
import torch
from scripts.miles import audit_refresh_samples

MODEL = {
    "model_type": "olmo3moe",
    "num_hidden_layers": 2,
    "num_experts_per_tok": 2,
    "n_routed_experts": 4,
    "dense_layers_indices": [0],
}


def sample():
    provenance = {"replay_version": 3, "spans": [dict(version=1, start=0, end=2), dict(version=3, start=2, end=3)]}
    routes = np.zeros((4, 2, 2), dtype=np.int32)
    routes[:, 1] = [1, 3]
    return {
        "index": 0,
        "group_index": 0,
        "tokens": [1, 2, 3, 4, 5],
        "response_length": 3,
        "status": "completed",
        "rollout_log_probs": [-1.0, -2.0, -3.0],
        "metadata": {"policy_refresh": copy.deepcopy(provenance)},
        "train_metadata": {"policy_refresh": copy.deepcopy(provenance)},
        "weight_versions": [[dict(version="1", abs_start=2, abs_end=4), dict(version="3", abs_start=4, abs_end=5)]],
        "rollout_routed_experts": routes,
    }


def test_mixed_prefix_preserves_oldest_version_and_ignores_dense_route_padding():
    report = audit_refresh_samples.audit_sample(sample(), MODEL)
    assert report["first_version"] == 1
    assert report["replay_version"] == 3
    assert report["historical_prefix_tokens"] == 2
    assert report["mixed"]


def test_prefix_version_cannot_be_reset_to_suffix_version_in_staleness_ledger():
    row = sample()
    row["weight_versions"][0][0]["version"] = "3"
    with pytest.raises(ValueError, match="staleness ledger"):
        audit_refresh_samples.audit_sample(row, MODEL)


@pytest.mark.parametrize("experts,message", [([1, 4], "invalid expert ID"), ([1, 1], "duplicate experts")])
def test_corrupt_routes_are_detected_even_when_shapes_match(experts, message):
    row = sample()
    row["rollout_routed_experts"][0, 1] = experts
    with pytest.raises(ValueError, match=message):
        audit_refresh_samples.audit_sample(row, MODEL)


def test_nonfinite_behavior_scores_rejected():
    row = sample()
    row["rollout_log_probs"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        audit_refresh_samples.audit_sample(row, MODEL)


def test_saved_tensor_collection_is_reconciled_against_consumed_counts(tmp_path):
    root = tmp_path / "run"
    (root / "rollouts").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    flow = root / "checkpoints/rollout_flow.jsonl"
    counts = dict(rollout_id=0, samples=1, response_tokens=3, mixed_responses=1)
    flow.write_text(json.dumps(counts) + "\n")
    torch.save(dict(rollout_id=0, samples=[sample()]), root / "rollouts/0.pt")
    report = audit_refresh_samples.audit(root, MODEL, tmp_path / "valid")
    assert report["collections"][0]["mixed_responses"] == 1
    counts["mixed_responses"] = 0
    flow.write_text(json.dumps(counts) + "\n")
    with pytest.raises(ValueError, match="consumed rollout accounting"):
        audit_refresh_samples.audit(root, MODEL, tmp_path / "invalid")
    assert not (tmp_path / "invalid/summary.json").exists()
