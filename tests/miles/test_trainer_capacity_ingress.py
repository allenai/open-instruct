"""Retained-batch screen must retain MILES tensor/precision ingress semantics."""

import contextlib
from types import SimpleNamespace

import numpy as np
import torch
from miles.backends.training_utils import data as miles_data
from scripts.miles import profile_trainer_capacity


def test_real_ingress_converts_routes_and_behavior_scores(monkeypatch):
    raw = dict(
        tokens=[[1, 2, 3, 4]],
        loss_masks=[[1, 1]],
        total_lengths=[4],
        response_lengths=[2],
        rollout_routed_experts=[np.zeros((3, 2, 2), dtype=np.int32)],
        rollout_log_probs=[[-0.123456, -1.234567]],
    )
    args = SimpleNamespace(enable_witness=False, qkv_format="thd", true_on_policy_mode=True, bf16=True)
    state = SimpleNamespace(effective_dp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(miles_data, "get_parallel_state", lambda: state)
    monkeypatch.setattr(miles_data, "process_rollout_data", lambda *a, **kw: (raw, contextlib.nullcontext()))
    monkeypatch.setattr(miles_data, "slice_log_prob_with_cp", lambda value, *a: value)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    result, _ = profile_trainer_capacity.actor.miles_data.get_rollout_data(args, None)
    assert result["rollout_routed_experts"][0].dtype == torch.int32
    assert result["tokens"][0].dtype == torch.int64
    assert result["rollout_log_probs"][0].dtype == torch.bfloat16
    torch.testing.assert_close(
        result["rollout_log_probs"][0], torch.tensor([-0.123456, -1.234567], dtype=torch.bfloat16)
    )
