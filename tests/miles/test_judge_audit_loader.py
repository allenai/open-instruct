"""Retained router replay arrays remain readable with restricted deserialization."""

import pickle

import numpy as np
import pytest
import torch
from scripts.miles import audit_judge_exercise


class UnexpectedPayload:
    pass


def test_loads_int32_replay_array_without_global_allowlist_leak(tmp_path):
    routes = np.arange(24, dtype=np.int32).reshape(3, 2, 4)
    path = tmp_path / "rollout.pt"
    torch.save({"samples": [{"rollout_routed_experts": routes, "tokens": [1, 2, 3, 4]}]}, path)
    before = torch.serialization.get_safe_globals().copy()
    result = audit_judge_exercise.load_rollout(path)
    np.testing.assert_array_equal(result["samples"][0]["rollout_routed_experts"], routes)
    assert torch.serialization.get_safe_globals() == before


def test_unknown_pickle_types_still_rejected(tmp_path):
    path = tmp_path / "unexpected.pt"
    torch.save({"samples": [UnexpectedPayload()]}, path)
    before = torch.serialization.get_safe_globals().copy()
    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        audit_judge_exercise.load_rollout(path)
    assert torch.serialization.get_safe_globals() == before
