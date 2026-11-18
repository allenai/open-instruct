"""Regression coverage for original context required by specialized graders."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from open_instruct import ground_truth_utils
from open_instruct import test_miles_grading as shared
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import rewards


@pytest.mark.parametrize("factory", ["MaxLenVerifier", "UpToMaxLenVerifier"])
def test_token_based_verifiers_keep_full_tokens_and_reject_selected_scope(tmp_path, factory):
    path = tmp_path / "length.json"
    spec = {"factory": "open_instruct.ground_truth_utils." + factory, "config": {"max_length_verifier_max_length": 4}}
    path.write_text(json.dumps({"test": spec}))
    registry = rewards._registry(str(path))
    assert registry.grading_policies["test"].scope == "full"
    sample = shared.item("work</think>answer", target="2")
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=str(path), reward_final_answer_only=True))
    # Three actual response tokens give 0.75. Empty tokens instead give 0.5
    # (MaxLen) or 1.0 (UpToMaxLen), so both cases detect lost token inputs.
    assert asyncio.run(rewards.registered_reward(args, sample)) == 0.75
    spec["grading"] = {"objective": "answer", "scope": "response"}
    invalid = tmp_path / "bad-length.json"
    invalid.write_text(json.dumps({"test": spec}))
    with pytest.raises(ValueError, match="Token-based"):
        rewards._registry(str(invalid))


def test_symbolic_isolation_preserves_declared_token_requirements(tmp_path, monkeypatch):
    factory = type(
        "TokenAwareMath",
        (ground_truth_utils.StrictMathVerifier,),
        {"grading_objective": "process", "grading_scope": "full", "grading_uses_tokens": True},
    )
    monkeypatch.setattr(ground_truth_utils, "TokenAwareMath", factory, raising=False)
    path = tmp_path / "token-math.json"
    path.write_text(json.dumps({"test": {"factory": "open_instruct.ground_truth_utils.TokenAwareMath"}}))
    registry = rewards._registry(str(path))
    assert isinstance(registry["test"], rewards._IsolatedVerifier)
    assert registry.grading_policies["test"].scope == "full"
    invalid = tmp_path / "bad-token-math.json"
    invalid.write_text(
        json.dumps(
            {
                "test": {
                    "factory": "open_instruct.ground_truth_utils.TokenAwareMath",
                    "grading": {"objective": "answer", "scope": "response"},
                }
            }
        )
    )
    with pytest.raises(ValueError, match="Token-based"):
        rewards._registry(str(invalid))
