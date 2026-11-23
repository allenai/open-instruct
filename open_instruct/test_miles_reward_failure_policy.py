"""Callers choose how reported infrastructure failures affect their rewards."""

import asyncio
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.distillation import opd_rewards
from open_instruct.miles.rewards import rewards


def sample(response="failed", group_index=None):
    return SimpleNamespace(
        prompt="question",
        response=response,
        tokens=[],
        response_length=0,
        group_index=group_index,
        metadata={"verifiers": [{"name": "test", "target": "answer"}]},
    )


@pytest.fixture(params=["service_error", "judge_error"])
def scoring(request, monkeypatch, tmp_path):
    status = request.param

    async def verify(tokens, response, target, **kwargs):
        if response == "invalid_target":
            raise ValueError("invalid target")
        if response == "exception":
            raise RuntimeError("unexpected verifier exception")
        return SimpleNamespace(
            score=1.0,
            cost=0.0,
            diagnostics={
                "status": "completed" if response == "gold" or target == "healthy" else status,
                "error": "unavailable",
            },
        )

    async def judge(args, scored, **kwargs):
        result = await verify([], scored.response, kwargs["target"])
        scored.metadata.setdefault("verifier_diagnostics", {})["test"] = result.diagnostics
        return result.score

    monkeypatch.setattr(rewards, "_registry", lambda path: {"test": SimpleNamespace(async_call=verify)})
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: status == "judge_error")
    monkeypatch.setattr(rewards.general_judge, "general_judge_score", judge)
    return SimpleNamespace(olmo_core=CoreConfig(reward_config="unused"), save=str(tmp_path))


@pytest.mark.parametrize("batch", [False, True])
def test_explicit_zero_preserves_failure_diagnostics_and_success(scoring, batch):
    failed = sample()
    inputs = [failed, sample("gold")] if batch else failed
    result = asyncio.run(rewards.registered_reward(scoring, inputs, failure_policy="zero"))
    assert result == ([0.0, 1.0] if batch else 0.0)
    diagnostics = failed.metadata["verifier_diagnostics"]["test"]
    assert diagnostics["status"] in {"service_error", "judge_error"}
    assert diagnostics["scoring_failure_policy"] == "zero"
    assert failed.metadata["reward_components"][0]["score"] == 0.0
    assert "reward_omission" not in failed.metadata


@pytest.mark.parametrize("options", [{}, {"failure_policy": "raise"}])
def test_default_and_explicit_raise_reject_reported_failures(scoring, options):
    with pytest.raises(RuntimeError, match="failed to grade"):
        asyncio.run(rewards.registered_reward(scoring, sample(), **options))


def test_zero_only_replaces_failed_component_in_weighted_reward(scoring):
    scored = sample()
    scored.metadata["verifiers"] = [
        {"name": "test", "target": "healthy", "weight": 2.0},
        {"name": "test", "target": "answer", "weight": 3.0},
    ]
    assert asyncio.run(rewards.registered_reward(scoring, scored, failure_policy="zero")) == 2.0
    assert [c["score"] for c in scored.metadata["reward_components"]] == [1.0, 0.0]


def test_opd_and_regular_evaluation_request_zero_but_training_omits(scoring):
    assert asyncio.run(opd_rewards.score(sample(), "unused", scoring)) == 0.0
    assert asyncio.run(rewards.rollout_reward(scoring, sample())) == 0.0
    failed = sample(group_index=7)
    assert asyncio.run(rewards.rollout_reward(scoring, failed)) is None
    assert failed.metadata["reward_omission"]["verifier"] == "test"


@pytest.mark.parametrize("response,error", [("invalid_target", ValueError), ("exception", RuntimeError)])
def test_zero_does_not_hide_invalid_targets_or_unexpected_exceptions(scoring, response, error):
    with pytest.raises(error):
        asyncio.run(rewards.registered_reward(scoring, sample(response), failure_policy="zero"))


def test_invalid_policy_is_rejected_even_for_an_empty_batch(scoring):
    with pytest.raises(ValueError, match="failure_policy"):
        asyncio.run(rewards.registered_reward(scoring, [], failure_policy="ignore"))
