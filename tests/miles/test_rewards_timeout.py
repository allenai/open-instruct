"""A verifier timeout scores the sample 0 instead of failing the reward call."""

import asyncio
from types import SimpleNamespace

import pytest

from open_instruct.miles import rewards


class _StallingVerifier:
    async def async_call(self, *args, **kwargs):
        raise TimeoutError("Math verifier exceeded 45.0 seconds")


class _FineVerifier:
    async def async_call(self, *args, **kwargs):
        return SimpleNamespace(score=1.0, cost=0.0)


def _sample():
    return SimpleNamespace(
        metadata={"verifiers": [{"name": "math", "target": "1", "weight": 2.0}, {"name": "fine", "target": "1"}]},
        tokens=[1, 2, 3],
        response="\\boxed{",
        response_length=2,
        prompt="q",
    )


def test_verifier_timeout_scores_zero_and_is_recorded(monkeypatch):
    monkeypatch.setattr(rewards, "_registry", lambda path: {"math": _StallingVerifier(), "fine": _FineVerifier()})
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: False)
    sample = _sample()
    total = asyncio.run(rewards.score(sample, "unused.json"))
    assert total == 1.0  # 2.0 * 0 (timed out) + 1.0 * 1
    components = {c["name"]: c for c in sample.metadata["reward_components"]}
    assert components["math"]["score"] == 0.0
    assert sample.metadata["verifier_diagnostics"]["math"]["timed_out"] is True


def test_other_verifier_failures_still_raise(monkeypatch):
    class _Broken:
        async def async_call(self, *args, **kwargs):
            raise RuntimeError("Math verifier failed: bad worker")

    monkeypatch.setattr(rewards, "_registry", lambda path: {"math": _Broken()})
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: False)
    with pytest.raises(RuntimeError, match="bad worker"):
        asyncio.run(rewards.score(_sample(), "unused.json"))
