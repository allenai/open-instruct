"""Performance penalties preserve non-negative code rewards in MILES."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from open_instruct.miles.rewards import code_rewards


@pytest.mark.parametrize("stdio", [False, True])
@pytest.mark.parametrize(
    "results,runtimes,penalty,threshold,expected",
    [
        ([1], [1.5], True, 0.0, 0.0),
        ([1, 1], [0.5, 2.0], True, 0.0, 0.25),
        ([1, 1], [0.1, 0.2], True, 0.0, 0.85),
        ([1], [1.0], True, 0.0, 0.0),
        ([1, 0], [0.5, 2.0], True, 0.0, 0.25),
        ([1, 0], [0.5, 2.0], True, 0.75, 0.0),
        ([1], [1.5], False, 0.0, 1.0),
    ],
)
def test_performance_penalty(monkeypatch, stdio, results, runtimes, penalty, threshold, expected):
    args = SimpleNamespace(
        code_api_url="http://unused.invalid/test_program",
        code_max_execution_time=1.0,
        code_pass_rate_reward_threshold=threshold,
        code_apply_perf_penalty=penalty,
    )
    monkeypatch.setattr(
        code_rewards.asyncio, "to_thread", AsyncMock(return_value={"results": results, "runtimes": runtimes})
    )
    score, diagnostics = asyncio.run(
        code_rewards.execute(args, "```python\ndef f():\n    return 1\n```", ["assert f() == 1"], stdio=stdio)
    )
    assert score == pytest.approx(expected)
    assert diagnostics["status"] == "ok"
    assert diagnostics["stdio"] is stdio
