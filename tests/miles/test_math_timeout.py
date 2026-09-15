"""Bounded symbolic grading recovers after a timeout without hiding other errors."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from open_instruct.miles import rewards, rollout_metrics

WORKER = """
import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request.get('prediction') == 'hang':
        time.sleep(60)
    print('MILES_VERIFIER_RESULT ' + json.dumps({'schema_version': 1, 'result': {'score': 1.0, 'cost': 0.0}}), flush=True)
"""


def test_timed_out_process_is_reaped_and_next_request_succeeds(monkeypatch):
    async def run():
        children = []

        async def start():
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-c", WORKER, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
            )
            children.append(child)
            return child

        monkeypatch.setattr(rewards._MathProcessPool, "_start", staticmethod(start))
        pool = rewards._MathProcessPool(workers=1, timeout=0.3)
        try:
            with pytest.raises(rewards.MathVerifierTimeout):
                await pool.score({"prediction": "hang"})
            assert (await pool.score({"prediction": "valid"})).score == 1.0
            assert len(children) == 2 and children[0].returncode is not None
        finally:
            await pool.close()
        assert all(child.returncode is not None for child in children)

    asyncio.run(run())


def test_timeout_zero_is_diagnostic_and_pool_can_continue(monkeypatch):
    calls = 0

    async def score(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise rewards.MathVerifierTimeout("Math verifier exceeded 45.0 seconds")
        return SimpleNamespace(score=1.0, cost=0.0)

    monkeypatch.setattr(rewards, "isolated_verifier_call", score)
    monkeypatch.delenv("OI_MILES_MATH_TIMEOUT_POLICY", raising=False)

    async def run():
        verifier = rewards._IsolatedVerifier({})
        timeout = await verifier.async_call([], "long answer", "2")
        valid = await verifier.async_call([], "2", "2")
        assert timeout.score == 0 and timeout.diagnostics["status"] == "timeout"
        assert valid.score == 1 and valid.diagnostics["status"] == "ok"
        metrics = rollout_metrics.math_verifier_metrics(
            [SimpleNamespace(metadata={"verifier_diagnostics": {"math": r.diagnostics}}) for r in (timeout, valid)]
        )
        assert metrics["rollout/math_verifier/timeouts"] == 1
        assert metrics["rollout/math_verifier/timeout_fraction"] == 0.5

    asyncio.run(run())


@pytest.mark.parametrize("error", [RuntimeError("worker died"), ValueError("bad schema"), asyncio.CancelledError()])
def test_other_errors_and_cancellation_are_not_zeroed(monkeypatch, error):
    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(rewards, "isolated_verifier_call", fail)
    with pytest.raises(type(error)):
        asyncio.run(rewards._IsolatedVerifier({}).async_call([], "", ""))


def test_strict_policy_and_invalid_policy_fail(monkeypatch):
    async def fail(*args, **kwargs):
        raise rewards.MathVerifierTimeout("timed out")

    monkeypatch.setattr(rewards, "isolated_verifier_call", fail)
    monkeypatch.setenv("OI_MILES_MATH_TIMEOUT_POLICY", "raise")
    with pytest.raises(rewards.MathVerifierTimeout):
        asyncio.run(rewards._IsolatedVerifier({}).async_call([], "", ""))
    monkeypatch.setenv("OI_MILES_MATH_TIMEOUT_POLICY", "typo")
    with pytest.raises(ValueError, match="must be zero or raise"):
        asyncio.run(rewards._IsolatedVerifier({}).async_call([], "", ""))
