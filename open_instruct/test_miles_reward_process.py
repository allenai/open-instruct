"""Symbolic grading runs on isolated main threads and reaps failed workers."""

import asyncio
import dataclasses
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from open_instruct.miles import rewards


@dataclasses.dataclass
class _Config:
    pass


@dataclasses.dataclass
class _Result:
    score: float
    cost: float = 0.0
    reasoning: str | None = None


class ProcessFixtureVerifier:
    """Trusted test worker exercising logs, exceptions, cancellation and reuse."""

    @classmethod
    def get_config_class(cls):
        return _Config

    def __init__(self, verifier_config):
        self.calls = 0

    def __call__(self, tokenized_prediction, prediction, label, query=None, rollout_state=None):
        self.calls += 1
        print("Import/library-style stdout must not corrupt the protocol")
        os.write(1, b"Native stdout must not corrupt the protocol either\n")
        if prediction == "raise":
            raise ValueError("deliberate verifier failure")
        if prediction == "exit":
            os._exit(7)
        if prediction == "sleep":
            time.sleep(30)
        return _Result(float(label), reasoning=f"{os.getpid()}:{self.calls}")


def _request(prediction="ok"):
    return dict(
        factory_spec={"factory": "open_instruct.test_miles_reward_process.ProcessFixtureVerifier"},
        tokenized_prediction=[],
        prediction=prediction,
        label=1.0,
    )


def test_symbolic_fraction_equivalence_through_async_registry(tmp_path):
    registry = tmp_path / "verifiers.json"
    registry.write_text(
        json.dumps(
            {
                "math": {"factory": "open_instruct.ground_truth_utils.MathVerifier"},
                "strict": {"factory": "open_instruct.ground_truth_utils.StrictMathVerifier"},
            }
        )
    )
    args = SimpleNamespace(olmo_core=SimpleNamespace(reward_config=str(registry)))
    samples = [
        SimpleNamespace(
            tokens=[],
            response_length=0,
            response=response,
            prompt="Compute one half",
            metadata={"verifiers": [{"name": name, "target": r"\frac{1}{2}"}]},
        )
        for name in ("math", "strict")
        for response in (r"$\boxed{\frac{2}{4}}$", r"$\boxed{\frac{3}{4}}$")
    ]
    children_before = set(rewards._CHILDREN)
    assert asyncio.run(rewards.registered_reward(args, samples)) == [1.0, 0.0, 1.0, 0.0]
    assert children_before == rewards._CHILDREN


def test_process_reuse_bounded_concurrency_errors_and_stdout():
    async def run():
        pool = rewards._MathProcessPool(workers=2, timeout=20)
        try:
            results = await asyncio.gather(*(pool.score(_request()) for _ in range(6)))
            pids = {result.reasoning.split(":")[0] for result in results}
            assert len(pids) <= 2
            assert any(int(result.reasoning.split(":")[1]) > 1 for result in results)
            with pytest.raises(RuntimeError, match="deliberate verifier failure"):
                await pool.score(_request("raise"))
            assert (await pool.score(_request())).score == 1.0
            with pytest.raises(RuntimeError, match="exited without a result"):
                await pool.score(_request("exit"))
            assert (await pool.score(_request())).score == 1.0
        finally:
            await pool.close()
        assert not rewards._CHILDREN

    asyncio.run(run())


def test_timeout_and_cancellation_reap_worker_and_allow_next_request():
    async def run():
        pool = rewards._MathProcessPool(workers=1, timeout=20)
        try:
            first = await pool.score(_request())
            first_pid = int(first.reasoning.split(":")[0])
            pool.timeout = 0.1
            with pytest.raises(TimeoutError, match="exceeded"):
                await pool.score(_request("sleep"))
            pool.timeout = 20
            second = await pool.score(_request())
            assert int(second.reasoning.split(":")[0]) != first_pid
            request = asyncio.create_task(pool.score(_request("sleep")))
            await asyncio.sleep(0.05)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            third = await pool.score(_request())
            assert third.reasoning.split(":")[0] != second.reasoning.split(":")[0]
        finally:
            await pool.close()
        assert not rewards._CHILDREN

    asyncio.run(run())


def test_subprocess_start_from_background_event_loop():
    async def run():
        request = _request()
        return await rewards.isolated_verifier_call(**request)

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(asyncio.run, run()).result(timeout=20)
    assert result.score == 1.0
    assert not rewards._CHILDREN
