"""The publication boundary must survive a producer that ignores its first cancel."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from open_instruct.miles import async_rollout


class _Source:
    def __init__(self):
        self.requeued = []

    def requeue_pending_groups(self, identities):
        self.requeued.append(list(identities))


def _producer(worker_coro):
    fn = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
    fn.args = SimpleNamespace(rollout_health_check_timeout=0.2, rollout_submission_granularity="group")
    fn.state = SimpleNamespace(aborted=False, reset=lambda: None)
    fn.data_source = _Source()
    fn._producer_resumed = asyncio.Event()
    fn._producing_groups = {7: object()}
    fn._interrupted_groups = []
    fn._active_tasks = set()
    fn._publication_paused = False
    fn._worker = asyncio.create_task(worker_coro)
    return fn


@pytest.fixture
def engines(monkeypatch):
    calls = []

    async def urls(args):
        return ["http://engine-a", "http://engine-b"]

    async def post(url, payload, max_retries=1):
        calls.append((url, payload))

    monkeypatch.setattr(async_rollout, "get_worker_urls", urls)
    monkeypatch.setattr(async_rollout, "post", post)
    monkeypatch.setattr(async_rollout, "_PUBLICATION_JOIN_RETRY_SECONDS", 2.0)
    return calls


def test_prompt_join_aborts_engines_once(engines):
    async def scenario():
        async def worker():
            await asyncio.Event().wait()

        fn = _producer(worker())
        identities = await fn.prepare_publication()
        assert identities == [7]
        assert fn._worker is None

    asyncio.run(scenario())
    assert sorted(url for url, _ in engines) == ["http://engine-a/abort_request", "http://engine-b/abort_request"]
    assert all(payload == {"abort_all": True} for _, payload in engines)


def test_stubborn_worker_is_released_by_the_engine_abort(engines):
    async def scenario():
        released = asyncio.Event()

        async def worker():
            # Swallow the first cancel, the way a task blocked behind an engine
            # response effectively does, and only finish once the abort lands.
            with suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
            await released.wait()

        fn = _producer(worker())
        original = async_rollout.post

        async def post_and_release(url, payload, max_retries=1):
            await original(url, payload, max_retries=max_retries)
            released.set()

        async_rollout.post = post_and_release
        try:
            identities = await fn.prepare_publication()
        finally:
            async_rollout.post = original
        assert identities == [7]
        assert fn._worker is None
        assert fn.data_source.requeued == [[7]]

    asyncio.run(scenario())
    # The abort ran exactly once per engine: during the retry, not again afterwards.
    assert len(engines) == 2


def test_worker_error_still_surfaces(engines):
    async def scenario():
        async def worker():
            raise RuntimeError("producer broke")

        fn = _producer(worker())
        await asyncio.sleep(0)  # let the worker run and fail before the boundary
        with pytest.raises(RuntimeError, match="producer broke"):
            await fn.prepare_publication()

    asyncio.run(scenario())
