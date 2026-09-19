"""The publication boundary must survive a producer that ignores its first cancel."""

import asyncio
import json
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


def test_final_shutdown_accounts_for_completions_blocked_by_full_buffer(tmp_path):
    async def scenario():
        class Buffer:
            _capacity = 1

            def __init__(self):
                self._buffer = []

            async def put(self, item):
                if self._buffer:
                    await asyncio.Event().wait()
                self._buffer.append(item)

        fn = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
        fn.args = SimpleNamespace(
            save=str(tmp_path), sglang_server_concurrency=4, rollout_num_gpus=1, rollout_num_gpus_per_engine=1
        )
        fn.state = SimpleNamespace(generate_fn_semaphore=asyncio.Semaphore(4))
        fn._output = Buffer()
        fn._producer_resumed = asyncio.Event()
        fn._producer_resumed.set()
        fn._producer_idle = asyncio.Event()
        fn._stop_requested = asyncio.Event()
        fn._stopping = fn._shutdown_complete = fn._publication_paused = False
        fn._active_tasks = set()
        fn._producing_groups = {}
        fn._task_groups = {}
        fn._transport_requeues = fn._consecutive_transport_failures = 0
        fn._ready_completion_counts = {}
        fn._shutdown_unqueued_counts = dict(groups=0, samples=0, response_tokens=0)
        fn._completed_put_wait_seconds = 0.0
        fn._completed_put_started = None
        fn._max_in_flight_groups = lambda: 3
        fn._scheduler = SimpleNamespace(
            has_capacity=lambda pending_groups, group_budget: pending_groups < group_budget,
            wait_for_progress=lambda tasks: asyncio.wait(tasks),
        )

        def submit():
            identity = len(fn._producing_groups)
            prompt = [SimpleNamespace(group_index=identity)]
            fn._producing_groups[identity] = prompt

            async def generate():
                return SimpleNamespace(prompt_group=prompt, group=[SimpleNamespace(response_length=5)])

            return asyncio.create_task(generate())

        fn._submit_one_group = submit
        fn._worker = asyncio.create_task(fn._worker_loop())
        async with asyncio.timeout(2):
            while len(fn._ready_completion_counts) != 2:
                await asyncio.sleep(0.001)
            await fn.shutdown()
        assert len(fn._output._buffer) == 1
        assert fn._shutdown_unqueued_counts == dict(groups=2, samples=2, response_tokens=10)
        assert fn._completed_put_wait_seconds > 0
        assert not fn._active_tasks and not fn._ready_completion_counts
        records = [json.loads(line) for line in (tmp_path / "pipeline_lifecycle.jsonl").read_text().splitlines()]
        assert [row["event"] for row in records] == ["shutdown_start", "shutdown_complete"]
        assert records[0]["producer_ready"]["groups"] == 2
        assert records[-1]["completed_queue"] == dict(groups=1, samples=1, response_tokens=5)
        assert records[-1]["shutdown_unqueued"] == fn._shutdown_unqueued_counts

    asyncio.run(scenario())


@pytest.mark.parametrize("filtered", [True, False])
def test_completed_filter_drop_retires_prompt_ledger_once(filtered):
    async def scenario():
        acknowledged = []

        class Buffer:
            async def put(self, item):
                return not filtered

        producer = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
        group = [SimpleNamespace(group_index=7)]
        producer.args = SimpleNamespace()
        producer.data_source = SimpleNamespace(acknowledge_groups=acknowledged.extend)
        producer._output = Buffer()
        producer._stop_requested = asyncio.Event()
        producer._producing_groups = {7: group}
        producer._completed_put_wait_seconds = 0.0
        producer._completed_put_started = None
        # True means the completion was handled, whether queued or deliberately dropped.
        assert await producer._put_or_stop(SimpleNamespace(prompt_group=group)) is True
        assert acknowledged == ([group] if filtered else [])
        assert not producer._producing_groups
        assert producer._completed_put_started is None

    asyncio.run(scenario())
