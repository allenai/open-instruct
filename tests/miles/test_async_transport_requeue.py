"""A lost HTTP transport on one request requeues its prompt group instead of failing the run."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter

from open_instruct.miles import async_rollout


async def _router_transport_failure():
    """Exercise the pinned router's real HTTP error translation, not a fake 503."""
    router = MilesRouter(
        MilesRouterConfig(
            host="127.0.0.1",
            port=8000,
            max_connections=4,
            timeout=10,
            health_check_interval=1,
            health_check_failure_threshold=3,
        )
    )
    await router.client.aclose()

    async def disconnected(request):
        raise httpx.ReadError("injected engine disconnect", request=request)

    router.client = httpx.AsyncClient(transport=httpx.MockTransport(disconnected))
    router.worker_request_counts["http://engine"] = 0
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=router.app), base_url="http://router"
        ) as client:
            response = await client.post("/generate", json={"rid": "test-request"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Rollout worker unavailable"}
        assert router.worker_request_counts["http://engine"] == 0
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            return error
        pytest.fail("Expected a router transport failure")
    finally:
        await router.close()


class _Source:
    def __init__(self):
        self.requeued = []

    def requeue_pending_groups(self, identities):
        self.requeued.append(list(identities))
        return len(identities)


def _producer():
    fn = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
    fn.data_source = _Source()
    fn._producing_groups = {7: object(), 8: object()}
    fn._task_groups = {}
    fn._transport_requeues = 0
    fn._consecutive_transport_failures = 0
    return fn


def test_transport_error_requeues_the_group_and_drops_it_from_production():
    fn = _producer()
    assert fn._requeue_transport_failure(7, httpx.ReadError("connection reset")) is True
    assert fn.data_source.requeued == [[7]]
    assert 7 not in fn._producing_groups and 8 in fn._producing_groups
    assert (fn._transport_requeues, fn._consecutive_transport_failures) == (1, 1)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Policy refresh request did not finish; refusing a partial training sample"),
        httpx.HTTPStatusError("500", request=httpx.Request("POST", "http://r/generate"), response=httpx.Response(500)),
        TimeoutError("core.refresh_request_timeout"),
    ],
)
def test_other_failures_still_propagate(error):
    fn = _producer()
    assert fn._requeue_transport_failure(7, error) is False
    assert fn.data_source.requeued == [] and 7 in fn._producing_groups
    assert fn._requeue_transport_failure(None, httpx.ReadError("no group")) is False


def test_consecutive_transport_failures_are_bounded():
    fn = _producer()
    fn._producing_groups = {i: object() for i in range(20)}
    for i in range(async_rollout.TRANSPORT_FAILURE_BUDGET):
        assert fn._requeue_transport_failure(i, httpx.ConnectError("refused")) is True
    with pytest.raises(RuntimeError, match="consecutive async requests lost their HTTP transport"):
        fn._requeue_transport_failure(async_rollout.TRANSPORT_FAILURE_BUDGET, httpx.ConnectError("refused"))
    # A completed group in between resets the streak, as the worker loop does.
    fn._consecutive_transport_failures = 0
    assert fn._requeue_transport_failure(15, httpx.RemoteProtocolError("closed")) is True


@pytest.mark.parametrize("through_router", [False, True])
def test_worker_loop_requeues_transport_failures_and_delivers_the_rest(monkeypatch, through_router):
    """Drive the real worker loop: one group loses its transport, the next two complete."""

    async def observe(*a, **k):
        await asyncio.Event().wait()

    monkeypatch.setattr(async_rollout.pipeline_observer, "observe", observe)
    monkeypatch.setattr(async_rollout.pipeline_observer, "observe_engines", observe)
    monkeypatch.setattr(async_rollout.pipeline_observer, "completion_counts", lambda items: {})

    async def scenario():
        fn = _producer()
        fn._producing_groups = {}
        fn._producer_resumed = asyncio.Event()
        fn._producer_resumed.set()
        fn._producer_idle = asyncio.Event()
        fn._stopping = False
        fn._publication_paused = False
        fn._active_tasks = set()
        fn._ready_completion_counts = {}
        fn._shutdown_unqueued_counts = {}
        fn.args = SimpleNamespace(_olmo_rollout_generation_interrupted=False, _olmo_rollout_pool_exhausted=False)
        delivered = []
        groups = iter(
            [[SimpleNamespace(group_index=1)], [SimpleNamespace(group_index=2)], [SimpleNamespace(group_index=3)]]
        )
        fn.data_source.get_samples = lambda n: [next(groups)]
        submitted = []

        class Scheduler:
            def has_capacity(self, *, pending_groups, group_budget):
                return len(submitted) < 3 and pending_groups < 1

            def on_submit(self, gs):
                submitted.extend(gs)

            async def wait_for_progress(self, pendings):
                return await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)

        fn._scheduler = Scheduler()
        fn._max_in_flight_groups = lambda: 1

        async def generate(group):
            await asyncio.sleep(0)
            if group[0].group_index == 1:
                if through_router:
                    raise await _router_transport_failure()
                raise httpx.ReadError("connection reset by router")
            return SimpleNamespace(prompt_group=group, group=[])

        fn._generate_group = generate

        async def put_or_stop(item):
            delivered.append(item.prompt_group[0].group_index)
            if len(delivered) == 2:
                fn._stopping = True
            return True

        fn._put_or_stop = put_or_stop

        await asyncio.wait_for(fn._worker_loop(), timeout=5)
        assert fn.data_source.requeued == [[1]]
        assert delivered == [2, 3]
        assert fn._transport_requeues == 1 and fn._consecutive_transport_failures == 0
        assert fn._task_groups == {}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,detail,path",
    [
        (500, "Rollout worker unavailable", "/generate"),
        (503, "Unknown engine failure", "/generate"),
        (503, "No healthy workers", "/generate"),
        (400, "Invalid request", "/generate"),
        (503, "Rollout worker unavailable", "/reward"),
    ],
)
def test_unrecognized_http_failures_are_not_reclassified(status, detail, path):
    request = httpx.Request("POST", "http://router" + path)
    response = httpx.Response(status, request=request, json={"detail": detail})
    error = httpx.HTTPStatusError("unavailable", request=request, response=response)
    fn = _producer()
    assert fn._requeue_transport_failure(7, error) is False
    assert fn.data_source.requeued == []


def test_router_transport_failure_uses_the_same_bounded_group_retry_budget():
    error = asyncio.run(_router_transport_failure())
    fn = _producer()
    fn._producing_groups = {i: object() for i in range(20)}
    for i in range(async_rollout.TRANSPORT_FAILURE_BUDGET):
        assert fn._requeue_transport_failure(i, error) is True
    with pytest.raises(RuntimeError, match="consecutive async requests lost their HTTP transport"):
        fn._requeue_transport_failure(async_rollout.TRANSPORT_FAILURE_BUDGET, error)
