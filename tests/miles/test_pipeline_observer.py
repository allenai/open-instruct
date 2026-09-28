"""Hard constraints, advisory limits, and reproducible topology comparisons."""

import asyncio
import json
from types import SimpleNamespace

import httpx
from miles.backends.core_utils.rollout import pipeline_observer

from open_instruct.miles.configuration.config import CoreConfig


def test_pipeline_observer_does_not_consume_or_reset_queue(tmp_path):
    async def exercise():
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        waiting = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0)
        buffer = [object(), object()]
        producer = SimpleNamespace(
            args=SimpleNamespace(
                sglang_server_concurrency=1,
                rollout_num_gpus=1,
                rollout_num_gpus_per_engine=1,
                save=str(tmp_path),
                olmo_core=CoreConfig(pipeline_observation_interval=0.001),
            ),
            state=SimpleNamespace(generate_fn_semaphore=semaphore),
            _output=SimpleNamespace(_delegate=SimpleNamespace(_buffer=buffer, _capacity=4)),
            _producing_groups={1: []},
            _errors=SimpleNamespace(metrics=lambda: {"errors/transport_groups": 3}),
            _active_tasks={waiting},
            _scheduler=SimpleNamespace(),
            _producer_resumed=asyncio.Event(),
        )
        observation = pipeline_observer.snapshot(producer)
        assert observation["completed_queue_groups"] == 2
        assert observation["errors/transport_groups"] == 3
        assert observation["http_active_requests"] == observation["http_waiting_requests"] == 1
        assert observation["producer_unfinished_samples"] is None
        assert len(buffer) == 2 and semaphore.locked() and not waiting.done()
        task = asyncio.create_task(pipeline_observer.observe(producer))
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        records = [json.loads(line) for line in (tmp_path / "pipeline_occupancy.jsonl").read_text().splitlines()]
        assert records and all(r["completed_queue_groups"] == 2 for r in records)

    asyncio.run(exercise())


def test_engine_metrics_preserve_labels_and_missing_values():
    result = pipeline_observer.engine_metrics(
        '# HELP ignored\nsglang:num_running_reqs{engine="a"} 7\n'
        'sglang:num_running_reqs{engine="b"} 2\nsglang:fwd_occupancy NaN\n'
        "sglang:unrelated_metric 900\n"
    )
    assert [r["value"] for r in result] == [7, 2, None]
    assert result[0]["labels"] != result[1]["labels"]
    assert not any(r["name"] == "num_queue_reqs" for r in result)


def test_engine_observer_shards_and_records_endpoint_failures(monkeypatch, tmp_path):
    client = httpx.AsyncClient

    def response(request):
        if request.url.host == "broken":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text='sglang:num_running_reqs{rank="0"} 3\n')

    monkeypatch.setattr(
        pipeline_observer.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(response), **kw)
    )

    async def exercise():
        producer = SimpleNamespace(
            args=SimpleNamespace(save=str(tmp_path), olmo_core=CoreConfig(pipeline_observation_interval=0.001))
        )

        async def urls(args):
            return ["http://working:1", "http://broken:2"]

        task = asyncio.create_task(pipeline_observer.observe_engines(producer, urls))
        try:
            async with asyncio.timeout(2):
                while len(list(tmp_path.glob("engine_occupancy_*.jsonl"))) < 2:
                    await asyncio.sleep(0.001)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        records = [json.loads(path.read_text()) for path in tmp_path.glob("engine_occupancy_*.jsonl")]
        assert len(records) == 2
        assert next(r for r in records if "working" in r["engine"])["series"][0]["value"] == 3
        assert "503" in next(r for r in records if "broken" in r["engine"])["error"]

    asyncio.run(exercise())


def test_terminal_inventory_distinguishes_buffered_ready_and_shutdown_leftovers(tmp_path):
    async def exercise():
        entry = SimpleNamespace(group=[SimpleNamespace(response_length=7), [SimpleNamespace(response_length=11)]])
        ready = SimpleNamespace(group=[SimpleNamespace(response_length=13)])

        async def finished():
            return ready

        task = asyncio.create_task(finished())
        await task
        producer = SimpleNamespace(
            args=SimpleNamespace(
                sglang_server_concurrency=2, rollout_num_gpus=2, rollout_num_gpus_per_engine=1, save=str(tmp_path)
            ),
            state=SimpleNamespace(generate_fn_semaphore=asyncio.Semaphore(4)),
            _output=SimpleNamespace(_buffer=[entry], _capacity=1),
            _producing_groups={1: [], 2: []},
            _active_tasks={task},
            _scheduler=SimpleNamespace(),
            _producer_resumed=asyncio.Event(),
            # A put can complete just before the worker removes its ready entry.
            _ready_completion_counts={id(entry): pipeline_observer.completion_counts([entry])},
            _shutdown_unqueued_counts=dict(groups=1, samples=4, response_tokens=17),
        )
        pipeline_observer.write_lifecycle(producer, "shutdown_start")
        record = json.loads((tmp_path / "pipeline_lifecycle.jsonl").read_text())
        assert record["completed_queue"] == dict(groups=1, samples=2, response_tokens=18)
        assert record["producer_ready"] == dict(groups=1, samples=1, response_tokens=13)
        assert record["shutdown_unqueued"]["response_tokens"] == 17
        assert producer._output._buffer == [entry] and task.result() is ready
        assert pipeline_observer.completion_counts([object()]) is None

    asyncio.run(exercise())
