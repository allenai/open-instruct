"""Pinned-runtime checks for snapshot bytes and producer lifecycle boundaries."""

import asyncio
from types import SimpleNamespace

import numpy as np
import torch
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample

from open_instruct.miles import actor, draining_rollout, engine_delivery
from open_instruct.miles.async_buffer import HomogeneousPolicyDataBuffer
from open_instruct.miles.draining_rollout import DrainingRolloutFn
from open_instruct.miles.engine_drain import Engine, EngineDrain, WeightSnapshot
from open_instruct.miles.rolling_publication import RollingPublication
from open_instruct.miles.state import PolicyClock


def test_snapshot_has_no_live_parameter_storage(monkeypatch):
    stored = {}
    model = {
        "first": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        "second": torch.ones(4, dtype=torch.bfloat16),
    }
    expected = {key: value.clone() for key, value in model.items()}
    monkeypatch.setattr(engine_delivery.models, "iter_export_state", lambda *a, **k: iter(model.items()))
    monkeypatch.setattr(engine_delivery.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(engine_delivery.dist, "barrier", lambda: None)

    def put(array):
        stored[len(stored)] = array.copy()
        return len(stored) - 1

    monkeypatch.setattr(engine_delivery.ray, "put", put)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            update_weight_buffer_size=32, olmo_core=SimpleNamespace(stream_moe_export=True, expert_publication="fused")
        ),
        clock=PolicyClock(completed_steps=7),
        train_module=None,
        hf_config=None,
        _agree=lambda operation: operation(),
    )
    snapshot = engine_delivery.capture(actor)
    for value in model.values():
        value.add_(100)
    assert snapshot.version == actor.clock.snapshot_ready_step == 7
    assert snapshot.nbytes == 40
    assert len(snapshot.buckets) == 2
    for metadata, reference in snapshot.buckets:
        offset = 0
        for name, shape, dtype in metadata:
            assert dtype == "bfloat16"
            count = int(np.prod(shape))
            restored = (
                torch.from_numpy(stored[reference][offset : offset + count * 2]).view(torch.bfloat16).reshape(shape)
            )
            assert torch.equal(restored, expected[name])
            offset += count * 2


def buffer_args():
    return SimpleNamespace(
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        global_batch_size=2,
        max_weight_staleness=2,
        async_data_buffer_capacity_factor=1,
        dynamic_sampling_filter_path=None,
    )


def entry(index, version=0):
    samples = [
        Sample(index=index * 2 + i, group_index=index, weight_versions=[str(version)], status=Sample.Status.COMPLETED)
        for i in range(2)
    ]
    return DataBufferInput(prompt_group=samples, group=samples)


def test_saturated_buffer_accepts_only_bounded_owned_completions_at_barrier():
    async def run():
        buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(buffer_args(), lambda group: None))
        await buffer.put(entry(0))
        waiting = asyncio.create_task(buffer.put(entry(1)))
        await asyncio.sleep(0)
        assert not waiting.done()
        original = await buffer.reserve_drain_capacity(1)
        await asyncio.wait_for(waiting, 1)
        await buffer.restore_capacity(original)
        another = asyncio.create_task(buffer.put(entry(2)))
        await asyncio.sleep(0)
        assert not another.done()
        assert (await buffer.get(current_version=0)).group[0].group_index == 0
        await asyncio.sleep(0)
        assert not another.done()
        assert (await buffer.get(current_version=0)).group[0].group_index == 1
        await asyncio.wait_for(another, 1)
        assert (await buffer.get(current_version=0)).group[0].group_index == 2

    asyncio.run(run())


def test_producer_boundary_drains_without_cancelling_or_waiting_for_consumer(monkeypatch):
    async def run():
        producer = DrainingRolloutFn.__new__(DrainingRolloutFn)
        producer.args = SimpleNamespace(olmo_core=SimpleNamespace(engine_drain_timeout=1, engine_update_timeout=1))
        producer._event = lambda record: None

        async def deliver(engine, snapshot):
            return snapshot.version

        producer.controller = EngineDrain([Engine("a", "a", 0)], deliver, max_lag=2)
        producer._producer_resumed = asyncio.Event()
        producer._producer_resumed.set()
        producer._producer_idle = asyncio.Event()
        producer._boundary_capacity = None
        producer._publication_paused = False
        producer._output = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(buffer_args(), lambda group: None))
        await producer._output.put(entry(0))
        child = asyncio.create_task(producer._output.put(entry(1)))
        producer._active_tasks = {child}
        producer._producing_groups = {1: entry(1).prompt_group}

        async def worker():
            await child
            producer._producer_idle.set()

        producer._worker = asyncio.create_task(worker())
        await asyncio.wait_for(producer.prepare_publication(), 1)
        assert child.done() and not child.cancelled()
        assert producer._publication_paused
        assert len(producer._output._delegate._buffer) == 2
        await producer.finish_publication()
        assert producer._output._delegate._capacity == 1
        assert not producer._producer_resumed.is_set()
        await producer._output.get(current_version=0)
        producer._resume_if_buffer_allows()
        assert producer._producer_resumed.is_set()

    asyncio.run(run())


def test_boundary_reserves_completed_tasks_still_waiting_for_buffer_insertion():
    async def run():
        producer = DrainingRolloutFn.__new__(DrainingRolloutFn)
        producer.args = SimpleNamespace(olmo_core=SimpleNamespace(engine_drain_timeout=1, engine_update_timeout=1))
        producer._event = lambda record: None

        async def deliver(engine, snapshot):
            return snapshot.version

        producer.controller = EngineDrain([Engine("a", "a", 0)], deliver, max_lag=2)
        producer._producer_resumed = asyncio.Event()
        producer._producer_resumed.set()
        producer._producer_idle = asyncio.Event()
        producer._boundary_capacity = None
        producer._publication_paused = False
        producer._output = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(buffer_args(), lambda group: None))
        await producer._output.put(entry(0))
        # The worker has collected four finished tasks and is inserting their
        # results serially. None is in _active_tasks, but all remain owned.
        producer._active_tasks = set()
        producer._producing_groups = {i: entry(i).prompt_group for i in range(1, 5)}

        async def worker():
            for i in range(1, 5):
                await producer._output.put(entry(i))
                producer._producing_groups.pop(i)
            producer._producer_idle.set()

        producer._worker = asyncio.create_task(worker())
        await asyncio.sleep(0)
        await asyncio.wait_for(producer.prepare_publication(), 1)
        assert not producer._producing_groups
        assert len(producer._output._delegate._buffer) == 5
        assert producer._output._delegate._capacity == 5
        await producer.finish_publication()
        assert producer._output._delegate._capacity == 1
        assert not producer._producer_resumed.is_set()
        for index in range(4):
            await producer._output.get(current_version=0)
            producer._resume_if_buffer_allows()
            assert producer._producer_resumed.is_set() == (index == 3)
            if index < 3:
                # No new owned groups are admitted while draining the excess.
                await producer.prepare_publication()
                await producer.finish_publication()
                assert producer._output._delegate._capacity == 1
                assert not producer._producer_resumed.is_set()

    asyncio.run(run())


def test_direct_request_checks_execution_version_and_releases_before_reward(monkeypatch):
    async def run():
        producer = DrainingRolloutFn.__new__(DrainingRolloutFn)

        async def deliver(engine, weights):
            return weights.version

        producer.controller = EngineDrain([Engine("a", "inc", 0)], deliver, max_lag=2)
        assignment = await producer.controller.reserve(12, 1)
        sample = Sample(index=24, group_index=12, prompt="hello", response="")
        producer._assignments = {id(sample): (assignment, assignment.requests[0])}
        producer._urls = {"a": "http://reserved-engine"}
        producer._probe_delay = 0
        producer.args = SimpleNamespace(olmo_core=SimpleNamespace(engine_drain_timeout=1))
        calls = []

        async def post(url, payload, max_retries):
            calls.append((url, payload, max_retries))
            return {
                "text": "answer",
                "meta_info": {
                    "weight_version": "0",
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [(-0.5, 11, None)],
                },
            }

        monkeypatch.setattr(draining_rollout, "post", post)
        monkeypatch.setattr(draining_rollout, "compute_prompt_ids_from_sample", lambda *args: [1, 2])
        args = SimpleNamespace(
            rollout_max_response_len=16,
            rollout_max_context_len=32,
            use_rollout_routing_replay=False,
            use_rollout_indexer_replay=False,
            lora_rank=0,
            sglang_speculative_algorithm=None,
        )
        inp = SimpleNamespace(sample=sample, state=None, args=args, sampling_params={"max_new_tokens": 16})
        await producer._generate_response(inp)
        assert calls[0][0] == "http://reserved-engine/generate"
        assert calls[0][1]["rid"] == assignment.requests[0]
        assert calls[0][2] == 1
        assert sample.weight_versions == ["0"]
        assert sample.rollout_log_probs == [-0.5]
        assert not producer.controller.engines["a"].requests
        assert producer.controller.status()["groups_in_flight"] == 1
        assert sample.reward is None
        producer.controller.graded(assignment)

    asyncio.run(run())


def test_consuming_clock_advances_before_snapshot_capacity_wait():
    async def run():
        calls = []
        capacity = asyncio.Event()

        async def control(operation, **kwargs):
            calls.append((operation, kwargs))
            if operation == "capacity":
                await capacity.wait()
            return {}

        async def broadcast(method):
            calls.append((method, {}))
            return [WeightSnapshot(1, (), 0, 0)]

        manager = SimpleNamespace(core_engine_drain=SimpleNamespace(remote=control))
        publisher = RollingPublication(None, SimpleNamespace(_broadcast=broadcast), manager)
        publisher.version = 0
        await publisher.optimizer_step_completed()
        pending = asyncio.create_task(publisher.publish())
        await asyncio.sleep(0)
        assert calls == [("step", {"version": 1}), ("capacity", {})]
        capacity.set()
        await pending
        assert [name for name, _ in calls] == ["step", "capacity", "capture_weight_snapshot", "publish"]

    asyncio.run(run())


def test_delivery_location_uses_selected_gpu_uuid_without_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(actor.torch.cuda, "current_device", lambda: 3)

    def properties(device):
        assert device == 3
        return SimpleNamespace(name="NVIDIA B300", uuid="12345678-abcd-ef12-abcd-123456789abc")

    monkeypatch.setattr(actor.torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(actor.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "source-node"))
    worker = SimpleNamespace(clock=PolicyClock(completed_steps=4))
    location = actor.OLMoCoreTrainRayActor.delivery_location(worker)
    assert location == {
        "node_id": "source-node",
        "cuda_visible_devices": "GPU-12345678-abcd-ef12-abcd-123456789abc",
        "device_index": 0,
        "version": 4,
    }


def test_qualification_delay_starts_after_admission_closes():
    async def run():
        events = []

        async def deliver(engine, snapshot):
            return snapshot.version

        producer = DrainingRolloutFn.__new__(DrainingRolloutFn)
        producer.args = SimpleNamespace(olmo_core=SimpleNamespace(engine_drain_timeout=1, engine_update_timeout=1))
        producer.controller = EngineDrain([Engine("a", "a", 0)], deliver, max_lag=2, event=events.append)
        producer._probe_delay = 0.01
        assignment = await producer.controller.reserve(1, 1)
        held = asyncio.create_task(producer._delay_until_draining(assignment, assignment.requests[0]))
        await asyncio.sleep(0)
        assert not any(event["event"] == "qualification_delay_started" for event in events)
        producer.controller.set_step(1)
        producer.controller.publish(WeightSnapshot(1, (), 0, 0))
        await asyncio.wait_for(held, 1)
        names = [event["event"] for event in events]
        assert names.index("drain_started") < names.index("qualification_delay_started")
        producer.controller.decoded(assignment, assignment.requests[0], version=0, tokens=1)
        producer.controller.graded(assignment)
        await producer.controller.barrier()

    asyncio.run(run())


def test_stale_excess_queue_wakes_producer_without_a_successful_dequeue():
    async def run():
        discarded = []
        producer = DrainingRolloutFn.__new__(DrainingRolloutFn)

        async def deliver(engine, snapshot):
            return snapshot.version

        producer.controller = EngineDrain([Engine("a", "a", 3)], deliver, max_lag=2)
        producer._publication_paused = False
        producer._producer_resumed = asyncio.Event()
        producer._draining_groups = []
        producer._interrupted_groups = []
        producer._output = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(buffer_args(), discarded.append))
        original = await producer._output.reserve_drain_capacity(1)
        await producer._output.put(entry(0, 0))
        await producer._output.put(entry(1, 0))
        await producer._output.restore_capacity(original)
        stop = asyncio.Event()

        async def worker():
            await producer._producer_resumed.wait()
            await producer._output.put(entry(2, 3))
            await stop.wait()

        producer._worker = asyncio.create_task(worker())
        try:
            result = await asyncio.wait_for(producer._next_group(3), 2)
            assert result.group[0].group_index == 2
            assert len(discarded) == 2
            assert producer._producer_resumed.is_set()
        finally:
            stop.set()
            await producer._worker

    asyncio.run(run())
