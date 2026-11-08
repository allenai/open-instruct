"""Exercise the real drain, conversion and checkpoint ledger across two loops."""

import asyncio
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from miles.backends.core_utils.publication import policy_refresh
from miles.backends.core_utils.rollout import async_rollout, prepared_batch, rollout_errors
from miles.backends.core_utils.rollout.async_buffer import RefreshPolicyDataBuffer
from miles.ray.rollout import rollout_executor
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils import async_utils
from miles.utils.types import Sample
from test_async import make_source


def config():
    return SimpleNamespace(
        olmo_core=SimpleNamespace(prepared_batch_overlap=True, publication_mode="refresh"),
        fully_async=True,
        offload_rollout=False,
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=4,
        update_weights_interval=1,
        async_data_buffer_capacity_factor=1,
        max_weight_staleness=2,
        dynamic_sampling_filter_path=None,
        reward_key=None,
        load_debug_rollout_data=None,
        disable_rollout_trim_samples=False,
        use_dynamic_global_batch_size=False,
        debug_rollout_only=False,
        debug_train_only=False,
        debug_skip_weight_update=False,
        lora_rank=0,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        delay_split_train_data_by_dp=True,
        rollout_top_logprobs_num=0,
        wandb_always_use_train_step=False,
        use_wandb=False,
        use_tensorboard=False,
        use_miles_dashboard=False,
        rollout_batch_size_original=2,
    )


async def on_producer(coro):
    return await asyncio.wrap_future(async_utils.submit(coro))


def entry(source, version=1, length=4):
    [prompt] = source.get_samples(1)
    group = deepcopy(prompt)
    for i, sample in enumerate(group):
        sample.tokens = [9] + [10] * length
        sample.response_length = length
        sample.reward = float(i)
        sample.status = Sample.Status.COMPLETED
        sample.response = "answer"
        sample.rollout_log_probs = [-0.5] * length
        policy_refresh.record_response(
            sample,
            {
                "weight_version": str(version),
                "weight_versions": [dict(version=str(version), start=0, end=length)],
                "output_token_logprobs": [[-0.5, 10]] * length,
            },
        )
    return DataBufferInput(prompt_group=prompt, group=group)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    args = config()
    source = make_source(tmp_path)
    producer = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
    producer.args, producer.data_source = args, source
    producer._publication_paused = False
    producer._draining_groups = producer._reserved_groups = producer._reservation_task = None
    producer._interrupted_groups = []
    producer._errors = rollout_errors.RolloutErrors()
    producer._sample_filter = None
    producer._drain_yield_at = 0.0
    producer._output = RefreshPolicyDataBuffer(
        DataBufferConstructorInput(args, lambda g: source.acknowledge_groups([g]))
    )

    async def idle():
        await asyncio.Event().wait()

    async def initialize():
        producer._worker = asyncio.create_task(idle())

    async_utils.run(initialize())
    executor = rollout_executor.RolloutExecutor.__new__(rollout_executor.RolloutExecutor)
    executor.args, executor.generate_rollout, executor.data_source = args, producer, source
    executor.use_legacy_rollout_v1 = False
    executor._weight_versions_of_model_id = {None: 2}
    executor._rollouts_since_publish_of_model_id = defaultdict(int)
    executor._train_parallel_configs_of_model_id = {None: {"dp_size": 1}}
    executor._diagnostic_writer = None
    executor.custom_convert_samples_to_train_data_func = None
    executor.custom_reward_post_process_func = None
    stored, removed = [], []

    def put(*, value, value_spec):
        stored.append(deepcopy(value))
        return None

    monkeypatch.setattr(rollout_executor.object_store, "get_instance", lambda: SimpleNamespace(put=put))
    monkeypatch.setattr(rollout_executor, "save_debug_rollout_data", Mock())
    monkeypatch.setattr(rollout_executor, "log_rollout_data", Mock())
    monkeypatch.setattr(rollout_executor.RolloutDataInjectionUtil, "should_inject", lambda *_: False)
    monkeypatch.setattr(prepared_batch.tracking, "log", Mock())
    monkeypatch.setattr(prepared_batch, "remove_rollout_data_refs", lambda args, batch: removed.append(batch))
    yield SimpleNamespace(
        executor=executor, producer=producer, source=source, stored=stored, removed=removed, root=tmp_path
    )

    async def finish():
        await producer.release_reservation(consume=False)
        producer._worker.cancel()
        await asyncio.gather(producer._worker, return_exceptions=True)

    async_utils.run(finish())


async def seed(h, versions=(1, 1), length=4):
    for version in versions:
        await h.producer._output.put(entry(h.source, version, length))


@pytest.mark.parametrize("length", [1, 1024, 32768])
def test_prepared_matches_sequential_conversion_and_retains_checkpoint(harness, length):
    h = harness

    async def scenario():
        await on_producer(seed(h, length=length))
        await h.executor.core_prefetch(1)
        pending = h.executor._core_prepared_batch
        await pending.task
        assert len(h.source._pending_groups) == 2
        assert h.producer._reservation_task is None
        assert h.producer._output._delegate._capacity == 0
        # At a save boundary v+1 is published, but this batch is not consumed.
        h.source.save(0)
        restored = make_source(h.root, load=h.root)
        restored.load(0)
        regenerated = restored.get_samples(2)
        assert [g[0].group_index for g in regenerated] == [0, 1]
        assert all(not s.tokens for g in regenerated for s in g)
        expected = deepcopy(h.stored[-1])
        with pytest.raises(RuntimeError, match="Only one"):
            await h.executor.core_prefetch(2)
        h.executor.set_weight_version(3)
        batch = await h.executor.core_get_prepared(1)
        assert batch.sample_indices == [0, 1, 2, 3]
        assert not h.source._pending_groups
        assert h.producer._output._delegate._capacity == 2
        # Replay the same pristine prompt checkpoint through ordinary get.
        h.producer.data_source = restored
        h.executor.data_source = restored
        for group in regenerated:
            restored.add_samples([group])
        old = h.source
        h.source = restored
        await on_producer(seed(h, length=length))
        await h.executor.get(1)
        assert h.stored[-1] == expected
        h.source = old

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_future_version_discards_boundary_group_before_preparation(harness):
    h = harness

    async def scenario():
        # version 0 fits published v2, but is stale for consuming v3 at lag 2.
        await on_producer(seed(h, versions=(0, 1)))
        await h.executor.core_prefetch(1)
        await on_producer(seed(h, versions=(2,)))
        await h.executor._core_prepared_batch.task
        assert h.stored[-1]["sample_indices"] == [2, 3, 4, 5]
        assert 0 not in h.source._pending_groups
        h.executor.set_weight_version(3)
        await h.executor.core_get_prepared(1)
        assert not h.source._pending_groups

    asyncio.run(asyncio.wait_for(scenario(), 5))


@pytest.mark.parametrize("filled", [0, 1, 2])
def test_cancel_empty_partial_and_complete_preparation_recovers_prompts(harness, filled):
    h = harness

    async def scenario():
        await on_producer(seed(h, versions=(1,) * filled))
        await h.executor.core_prefetch(1)
        pending = h.executor._core_prepared_batch
        if filled == 2:
            await pending.task
        else:
            for _ in range(100):
                if h.producer._reserved_groups is not None and len(h.producer._reserved_groups) == filled:
                    break
                await asyncio.sleep(0.001)
        await h.executor.core_cancel_prepared()
        assert h.producer._reserved_groups is None
        assert h.producer._draining_groups is None
        assert h.producer._output._delegate._capacity == 2
        assert len(h.source._pending_groups) == filled
        recovered = h.source.get_samples(filled)
        assert [g[0].group_index for g in recovered] == list(range(filled))
        assert all(not s.tokens for g in recovered for s in g)
        assert len(h.removed) == (filled == 2)
        await h.executor.core_cancel_prepared()  # idempotent

    asyncio.run(asyncio.wait_for(scenario(), 5))


@pytest.mark.parametrize("actual", [2, 4])
def test_unexpected_publication_refuses_consumption_and_recovers(harness, actual):
    h = harness

    async def scenario():
        await on_producer(seed(h))
        await h.executor.core_prefetch(1)
        await h.executor._core_prepared_batch.task
        h.executor.set_weight_version(actual)
        with pytest.raises(RuntimeError, match="expected policy 3"):
            await h.executor.core_get_prepared(1)
        assert len(h.source._pending_groups) == 2
        await h.executor.core_cancel_prepared()
        assert len(h.removed) == 1
        assert h.source.get_samples(1)[0][0].group_index == 0

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_preparation_failure_keeps_ledger_and_restores_capacity(harness, monkeypatch):
    h = harness
    monkeypatch.setattr(
        rollout_executor, "convert_samples_to_train_data", Mock(side_effect=ValueError("bad conversion"))
    )

    async def scenario():
        await on_producer(seed(h))
        await h.executor.core_prefetch(1)
        with pytest.raises(ValueError, match="bad conversion"):
            await h.executor.core_get_prepared(1)
        await h.executor.core_cancel_prepared()
        assert len(h.source._pending_groups) == 2
        assert h.producer._output._delegate._capacity == 2
        assert not h.removed

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_reserved_batch_does_not_expand_queue_budget(harness):
    h = harness

    async def scenario():
        await on_producer(seed(h))
        await h.executor.core_prefetch(1)
        await h.executor._core_prepared_batch.task
        blocked = asyncio.create_task(on_producer(seed(h, versions=(2,))))
        await asyncio.sleep(0.02)
        assert not blocked.done()
        h.executor.set_weight_version(3)
        await h.executor.core_get_prepared(1)
        await blocked
        assert len(h.producer._output._delegate._buffer) == 1

    asyncio.run(asyncio.wait_for(scenario(), 5))


@pytest.mark.parametrize("cancel_after_transfer", [False, True])
def test_cancel_during_cross_loop_handoff_keeps_checkpoint_prompts(harness, monkeypatch, cancel_after_transfer):
    h = harness

    async def scenario():
        await on_producer(seed(h))
        await h.executor.core_prefetch(1)
        await h.executor._core_prepared_batch.task
        h.executor.set_weight_version(3)
        loop = asyncio.get_running_loop()
        original = h.producer.release_reservation

        async def transfer(*, consume):
            if not consume:
                return await original(consume=False)
            groups = await original(consume=True) if cancel_after_transfer else None
            # Cancel on the executor loop after producer-side transfer, before
            # the cross-thread future can deliver its result to the consumer.
            loop.call_soon_threadsafe(consumer.cancel)
            await asyncio.sleep(0.01)
            return groups if cancel_after_transfer else await original(consume=True)

        monkeypatch.setattr(h.producer, "release_reservation", transfer)
        consumer = asyncio.create_task(h.executor.core_get_prepared(1))
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert len(h.source._pending_groups) == 2
        await h.executor.core_cancel_prepared()
        assert h.producer._reserved_groups is None
        assert h.producer._output._delegate._capacity == 2
        assert len(h.removed) == 1
        h.source.save(0)
        restored = make_source(h.root, load=h.root)
        restored.load(0)
        recovered = restored.get_samples(2)
        assert [g[0].group_index for g in recovered] == [0, 1]
        assert all(not sample.tokens for group in recovered for sample in group)
        await h.executor.core_cancel_prepared()
        assert len(h.removed) == 1

    asyncio.run(asyncio.wait_for(scenario(), 5))


@pytest.mark.parametrize(
    "override",
    [
        {"offload_rollout": True},
        {"colocate": True},
        {"fully_async": False},
        {"global_batch_size": 2},
        {"update_weights_interval": 2},
        {"async_data_buffer_capacity_factor": 0.5},
        {"debug_skip_weight_update": True},
    ],
)
def test_unsupported_schedules_keep_sequential_path(override):
    args = config()
    assert prepared_batch.enabled(args)
    vars(args).update(override)
    assert not prepared_batch.enabled(args)


def test_repeated_cancel_races_leave_no_unowned_groups(harness):
    h = harness

    async def scenario():
        for i in range(100):
            await h.executor.core_prefetch(i + 1)
            delivery = asyncio.create_task(on_producer(seed(h, versions=(1,))))
            # Vary whether cancellation reaches the producer before or after
            # its queue waiter has received a completed group.
            for _ in range(i % 4):
                await asyncio.sleep(0)
            await h.executor.core_cancel_prepared()
            await delivery
            assert h.producer._reservation_task is None
            assert h.producer._draining_groups is None
            assert h.producer._output._delegate._capacity == 2
            # Either still buffered, or recovered to the pristine source.
            queued = h.producer._output._delegate._buffer
            if queued:
                h.source.acknowledge_groups([queued.pop().prompt_group])
            else:
                h.source.acknowledge_groups(h.source.get_samples(1))
            assert not h.source._pending_groups

    asyncio.run(asyncio.wait_for(scenario(), 10))
