"""Retiring a child task must not swallow cancellation of the producer itself."""

import asyncio
from types import SimpleNamespace

import pytest
from miles.backends.core_utils.rollout import async_rollout, rollout_errors


@pytest.mark.parametrize("cancel_during", ["completed_put_stop_waiter", "stopped_put"])
def test_completion_cleanup_preserves_parent_cancellation(cancel_during):
    async def scenario():
        cancelling = asyncio.Event()
        release = asyncio.Event()

        async def delayed_cancel():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelling.set()
                await release.wait()
                raise

        class Buffer:
            async def put(self, item):
                if cancel_during == "stopped_put":
                    await delayed_cancel()
                return True

        producer = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
        group = [SimpleNamespace(group_index=7)]
        producer.args = SimpleNamespace()
        producer._errors = rollout_errors.RolloutErrors()
        producer.data_source = SimpleNamespace(acknowledge_groups=lambda groups: None)
        producer._output = Buffer()
        producer._stop_requested = (
            SimpleNamespace(wait=delayed_cancel) if cancel_during == "completed_put_stop_waiter" else asyncio.Event()
        )
        if cancel_during == "stopped_put":
            producer._stop_requested.set()
        producer._producing_groups = {7: group}
        producer._completed_put_wait_seconds = 0.0
        producer._completed_put_started = None
        task = asyncio.create_task(producer._put_or_stop(SimpleNamespace(prompt_group=group)))
        try:
            await asyncio.wait_for(cancelling.wait(), 1)
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=1)
            assert done, "parent cancellation did not finish joining the child"
            assert task.cancelled(), "child cleanup swallowed the producer cancellation"
            assert producer._completed_put_started is None
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
