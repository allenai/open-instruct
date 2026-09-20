"""Serving admission must preserve queued work across an evaluation window."""

import asyncio

import pytest

from open_instruct.miles.generation_admission import GenerationAdmission


def test_pause_drains_active_call_but_preserves_waiter_identity():
    async def scenario():
        gate = GenerationAdmission(1)
        started, finish = asyncio.Event(), asyncio.Event()
        samples = [object(), object()]
        seen = []

        async def generate(sample):
            async with gate:
                seen.append(sample)
                if sample is samples[0]:
                    started.set()
                    await finish.wait()
            return sample

        active = asyncio.create_task(generate(samples[0]))
        await started.wait()
        queued = asyncio.create_task(generate(samples[1]))
        await asyncio.sleep(0)
        gate.pause()
        drained = asyncio.create_task(gate.wait_idle())
        await asyncio.sleep(0)
        assert not drained.done()
        finish.set()
        await asyncio.wait_for(drained, 1)
        assert await active is samples[0]
        assert not queued.done() and seen == samples[:1]
        assert gate.active == 0 and gate.waiting == 1
        gate.resume()
        assert await asyncio.wait_for(queued, 1) is samples[1]
        assert seen == samples and gate.active == gate.waiting == 0
        assert gate._semaphore._value == 1

    asyncio.run(scenario())


def test_pause_wins_race_with_a_released_semaphore_slot():
    async def scenario():
        gate = GenerationAdmission(1)
        await gate.acquire()
        queued = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)
        gate.release()  # Wakes queued; it has not executed its continuation yet.
        gate.pause()
        await asyncio.sleep(0)
        assert not queued.done() and gate.active == 0
        assert gate._semaphore._value == 1
        gate.resume()
        await asyncio.wait_for(queued, 1)
        gate.release()

    asyncio.run(scenario())


@pytest.mark.parametrize("paused", [False, True])
def test_cancelled_waiter_does_not_consume_a_permit(paused):
    async def scenario():
        gate = GenerationAdmission(1)
        await gate.acquire()
        if paused:
            gate.pause()
        waiter = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        gate.release()
        assert gate.active == gate.waiting == 0
        gate.resume()
        async with gate:
            assert gate.active == 1
        assert gate._semaphore._value == 1

    asyncio.run(scenario())


def test_abort_cleans_generation_waiters_and_post_generation_rewards():
    async def scenario():
        gate = GenerationAdmission(1)
        reward_started = asyncio.Event()
        active_started = asyncio.Event()

        async def reward():
            async with gate:
                pass
            reward_started.set()
            await asyncio.Event().wait()

        async def generation():
            async with gate:
                active_started.set()
                await asyncio.Event().wait()

        tasks = [asyncio.create_task(reward())]
        await reward_started.wait()
        tasks.append(asyncio.create_task(generation()))
        await active_started.wait()
        tasks.append(asyncio.create_task(generation()))
        await asyncio.sleep(0)
        await gate.abort()
        assert all(task.cancelled() for task in tasks)
        assert gate.active == gate.waiting == 0
        late = asyncio.create_task(gate.acquire())
        await asyncio.gather(late, return_exceptions=True)
        assert late.cancelled()
        with pytest.raises(RuntimeError, match="stopped"):
            gate.resume()

    asyncio.run(scenario())
