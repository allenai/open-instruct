"""Concurrency contract tests, independent of the model/runtime dependencies."""

import asyncio

import pytest
from scripts.miles import analyze_engine_drain

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.publication.engine_drain import Engine, EngineDrain, WeightSnapshot


def snapshot(version):
    return WeightSnapshot(version, (b"immutable weights",), 17, 0.01)


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def test_slow_engine_does_not_hold_fast_engine_or_trainer():
    async def run():
        events = []
        delivering = asyncio.Event()
        release_delivery = asyncio.Event()

        async def deliver(engine, weights):
            if engine == "fast":
                delivering.set()
                await release_delivery.wait()
            return weights.version

        ctl = EngineDrain([Engine("fast", "a", 0), Engine("slow", "b", 0)], deliver, max_lag=2, event=events.append)
        fast = await ctl.reserve(10, 2)
        slow = await ctl.reserve(20, 2)
        assert (fast.engine, slow.engine) == ("fast", "slow")
        # The second slow request has not even reached the HTTP semaphore yet.
        ctl.decoded(fast, fast.requests[0], version=0, tokens=12)
        ctl.decoded(fast, fast.requests[1], version=0, tokens=15)
        ctl.set_step(1)
        ctl.publish(snapshot(1))
        await delivering.wait()
        # Model work may advance while a delivery still owns immutable v1.
        ctl.set_step(2)
        assert ctl.status()["retained_snapshot_bytes"] == 17
        release_delivery.set()
        await until(lambda: ctl.engines["fast"].state == "serving")
        next_group = await ctl.reserve(30, 2)
        assert next_group.version == 1 and next_group.engine == "fast"
        assert ctl.engines["slow"].version == 0
        assert ctl.engines["slow"].state == "draining"
        # Slow grading on the first group did not hold its engine's old weights.
        ctl.graded(fast)
        for request in slow.requests:
            ctl.decoded(slow, request, version=0, tokens=7)
        await ctl.barrier()
        assert ctl.status()["retained_snapshot_bytes"] == 0
        assert any(e["event"] == "engine_reopened" and e["engine"] == "fast" for e in events)
        assert not any("cancel" in e["event"] for e in events)

    asyncio.run(run())


def test_group_reservation_is_atomic_with_drain_and_includes_unsent_siblings():
    async def run():
        async def deliver(engine, weights):
            return weights.version

        ctl = EngineDrain([Engine("one", "a", 0)], deliver, max_lag=2)
        old = await ctl.reserve(1, 8)
        ctl.set_step(1)
        ctl.publish(snapshot(1))
        admission = asyncio.create_task(ctl.reserve(2, 8))
        for request in old.requests[:-1]:
            ctl.decoded(old, request, version=0, tokens=1)
        await asyncio.sleep(0)
        assert not admission.done()
        assert ctl.engines["one"].version == 0
        ctl.decoded(old, old.requests[-1], version=0, tokens=1)
        new = await admission
        assert new.version == 1
        assert len(ctl.engines["one"].requests) == 8
        ctl.graded(old)
        with pytest.raises(RuntimeError, match="retired twice"):
            ctl.graded(old)
        await ctl.barrier()

    asyncio.run(run())


def test_snapshot_capacity_and_per_engine_version_order():
    async def run():
        release = asyncio.Event()
        calls = []

        async def deliver(engine, weights):
            calls.append((engine, weights.version))
            await release.wait()
            return weights.version

        ctl = EngineDrain([Engine("a", "a", 0), Engine("b", "b", 0)], deliver, max_lag=3, capacity=2)
        ctl.set_step(1)
        ctl.publish(snapshot(1))
        ctl.set_step(2)
        ctl.publish(snapshot(2))
        capacity = asyncio.create_task(ctl.wait_capacity())
        await asyncio.sleep(0)
        assert not capacity.done()
        assert ctl.status()["retained_snapshot_bytes"] == 34
        ctl.set_step(3)
        with pytest.raises(RuntimeError, match="capacity exhausted"):
            ctl.publish(snapshot(3))
        release.set()
        await capacity
        await ctl.barrier()
        for engine in ("a", "b"):
            assert [version for identity, version in calls if identity == engine] == [1, 2]
        assert ctl.status()["fleet_converged"] == 2
        assert not ctl._snapshots

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["drain_timeout", "update_timeout", "partial_update", "wrong_ack"])
def test_failure_closes_admission_and_reaches_barrier(failure):
    async def run():
        async def deliver(engine, weights):
            if failure == "update_timeout":
                await asyncio.Event().wait()
            if failure == "partial_update":
                raise OSError("bucket 2 failed")
            return -1 if failure == "wrong_ack" else weights.version

        ctl = EngineDrain([Engine("a", "incarnation", 0)], deliver, max_lag=2, drain_timeout=0.01, update_timeout=0.01)
        if failure == "drain_timeout":
            await ctl.reserve(1, 1)
        ctl.set_step(1)
        ctl.publish(snapshot(1))
        with pytest.raises(RuntimeError, match="rolling publication failed"):
            await ctl.barrier()
        assert ctl.engines["a"].state == "unavailable"
        assert ctl.engines["a"].version == 0
        with pytest.raises(RuntimeError, match="rolling publication failed"):
            await ctl.reserve(2, 1)
        with pytest.raises(RuntimeError, match="rolling publication failed"):
            await ctl.wait_capacity()

    asyncio.run(run())


def test_response_version_and_incarnation_are_verified():
    async def run():
        async def deliver(engine, weights):
            return weights.version

        ctl = EngineDrain([Engine("a", "original", 0)], deliver, max_lag=2)
        group = await ctl.reserve(1, 1)
        with pytest.raises(ValueError, match="ownership/version mismatch"):
            ctl.decoded(group, group.requests[0], version=1, tokens=1)
        assert ctl.engines["a"].state == "unavailable"

    asyncio.run(run())


def test_lag_headroom_blocks_old_admission_but_does_not_relabel_requests():
    async def run():
        async def deliver(engine, weights):
            return weights.version

        ctl = EngineDrain([Engine("a", "a", 0)], deliver, max_lag=1)
        ctl.set_step(1)
        waiting = asyncio.create_task(ctl.reserve(1, 1))
        await asyncio.sleep(0)
        assert not waiting.done()
        ctl.publish(snapshot(1))
        group = await waiting
        assert group.version == 1
        await ctl.barrier()

    asyncio.run(run())


def test_pause_resume_shutdown_and_duplicate_groups():
    async def run():
        async def deliver(engine, weights):
            return weights.version

        ctl = EngineDrain([Engine("a", "a", 0)], deliver, max_lag=2)
        group = await ctl.reserve(1, 1)
        with pytest.raises(ValueError, match="already owns"):
            await ctl.reserve(1, 1)
        ctl.decoded(group, group.requests[0], version=0, tokens=1)
        ctl.graded(group)
        retry = await ctl.reserve(1, 1)
        assert retry.requests != group.requests
        assert retry.attempt > group.attempt
        ctl.decoded(retry, retry.requests[0], version=0, tokens=1)
        ctl.graded(retry)
        await ctl.pause()
        waiting = asyncio.create_task(ctl.reserve(2, 1))
        await asyncio.sleep(0)
        assert not waiting.done()
        ctl.resume()
        await waiting
        await ctl.close()
        with pytest.raises(RuntimeError, match="shutdown"):
            await ctl.reserve(3, 1)

    asyncio.run(run())


def test_mode_configuration_rejects_unsupported_combinations():
    options = dict(
        hf_checkpoint="/fixture/hf",
        global_batch_size=4,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        fully_async=True,
    )
    core = CoreConfig(publication_mode="engine_drain", max_policy_lag=2)
    RunConfig(core, options).validate()
    for change, match in [
        ({"fully_async": False}, "fully_async"),
        ({"rollout_num_gpus_per_engine": 2}, "TP1"),
        ({"eval_num_gpus": 1}, "shared-engine"),
        ({"prefill_num_servers": 1}, "prefill/decode"),
        ({"sglang_config": "/some/servers.yaml"}, "single-turn"),
        ({"load_debug_rollout_data": "/some/batch.pt"}, "single-turn"),
        ({"use_fault_tolerance": True}, "use_fault_tolerance"),
        ({"custom_generate_function_path": "custom.generate"}, "single-turn"),
    ]:
        with pytest.raises(ValueError, match=match):
            RunConfig(core, options | change).validate()
    with pytest.raises(ValueError, match="engine_drain_timeout"):
        CoreConfig(engine_drain_timeout=0)


def test_timeline_audit_does_not_confuse_phase_sums_with_overlap():
    events = [
        {"event": "group_reserved", "engine": "fast", "version": 0, "group": 1, "requests": ["r1"]},
        {
            "event": "decode_finished",
            "engine": "fast",
            "group": 1,
            "request": "r1",
            "tokens": 10,
            "assigned_version": 0,
            "executed_version": 0,
        },
        {"event": "drain_started", "engine": "fast", "version": 0, "time": 1},
        {"event": "drain_started", "engine": "slow", "version": 0, "time": 1},
        {"event": "drain_finished", "engine": "fast", "time": 2},
        {"event": "update_started", "engine": "fast", "time": 2},
        {"event": "engine_reopened", "engine": "fast", "version": 1, "time": 3},
        {"event": "drain_finished", "engine": "slow", "time": 5},
    ]
    result = analyze_engine_drain.analyze(
        events,
        [
            {"stage": "training", "passed": True, "started_unix": 2, "seconds": 2, "rollout_id": 1},
            {"stage": "training", "passed": True, "started_unix": 6, "seconds": 2, "rollout_id": 2},
        ],
    )
    assert result["ownership_errors"] == []
    assert len(result["fast_reopened_while_older_peer_drained"]) == 1
    assert [r["rollout_id"] for r in result["optimizer_completions_during_publication"]] == [1]
    assert result["generated_tokens_observed"] == 10
