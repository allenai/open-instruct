"""Distributed serving groups must retire while both peers still exist."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.execution import driver
from open_instruct.miles.publication.engine_drain import Engine, EngineDrain, WeightSnapshot
from open_instruct.miles.training import actor


def install_components(monkeypatch, manager):
    async def create_components(args):
        inference = SimpleNamespace(
            check_weights=manager.check_weights.remote if hasattr(manager, "check_weights") else AsyncMock(),
            prepare_rollout=AsyncMock(),
            dispose=AsyncMock(),
        )
        manager.get = getattr(manager, "generate", SimpleNamespace(remote=AsyncMock()))
        manager.set_weight_version = SimpleNamespace(remote=AsyncMock())
        return inference, manager, 1

    monkeypatch.setattr(driver.placement_group, "create_rollout_components", create_components)
    monkeypatch.setattr(
        driver.wiring,
        "launch_worker_manager",
        lambda *a, **k: SimpleNamespace(dispose=SimpleNamespace(remote=AsyncMock())),
    )


@pytest.mark.parametrize("early_stop", [False, True])
@pytest.mark.parametrize("with_export", [False, True])
@pytest.mark.parametrize("fully_async", [False, True])
@pytest.mark.parametrize("diagnostic_interval", [0, 1])
@pytest.mark.parametrize("start_rollout", [0, 2])
@pytest.mark.parametrize("with_eval", [False, True])
def test_driver_quiesces_then_retires_transport_before_engines(
    monkeypatch, fully_async, diagnostic_interval, start_rollout, with_eval, with_export, early_stop, tmp_path
):
    events = []

    async def event(name):
        events.append(name)

    manager = SimpleNamespace(
        check_weights=SimpleNamespace(remote=lambda **kwargs: event(kwargs["action"])),
        generate=SimpleNamespace(remote=lambda rollout_id: event("generated")),
        dispose=SimpleNamespace(remote=lambda: event("engines-disposed")),
        core_publication_boundary=SimpleNamespace(remote=lambda paused: event(f"paused-{paused}")),
    )
    learner = SimpleNamespace(
        execute_workers=lambda method: event(method),
        dispose=lambda: event("trainer-disposed"),
        update_weights=lambda rollout_id: event("published"),
        train=lambda rollout_id, batch: event("trained"),
        export_hf=lambda rollout_id, path: event(f"export-{rollout_id}-{path}"),
    )

    async def create(*args):
        return learner, None

    monkeypatch.setattr(driver.placement_group, "create_placement_groups", lambda args: {"rollout": None})
    install_components(monkeypatch, manager)
    monkeypatch.setattr(driver.placement_group, "create_training_models", create)
    monkeypatch.setattr(driver.object_store, "init_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "init_tracking", lambda args: None)
    monkeypatch.setattr(driver, "finish_tracking", lambda: None)
    monkeypatch.setattr(driver, "remove_rollout_data_refs", lambda *args: None)
    monkeypatch.setattr(
        driver,
        "EvalDispatcher",
        lambda *args: SimpleNamespace(drain=lambda: event("drained"), dispatch=lambda *a, **k: event("evaluated")),
    )
    args = SimpleNamespace(
        fully_async=fully_async,
        offload_rollout=False,
        check_weight_update_equal=True,
        check_weight_update_allow_quant_error=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=[],
        olmo_core=CoreConfig(diagnostic_interval=diagnostic_interval),
        save_trigger_sentinel=None,
        save_interval=None,
        update_weights_interval=1,
        debug_exit_after_rollout=1 if early_stop else None,
        eval_interval=1 if with_eval else None,
        skip_eval_before_train=False,
        hf_checkpoint="/hf",
        eval_uses_snapshots=False,
        save=str(tmp_path),
        sglang_server_concurrency=64,
        start_rollout_id=start_rollout,
        num_rollout=start_rollout + (2 if early_stop else 1),
    )
    # No save interval: cache progress must still follow completed training,
    # including resumed processes and deliberate early stops.
    monkeypatch.setattr(driver.startup_cache, "publish_progress", lambda args, rid: events.append(f"cache-{rid}"))
    result = asyncio.run(driver.train(args, export_hf="/final" if with_export else None))
    assert events.count(f"cache-{start_rollout}") == 1
    assert events.index("trained") < events.index(f"cache-{start_rollout}")

    assert result["completed_rollout_ids"] == [start_rollout]
    if with_export and not early_stop:
        export_index = events.index(f"export-{start_rollout}-/final")
        assert events.index("drained") < export_index < events.index("close_weight_transport")
        if fully_async:
            assert events[export_index - 1] == "paused-True"
    else:
        assert not any(event.startswith("export-") for event in events)
    timings = [json.loads(line) for line in (tmp_path / "driver_timing.jsonl").read_text().splitlines()]
    evaluations = [row for row in timings if row["stage"] == "evaluation"]
    assert len(evaluations) == events.count("evaluated") == (2 if with_eval else 0)
    if with_eval:
        assert [row["details"]["phase"] for row in evaluations] == ["initial", "periodic"]
        assert all(
            row["passed"] and row["details"]["configured_serving"]["sglang_server_concurrency"] == 64
            for row in evaluations
        )
    checks = 1 + int(diagnostic_interval > 0)
    roundtrips = int(diagnostic_interval > 0) + int(start_rollout > 0)
    assert events.count("compare") == checks
    assert events.count("snapshot") == events.count("reset_tensors") == roundtrips
    assert events.count("published") == 2 + roundtrips
    if checks:
        assert events.index("published") < events.index("compare") < events.index("generated")
    for index, name in enumerate(events):
        if name == "reset_tensors":
            assert events[index - 1 : index + 3] == ["snapshot", "reset_tensors", "published", "compare"]
            if fully_async:
                assert events[index + 3] == "paused-False"
    expected = ["close_weight_transport", "engines-disposed", "trainer-disposed"]
    if fully_async:
        expected.insert(0, "paused-True")
    assert events[-len(expected) :] == expected


def test_transport_destruction_starts_on_both_peers_before_wait(monkeypatch):
    events = []
    updater = SimpleNamespace(
        _model_update_groups="group",
        _group_name="miles",
        rollout_engines=[
            SimpleNamespace(destroy_weights_update_group=lambda name: events.append("engine-request") or "pending")
        ],
    )
    worker = SimpleNamespace(weight_updater=updater)
    monkeypatch.setattr(actor.dist, "destroy_process_group", lambda group: events.append("trainer-destroy"))
    monkeypatch.setattr(actor.async_utils, "submit", lambda value: value)
    monkeypatch.setattr(actor.async_utils, "wait_futures", lambda refs: events.append("engine-wait"))
    actor.OLMoCoreTrainRayActor.close_weight_transport(worker)
    assert events == ["engine-request", "trainer-destroy", "engine-wait"]
    assert updater._model_update_groups is None
    actor.OLMoCoreTrainRayActor.close_weight_transport(worker)
    assert len(events) == 3
    actor.OLMoCoreTrainRayActor.close_weight_transport(SimpleNamespace())
    assert len(events) == 3


@pytest.mark.parametrize("fully_async", [False, True])
def test_failed_periodic_comparison_aborts_before_resume_or_next_generation(monkeypatch, fully_async):
    events = []
    checks = []
    failure = RuntimeError("diagnostic publication mismatch")

    async def event(name):
        events.append(name)

    async def check(**kwargs):
        checks.append(kwargs)
        events.append(kwargs["action"])
        if kwargs["action"] == "compare" and sum(item["action"] == "compare" for item in checks) == 2:
            raise failure

    manager = SimpleNamespace(
        check_weights=SimpleNamespace(remote=check),
        generate=SimpleNamespace(remote=lambda rollout_id: event("generated")),
        dispose=SimpleNamespace(remote=lambda: event("manager-disposed")),
        core_publication_boundary=SimpleNamespace(remote=lambda paused: event(f"paused-{paused}")),
    )
    learner = SimpleNamespace(
        execute_workers=lambda method: event(method),
        dispose=lambda: event("trainer-disposed"),
        update_weights=lambda rollout_id: event("published"),
        train=lambda rollout_id, batch: event("trained"),
    )

    async def create(*args):
        return learner, None

    monkeypatch.setattr(driver.placement_group, "create_placement_groups", lambda args: {"rollout": None})
    install_components(monkeypatch, manager)
    monkeypatch.setattr(driver.placement_group, "create_training_models", create)
    monkeypatch.setattr(driver.object_store, "init_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "init_tracking", lambda args: None)
    monkeypatch.setattr(driver, "finish_tracking", lambda: None)
    monkeypatch.setattr(driver, "remove_rollout_data_refs", lambda *args: None)
    monkeypatch.setattr(driver, "EvalDispatcher", lambda *args: SimpleNamespace(drain=lambda: event("drained")))
    args = SimpleNamespace(
        fully_async=fully_async,
        offload_rollout=False,
        check_weight_update_equal=True,
        check_weight_update_allow_quant_error=False,
        check_weight_update_selector="all",
        check_weight_update_skip_list=["ignored_buffer"],
        olmo_core=CoreConfig(diagnostic_interval=1),
        save_trigger_sentinel=None,
        save_interval=None,
        update_weights_interval=1,
        debug_exit_after_rollout=None,
        eval_interval=None,
        start_rollout_id=0,
        num_rollout=2,
    )
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(driver.train(args))
    assert caught.value is failure
    assert events.count("generated") == events.count("trained") == 1
    assert events.count("published") == 3  # Initial + updated policy + same-version diagnostic replay.
    assert [item["action"] for item in checks] == ["compare", "snapshot", "reset_tensors", "compare"]
    assert all(item["selector"] == "all" for item in checks)
    assert all(item["skip_list"] == ["ignored_buffer"] for item in checks if item["action"] != "snapshot")
    failure_index = len(events) - 1 - events[::-1].index("compare")
    expected_cleanup = ["close_weight_transport", "manager-disposed", "trainer-disposed"]
    if fully_async:
        expected_cleanup.insert(0, "paused-True")
        assert events.count("paused-False") == 1  # Only the successful initial publication resumes the producer.
    assert events[failure_index + 1 :] == expected_cleanup


def test_rolling_checkpoint_does_not_wait_for_groups_waiting_for_one_step_lag(monkeypatch):
    events = []

    async def event(name):
        events.append(name)

    class Publisher:
        def __init__(self, *args):
            async def deliver(identity, snapshot):
                return snapshot.version

            self.controller = EngineDrain([Engine("0", "a", 0)], deliver, max_lag=1)
            self.pending = None

        async def initialize(self):
            pass

        async def optimizer_step_completed(self):
            self.controller.set_step(1)

            async def produce():
                assignment = await self.controller.reserve(1, 1)
                self.controller.decoded(assignment, assignment.requests[0], version=1, tokens=2)
                self.controller.graded(assignment)
                events.append("completed-owned-group")

            self.pending = asyncio.create_task(produce())
            await asyncio.sleep(0)
            assert not self.pending.done()  # Old engine has no admission headroom.

        async def publish(self):
            self.controller.publish(WeightSnapshot(1, (), 0, 0))
            events.append("captured")

        async def quiesce(self):
            assert self.pending is not None
            pytest.fail("Checkpoint must not quiesce inference")

        async def resume(self):
            self.controller.resume()

        async def close(self, *, failed):
            if self.pending is not None and not self.pending.done():
                self.pending.cancel()
                await asyncio.gather(self.pending, return_exceptions=True)
            await self.controller.close()

    manager = SimpleNamespace(
        generate=SimpleNamespace(remote=lambda rollout_id: event("generated")),
        save=SimpleNamespace(remote=lambda rollout_id: event("cursor-saved")),
        dispose=SimpleNamespace(remote=lambda: event("engines-disposed")),
        core_publication_boundary=SimpleNamespace(remote=lambda paused: event(f"paused-{paused}")),
    )
    learner = SimpleNamespace(
        execute_workers=lambda method: event(method),
        dispose=lambda: event("trainer-disposed"),
        update_weights=lambda rollout_id: event("initial-published"),
        train=lambda rollout_id, batch: event("trained"),
        save_model=lambda *a, **k: event("model-saved"),
        finalize_checkpoint=lambda *a: event("committed"),
    )

    async def create(*args):
        return learner, None

    monkeypatch.setattr(driver, "RollingPublication", Publisher)
    monkeypatch.setattr(driver.placement_group, "create_placement_groups", lambda args: {"rollout": None})
    install_components(monkeypatch, manager)
    monkeypatch.setattr(driver.placement_group, "create_training_models", create)
    monkeypatch.setattr(driver.object_store, "init_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "init_tracking", lambda args: None)
    monkeypatch.setattr(driver, "finish_tracking", lambda: None)
    monkeypatch.setattr(driver, "remove_rollout_data_refs", lambda *args: None)
    monkeypatch.setattr(driver, "EvalDispatcher", lambda *args: SimpleNamespace(drain=lambda: event("drained")))
    args = SimpleNamespace(
        fully_async=True,
        offload_rollout=False,
        check_weight_update_equal=False,
        olmo_core=CoreConfig(publication_mode="engine_drain", max_policy_lag=1),
        save_trigger_sentinel=None,
        save_interval=1,
        update_weights_interval=1,
        debug_exit_after_rollout=None,
        eval_interval=None,
        start_rollout_id=0,
        num_rollout=1,
    )
    asyncio.run(driver.train(args))
    assert events.index("captured") < events.index("cursor-saved")
    assert events.count("captured") == 1
    assert events.index("cursor-saved") < events.index("model-saved") < events.index("committed")


def test_refresh_cleanup_uses_configured_drain_budget(monkeypatch, tmp_path):
    deadlines = []
    original_wait_for = asyncio.wait_for

    async def wait_for(awaitable, timeout):
        deadlines.append(timeout)
        return await original_wait_for(awaitable, timeout)

    async def done(*args, **kwargs):
        return None

    manager = SimpleNamespace(
        core_publication_boundary=SimpleNamespace(remote=done), dispose=SimpleNamespace(remote=done)
    )
    learner = SimpleNamespace(update_weights=done, execute_workers=done, dispose=done)

    async def create(*args):
        return learner, None

    monkeypatch.setattr(driver.asyncio, "wait_for", wait_for)
    monkeypatch.setattr(driver.placement_group, "create_placement_groups", lambda args: {"rollout": None})
    install_components(monkeypatch, manager)
    monkeypatch.setattr(driver.placement_group, "create_training_models", create)
    monkeypatch.setattr(driver.object_store, "init_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "init_tracking", lambda args: None)
    monkeypatch.setattr(driver, "finish_tracking", lambda: None)
    monkeypatch.setattr(driver, "EvalDispatcher", lambda *args: SimpleNamespace(drain=done))
    args = SimpleNamespace(
        fully_async=True,
        offload_rollout=False,
        check_weight_update_equal=False,
        olmo_core=CoreConfig(publication_mode="refresh", engine_drain_timeout=900, engine_update_timeout=180),
        eval_interval=None,
        start_rollout_id=0,
        num_rollout=0,
        save=str(tmp_path),
    )
    asyncio.run(driver.train(args))
    assert deadlines == [180, 1080, 60, 120, 60, 60, 120]
    timings = [json.loads(line) for line in (tmp_path / "driver_timing.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in timings[-6:]] == [
        "final_generation_drain",
        "close_weight_transport",
        "rollout_dispose",
        "trainer_dispose",
        "inference_dispose",
        "worker_dispose",
    ]
    assert all(r["passed"] for r in timings)


@pytest.mark.parametrize("fail_at", [None, "cursor-saved", "model-saved", "committed"])
def test_refresh_checkpoint_saves_while_generation_is_unfinished(monkeypatch, tmp_path, fail_at):
    events = []
    live = {}

    async def event(name):
        events.append(name)
        if name == fail_at:
            raise OSError(f"injected {name}")

    async def boundary(paused, *, refresh=False):
        if refresh:
            await event(f"refresh-{paused}")
        elif paused:
            # Only final cleanup may wait for/stop outstanding inference.
            assert any(name in events for name in ["cursor-saved", "model-saved", "committed"])
            live["release"].set()
            await live["request"]
            await event("shutdown-drain")
        else:
            pytest.fail("Checkpoint must not close/reopen inference admission")

    async def generate(rollout_id):
        live["release"] = asyncio.Event()
        live["request"] = asyncio.create_task(live["release"].wait())
        await event("generated")

    async def save_cursor(rollout_id):
        assert not live["request"].done()
        await event("cursor-saved")

    async def save_model(*args, **kwargs):
        assert not live["request"].done()
        # Allow inference to make progress during the trainer save, without
        # requiring the unfinished request to produce a response.
        task = asyncio.create_task(event("inference-progress"))
        await task
        await event("model-saved")

    manager = SimpleNamespace(
        generate=SimpleNamespace(remote=generate),
        save=SimpleNamespace(remote=save_cursor),
        core_publication_boundary=SimpleNamespace(remote=boundary),
        dispose=SimpleNamespace(remote=lambda: event("engines-disposed")),
    )
    learner = SimpleNamespace(
        train=lambda *a: event("trained"),
        update_weights=lambda *a, **k: event("published"),
        save_model=save_model,
        finalize_checkpoint=lambda *a: event("committed"),
        execute_workers=lambda method: event(method),
        dispose=lambda: event("trainer-disposed"),
    )

    async def create(*args):
        return learner, None

    monkeypatch.setattr(driver.placement_group, "create_placement_groups", lambda args: {"rollout": None})
    install_components(monkeypatch, manager)
    monkeypatch.setattr(driver.placement_group, "create_training_models", create)
    monkeypatch.setattr(driver.object_store, "init_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "init_tracking", lambda args: None)
    monkeypatch.setattr(driver, "finish_tracking", lambda: None)
    monkeypatch.setattr(driver, "remove_rollout_data_refs", lambda *args: None)
    monkeypatch.setattr(driver, "EvalDispatcher", lambda *args: SimpleNamespace(drain=lambda: event("eval-drained")))
    args = SimpleNamespace(
        fully_async=True,
        offload_rollout=False,
        check_weight_update_equal=False,
        olmo_core=CoreConfig(publication_mode="refresh"),
        save_trigger_sentinel=None,
        save_interval=1,
        update_weights_interval=1,
        debug_exit_after_rollout=None,
        eval_interval=None,
        start_rollout_id=0,
        num_rollout=1,
        save=str(tmp_path),
    )

    async def exercise():
        async with asyncio.timeout(2):
            await driver.train(args)

    if fail_at:
        with pytest.raises(OSError, match=f"injected {fail_at}"):
            asyncio.run(exercise())
        expected = ["cursor-saved", "model-saved", "committed"]
        assert [name for name in events if name in expected] == expected[: expected.index(fail_at) + 1]
    else:
        asyncio.run(exercise())
        assert events.index("trained") < events.index("cursor-saved")
        assert events.index("cursor-saved") < events.index("inference-progress") < events.index("model-saved")
        assert events.index("model-saved") < events.index("committed") < events.index("shutdown-drain")
    timings = [json.loads(line) for line in (tmp_path / "driver_timing.jsonl").read_text().splitlines()]
    assert not any(row["stage"] == "checkpoint_drain" for row in timings)
    assert [row["passed"] for row in timings if row["stage"] == "checkpoint"] == [fail_at is None]
