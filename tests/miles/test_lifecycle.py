"""Distributed serving groups must retire while both peers still exist."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles import actor, driver
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.engine_drain import Engine, EngineDrain, WeightSnapshot


@pytest.mark.parametrize("with_export", [False, True])
@pytest.mark.parametrize("fully_async", [False, True])
@pytest.mark.parametrize("diagnostic_interval", [0, 1])
@pytest.mark.parametrize("start_rollout", [0, 2])
@pytest.mark.parametrize("with_eval", [False, True])
def test_driver_quiesces_then_retires_transport_before_engines(
    monkeypatch, fully_async, diagnostic_interval, start_rollout, with_eval, with_export, tmp_path
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
        _broadcast=lambda method: event(method),
        dispose=lambda: event("trainer-disposed"),
        update_weights=lambda rollout_id: event("published"),
        train=lambda rollout_id, batch: event("trained"),
        export_hf=lambda rollout_id, path: event(f"export-{rollout_id}-{path}"),
    )

    async def create(*args):
        return learner, None

    monkeypatch.setattr(driver.placement_group, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(driver.placement_group, "create_rollout_manager", lambda *args: (manager, 1))
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
        debug_exit_after_rollout=None,
        eval_interval=1 if with_eval else None,
        skip_eval_before_train=False,
        hf_checkpoint="/hf",
        eval_uses_snapshots=False,
        save=str(tmp_path),
        sglang_server_concurrency=64,
        start_rollout_id=start_rollout,
        num_rollout=start_rollout + 1,
    )
    result = asyncio.run(driver.train(args, export_hf="/final" if with_export else None))
    assert result["completed_rollout_ids"] == [start_rollout]
    if with_export:
        export_index = events.index(f"export-{start_rollout}-/final")
        assert events.index("drained") < export_index < events.index("close_weight_transport")
        if fully_async:
            assert events[export_index - 1] == "paused-True"
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
            SimpleNamespace(
                destroy_weights_update_group=SimpleNamespace(
                    remote=lambda name: events.append("engine-request") or "pending"
                )
            )
        ],
    )
    worker = SimpleNamespace(weight_updater=updater)
    monkeypatch.setattr(actor.dist, "destroy_process_group", lambda group: events.append("trainer-destroy"))
    monkeypatch.setattr(actor.ray, "get", lambda refs: events.append("engine-wait"))
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
        _broadcast=lambda method: event(method),
        dispose=lambda: event("trainer-disposed"),
        update_weights=lambda rollout_id: event("published"),
        train=lambda rollout_id, batch: event("trained"),
    )

    async def create(*args):
        return learner, None

    monkeypatch.setattr(driver.placement_group, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(driver.placement_group, "create_rollout_manager", lambda *args: (manager, 1))
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


def test_rolling_checkpoint_unblocks_groups_waiting_for_one_step_lag(monkeypatch):
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
            await asyncio.wait_for(self.pending, 0.5)
            await self.controller.pause()

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
        _broadcast=lambda method: event(method),
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
    monkeypatch.setattr(driver.placement_group, "create_rollout_manager", lambda *args: (manager, 1))
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
    assert events.index("captured") < events.index("completed-owned-group") < events.index("cursor-saved")
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
    learner = SimpleNamespace(update_weights=done, _broadcast=done, dispose=done)

    async def create(*args):
        return learner, None

    monkeypatch.setattr(driver.asyncio, "wait_for", wait_for)
    monkeypatch.setattr(driver.placement_group, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(driver.placement_group, "create_rollout_manager", lambda *args: (manager, 1))
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
    assert deadlines == [180, 1080, 60, 120, 60]
    timings = [json.loads(line) for line in (tmp_path / "driver_timing.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in timings[-4:]] == [
        "final_generation_drain",
        "close_weight_transport",
        "rollout_dispose",
        "trainer_dispose",
    ]
    assert all(r["passed"] for r in timings)
