"""Exercise the real driver's startup barriers with CPU-only worker doubles."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.execution import driver


@pytest.mark.parametrize("failure", [None, "serving", "trainer", "connection", "judges", "cancel"])
def test_parallel_driver_startup_join_publication_and_cleanup(monkeypatch, tmp_path, failure):
    events = []
    entered = {name: asyncio.Event() for name in ("serving", "trainer")}
    args = SimpleNamespace(
        olmo_core=CoreConfig(publication_mode="refresh", compiler_cache=False, prepared_batch_overlap=False),
        colocate=False,
        fully_async=True,
        offload_rollout=False,
        num_rollout=1,
        start_rollout_id=0,
        check_weight_update_equal=False,
        eval_interval=None,
        skip_eval_before_train=True,
        save_interval=None,
        save=str(tmp_path),
        save_trigger_sentinel=None,
        debug_exit_after_rollout=None,
        update_weights_interval=1,
    )
    learner = SimpleNamespace(train=AsyncMock(), execute_workers=AsyncMock(), dispose=AsyncMock())
    manager = SimpleNamespace(
        get=AsyncMock(return_value={}), core_publication_boundary=AsyncMock(), dispose=AsyncMock()
    )
    inference = SimpleNamespace(prepare_rollout=AsyncMock(), dispose=AsyncMock())
    models = SimpleNamespace(actor=SimpleNamespace(handle=learner))

    async def initialize(name):
        entered[name].set()
        events.append(name + "-start")
        try:
            await entered["trainer" if name == "serving" else "serving"].wait()
            if name == failure:
                raise RuntimeError(failure)
            if failure in {"serving", "trainer", "cancel"}:
                await asyncio.Event().wait()
            events.append(name + "-ready")
            return (inference, manager, 1) if name == "serving" else models
        finally:
            events.append(name + "-joined")

    async def connect(*_):
        assert "trainer-ready" in events and "serving-ready" in events
        if failure == "connection":
            raise RuntimeError(failure)
        events.append("connected")

    async def barrier(name):
        events.append(name + "-barrier")
        if name == "judges":
            assert "published" in events
            manager.get.assert_not_awaited()
            if failure == "judges":
                raise RuntimeError(failure)

    async def publish(*a, **k):
        assert "connected" in events
        events.append("published")

    async def shutdown(_):
        assert "serving-joined" in events and "trainer-joined" in events
        events.append("shutdown")

    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(driver.throughput, "report", lambda *_: {"warnings": []})
    monkeypatch.setattr(driver.startup, "wait_ready", barrier)
    monkeypatch.setattr(driver.wiring, "launch_worker_manager", Mock(return_value=object()))
    monkeypatch.setattr(driver.wiring, "shutdown_worker_manager", shutdown)
    monkeypatch.setattr(driver.placement_group, "create_rollout_components", lambda _: initialize("serving"))
    monkeypatch.setattr(driver.placement_group, "initialize_training_models", lambda _: initialize("trainer"))
    monkeypatch.setattr(driver.placement_group, "connect_training_models", connect)
    monkeypatch.setattr(driver.placement_group, "update_weights", publish)
    for name in ("init_tracking", "finish_tracking", "define_step_key_metric_group", "remove_rollout_data_refs"):
        monkeypatch.setattr(driver, name, Mock())
    monkeypatch.setattr(driver.object_store, "init_instance", Mock())
    monkeypatch.setattr(driver.startup_cache, "prepare", Mock())
    monkeypatch.setattr(driver.startup_cache, "publish_progress", Mock())
    monkeypatch.setattr(driver.startup_cache, "finish", AsyncMock())
    monkeypatch.setattr(driver, "EvalDispatcher", lambda *_: SimpleNamespace(drain=AsyncMock()))
    monkeypatch.setattr(driver, "should_run_periodic_action", lambda *a: False)

    async def scenario():
        task = asyncio.create_task(driver.train(args))
        if failure == "cancel":
            await entered["serving"].wait()
            await entered["trainer"].wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif failure:
            with pytest.raises(RuntimeError, match=failure):
                await task
        else:
            await task

    asyncio.run(asyncio.wait_for(scenario(), 3))
    assert events[-1] == "shutdown"
    if failure:
        manager.get.assert_not_awaited()
        learner.train.assert_not_awaited()
    else:
        assert events.index("judges-barrier") > events.index("published")
        learner.train.assert_awaited_once()
    if failure in {None, "connection", "judges"}:
        manager.dispose.assert_awaited_once()
        learner.dispose.assert_awaited_once()
        inference.dispose.assert_awaited_once()
