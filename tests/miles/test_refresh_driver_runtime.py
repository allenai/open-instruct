"""Drive publication and terminal cleanup without GPUs or producer progress."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct.miles.execution import driver


@pytest.mark.parametrize("mode", ["refresh", "barrier"])
@pytest.mark.parametrize("failure", [None, "initial", "update", "timeout", "training"])
def test_publication_and_failure_cleanup(monkeypatch, mode, failure):
    events = []
    manager = SimpleNamespace(get=AsyncMock(return_value={}), dispose=AsyncMock())
    learner = SimpleNamespace(train=AsyncMock(), execute_workers=AsyncMock(), dispose=AsyncMock())
    inference = SimpleNamespace(prepare_rollout=AsyncMock(), dispose=AsyncMock())
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            publication_mode=mode,
            max_run_seconds=None,
            diagnostic_interval=0,
            engine_update_timeout=0.01,
            engine_drain_timeout=1,
        ),
        fully_async=True,
        offload_rollout=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=1,
        eval_interval=None,
        save_interval=None,
        save_trigger_sentinel=None,
        update_weights_interval=1,
        debug_exit_after_rollout=None,
    )

    async def boundary(begin, **kwargs):
        events.append(("boundary", begin, kwargs))
        # A refresh publication must succeed even if admission RPCs never respond.
        if mode == "refresh" and not any(e[0] == "publish" for e in events):
            await asyncio.Event().wait()

    async def update_weights(*_args, rollout_id=None):
        events.append(("publish", rollout_id))
        if (failure == "initial" and rollout_id is None) or (failure == "update" and rollout_id == 0):
            raise RuntimeError("publication failed")
        if failure == "timeout" and rollout_id == 0:
            if mode == "barrier":
                raise TimeoutError("transport timed out")
            await asyncio.Event().wait()

    async def train(*_args):
        events.append(("train",))
        if failure == "training":
            raise RuntimeError("training failed")

    manager.core_publication_boundary = AsyncMock(side_effect=boundary)
    learner.train.side_effect = train
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(driver.throughput, "report", lambda *_args: {"warnings": []})
    monkeypatch.setattr(driver, "stage", lambda *_args, **_kwargs: nullcontext())
    for name in ("init_tracking", "finish_tracking", "define_step_key_metric_group", "remove_rollout_data_refs"):
        monkeypatch.setattr(driver, name, Mock())
    monkeypatch.setattr(driver.wandb, "run", None)
    monkeypatch.setattr(driver.object_store, "init_instance", Mock())
    monkeypatch.setattr(driver.startup_cache, "prepare", Mock())
    monkeypatch.setattr(driver.startup_cache, "publish_progress", Mock())
    monkeypatch.setattr(driver.startup_cache, "finish", AsyncMock())
    monkeypatch.setattr(driver.wiring, "launch_worker_manager", Mock(return_value=object()))
    monkeypatch.setattr(driver.wiring, "shutdown_worker_manager", AsyncMock())
    monkeypatch.setattr(
        driver.placement_group, "create_rollout_components", AsyncMock(return_value=(inference, manager, 1))
    )
    monkeypatch.setattr(driver.placement_group, "create_training_models", AsyncMock(return_value=(learner, None)))
    monkeypatch.setattr(driver.placement_group, "update_weights", update_weights)
    monkeypatch.setattr(
        driver, "EvalDispatcher", lambda *_args: SimpleNamespace(drain=AsyncMock(), dispatch=AsyncMock())
    )
    monkeypatch.setattr(driver, "should_run_periodic_action", lambda *_args: False)

    async def scenario():
        if failure:
            error = TimeoutError if failure == "timeout" else RuntimeError
            with pytest.raises(error):
                await driver.train(args)
        else:
            result = await driver.train(args)
            assert result["completed_rollout_ids"] == [0]

    asyncio.run(asyncio.wait_for(scenario(), 3))
    boundaries = [e for e in events if e[0] == "boundary"]
    if mode == "refresh":
        assert boundaries == [("boundary", True, {"abort": True} if failure else {})]
        assert events[0] == ("publish", None)
    else:
        assert boundaries[0] == ("boundary", True, {})
        assert boundaries[-1] == ("boundary", True, {})
        expected_boundaries = {"initial": 2, "training": 3, "update": 4, "timeout": 4, None: 5}
        assert len(boundaries) == expected_boundaries[failure]
    assert learner.train.await_count == (0 if failure == "initial" else 1)
    assert manager.get.await_count == learner.train.await_count
    for component in (manager, learner, inference):
        component.dispose.assert_awaited_once()
    learner.execute_workers.assert_awaited_once_with("close_weight_transport")
