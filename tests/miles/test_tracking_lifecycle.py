"""Worker argument snapshots must contain the driver's initialized tracking ID."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.execution import driver


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
@pytest.mark.parametrize("fail_placement", [False, True])
@pytest.mark.parametrize("ray_connected", [False, True])
@pytest.mark.parametrize("address", [None, "ray://head:10001"])
def test_tracking_precedes_worker_serialization_and_finishes(
    monkeypatch, mode, fail_placement, ray_connected, address
):
    args = SimpleNamespace(
        olmo_core=CoreConfig(compiler_cache=False), fully_async=False, wandb_mode=mode, wandb_run_id=None
    )
    if address is None:
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("RAY_ADDRESS", address)
    events = []
    run = Mock()
    monkeypatch.setattr(driver.wandb, "run", run)
    snapshot = None

    def init_tracking(args):
        events.append("tracking-init")
        args.wandb_run_id = "shared-run-id" if mode != "disabled" else None

    def launch(args, **kwargs):
        nonlocal snapshot
        # Ray serializes these arguments: later driver mutations cannot fix them.
        snapshot = deepcopy(args)
        events.append("worker-launch")
        if fail_placement:
            raise RuntimeError("placement failed")
        return "worker-manager"

    monkeypatch.setattr(driver.throughput, "report", lambda *args: {"warnings": []})
    monkeypatch.setattr(driver.startup_cache, "prepare", Mock())
    monkeypatch.setattr(driver.startup_cache, "finish", AsyncMock())
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: ray_connected)
    ray_init = Mock(side_effect=lambda **kwargs: events.append("ray-init"))
    monkeypatch.setattr(driver.ray, "init", ray_init)
    monkeypatch.setattr(driver, "init_tracking", init_tracking)
    monkeypatch.setattr(driver, "define_step_key_metric_group", Mock())
    monkeypatch.setattr(driver, "finish_tracking", lambda: events.append("tracking-finish"))
    monkeypatch.setattr(driver.wiring, "launch_worker_manager", launch)
    shutdown = AsyncMock()
    monkeypatch.setattr(driver.wiring, "shutdown_worker_manager", shutdown)
    monkeypatch.setattr(driver.object_store, "init_instance", Mock())
    # Stop after workers would have consumed their serialized configuration.
    monkeypatch.setattr(
        driver.placement_group, "create_rollout_components", AsyncMock(side_effect=RuntimeError("serving failed"))
    )

    with pytest.raises(RuntimeError, match="placement failed" if fail_placement else "serving failed"):
        asyncio.run(driver.train(args))

    assert events == ([] if ray_connected else ["ray-init"]) + ["tracking-init", "worker-launch", "tracking-finish"]
    run.finish.assert_called_once_with(exit_code=1)
    if ray_connected:
        ray_init.assert_not_called()
    else:
        ray_init.assert_called_once_with(address=address or "local")
    assert snapshot.wandb_run_id == ("shared-run-id" if mode != "disabled" else None)
    if fail_placement:
        shutdown.assert_not_awaited()
    else:
        shutdown.assert_awaited_once_with("worker-manager")


def test_core_rejects_non_ray_before_starting_services(monkeypatch):
    init = Mock()
    monkeypatch.setattr(driver.ray, "init", init)
    with pytest.raises(ValueError, match="requires the Ray"):
        asyncio.run(driver.train(SimpleNamespace(cluster_backend="kubernetes")))
    init.assert_not_called()


def test_cleanup_failure_marks_run_failed(monkeypatch):
    args = SimpleNamespace(
        olmo_core=CoreConfig(compiler_cache=False),
        fully_async=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=0,
        skip_eval_before_train=True,
        eval_interval=None,
    )
    run = Mock()
    monkeypatch.setattr(driver.wandb, "run", run)
    monkeypatch.setattr(driver.throughput, "report", lambda *args: {"warnings": []})
    monkeypatch.setattr(driver.startup_cache, "prepare", Mock())
    monkeypatch.setattr(driver.startup_cache, "finish", AsyncMock())
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(driver, "init_tracking", Mock())
    monkeypatch.setattr(driver, "define_step_key_metric_group", Mock())
    finish = Mock()
    monkeypatch.setattr(driver, "finish_tracking", finish)
    monkeypatch.setattr(driver.wiring, "launch_worker_manager", Mock(return_value=None))
    monkeypatch.setattr(driver.object_store, "init_instance", Mock())
    inference = SimpleNamespace(dispose=AsyncMock(side_effect=RuntimeError("inference cleanup failed")))
    manager = SimpleNamespace(dispose=AsyncMock())
    learner = SimpleNamespace(execute_workers=AsyncMock(), dispose=AsyncMock())
    monkeypatch.setattr(
        driver.placement_group, "create_rollout_components", AsyncMock(return_value=(inference, manager, 1))
    )
    monkeypatch.setattr(driver.placement_group, "create_training_models", AsyncMock(return_value=(learner, None)))
    monkeypatch.setattr(driver.placement_group, "update_weights", AsyncMock())
    monkeypatch.setattr(driver, "EvalDispatcher", Mock(return_value=SimpleNamespace(drain=AsyncMock())))
    with pytest.raises(RuntimeError, match="inference cleanup failed"):
        asyncio.run(driver.train(args))
    run.finish.assert_called_once_with(exit_code=1)
    finish.assert_called_once()
