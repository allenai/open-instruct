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
def test_tracking_precedes_worker_serialization_and_finishes(monkeypatch, mode, fail_placement, ray_connected):
    args = SimpleNamespace(
        olmo_core=CoreConfig(compiler_cache=False), fully_async=False, wandb_mode=mode, wandb_run_id=None
    )
    events = []
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
    monkeypatch.setattr(driver.ray, "init", lambda: events.append("ray-init"))
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
    assert snapshot.wandb_run_id == ("shared-run-id" if mode != "disabled" else None)
    if fail_placement:
        shutdown.assert_not_awaited()
    else:
        shutdown.assert_awaited_once_with("worker-manager")
