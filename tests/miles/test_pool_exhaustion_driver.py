"""Driver finalization when finite pools cannot fill another collection."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from miles.backends.core_utils.rollout.errors import PromptPoolExhausted

from open_instruct.miles.execution import driver


def test_old_exposure_only_runtime_is_rejected_before_startup(monkeypatch):
    monkeypatch.delattr(driver.rollout_errors, "FINITE_POOL_PROVIDERS")
    args = SimpleNamespace(olmo_core=SimpleNamespace(prompt_provider="pools"))
    with pytest.raises(RuntimeError, match="companion MILES runtime"):
        asyncio.run(driver.train(args))


@pytest.mark.parametrize("saved", [True, False])
def test_exhaustion_preserves_last_complete_checkpoint_without_duplicate_save(monkeypatch, saved):
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            prompt_provider="pools",
            publication_mode="barrier",
            max_run_seconds=None,
            diagnostic_interval=0,
            engine_update_timeout=1,
            engine_drain_timeout=1,
        ),
        fully_async=True,
        offload_rollout=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=10,
        eval_interval=None,
        save_interval=1 if saved else None,
        save="checkpoint",
        save_trigger_sentinel=None,
        update_weights_interval=1,
        debug_exit_after_rollout=None,
    )
    manager = SimpleNamespace(
        get=AsyncMock(side_effect=[{}, PromptPoolExhausted("empty")]),
        core_publication_boundary=AsyncMock(),
        core_commit_exposure=AsyncMock(return_value={"rollout/pool/math/trained": 1}),
        save=AsyncMock(),
        dispose=AsyncMock(),
    )
    learner = SimpleNamespace(
        train=AsyncMock(),
        execute_workers=AsyncMock(),
        dispose=AsyncMock(),
        save_model=AsyncMock(),
        finalize_checkpoint=AsyncMock(),
        export_hf=AsyncMock(),
    )
    inference = SimpleNamespace(prepare_rollout=AsyncMock(), dispose=AsyncMock())
    monkeypatch.setattr(driver.prepared_batch, "enabled", lambda _: False)
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(driver.throughput, "report", lambda *_: {"warnings": []})
    monkeypatch.setattr(driver, "stage", lambda *_a, **_k: nullcontext())
    for name in (
        "init_tracking",
        "finish_tracking",
        "define_step_key_metric_group",
        "remove_rollout_data_refs",
        "log",
    ):
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
    monkeypatch.setattr(driver.placement_group, "update_weights", AsyncMock())
    monkeypatch.setattr(driver, "EvalDispatcher", lambda *_: SimpleNamespace(drain=AsyncMock(), dispatch=AsyncMock()))
    monkeypatch.setattr(driver, "should_run_periodic_action", lambda _i, interval, *_: interval is not None)
    result = asyncio.run(driver.train(args, export_hf="final"))
    assert result["pools_exhausted"] and result["completed_rollout_ids"] == [0]
    learner.train.assert_awaited_once()
    manager.core_commit_exposure.assert_awaited_once_with(0)
    driver.log.assert_called_once_with(args, {"pool/update": 1, "pool/trained/math": 1}, step_key="pool/update")
    driver.define_step_key_metric_group.assert_any_call("pool", "pool/update")
    manager.save.assert_awaited_once_with(0)
    learner.save_model.assert_awaited_once_with(0, force_sync=True)
    learner.finalize_checkpoint.assert_awaited_once_with(0)
    learner.export_hf.assert_awaited_once_with(0, "final")
