"""Driver ordering across optimization, publication, checkpoint and early stops."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct.miles.execution import driver


@pytest.mark.parametrize("stop", [None, "training", "publication", "time", "evaluation", "debug"])
def test_prefetch_overlaps_training_and_is_joined_before_final_drain(monkeypatch, stop):
    events = []
    pending = None
    prepared = asyncio.Event()
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            publication_mode="refresh",
            max_run_seconds=0 if stop == "time" else None,
            diagnostic_interval=0,
            engine_update_timeout=1,
            engine_drain_timeout=1,
        ),
        fully_async=True,
        offload_rollout=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=3,
        eval_interval=1 if stop == "evaluation" else None,
        skip_eval_before_train=True,
        save_interval=1,
        save="checkpoint",
        save_trigger_sentinel=None,
        update_weights_interval=1,
        debug_exit_after_rollout=1 if stop == "debug" else None,
    )

    async def prefetch(i):
        nonlocal pending
        assert pending is None
        pending = i
        events.append(("prepare", i))
        prepared.set()

    async def consume(i):
        nonlocal pending
        assert pending == i
        assert ("publish", i - 1) in events
        events.append(("consume", i))
        pending = None
        return {}

    async def cancel():
        nonlocal pending
        events.append(("cancel", pending))
        pending = None

    async def train(i, batch):
        if i == 0 and stop not in {"evaluation", "debug"}:
            await prepared.wait()
            assert pending == 1
        events.append(("train", i))
        if stop == "training":
            raise RuntimeError("training failed")

    async def publish(*_args, rollout_id=None):
        events.append(("publish", rollout_id))
        if stop == "publication" and rollout_id is not None:
            raise RuntimeError("publication failed")

    async def save(i):
        events.append(("save", i, pending))

    async def boundary(*_args, **_kwargs):
        assert pending is None
        events.append(("boundary",))

    manager = SimpleNamespace(
        get=AsyncMock(return_value={}),
        core_prefetch=prefetch,
        core_get_prepared=consume,
        core_cancel_prepared=cancel,
        core_publication_boundary=boundary,
        save=save,
        dispose=AsyncMock(),
    )
    learner = SimpleNamespace(
        train=train,
        execute_workers=AsyncMock(),
        dispose=AsyncMock(),
        save_model=AsyncMock(),
        finalize_checkpoint=AsyncMock(),
    )
    inference = SimpleNamespace(prepare_rollout=AsyncMock(), dispose=AsyncMock())
    monkeypatch.setattr(driver.prepared_batch, "enabled", lambda _: True)
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(driver.throughput, "report", lambda *_: {"warnings": []})
    monkeypatch.setattr(driver, "stage", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(driver, "evaluation_stage", lambda *_args, **_kwargs: nullcontext())
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
    monkeypatch.setattr(driver.placement_group, "update_weights", publish)
    monkeypatch.setattr(driver, "EvalDispatcher", lambda *_: SimpleNamespace(drain=AsyncMock(), dispatch=AsyncMock()))
    monkeypatch.setattr(driver, "should_run_periodic_action", lambda _i, interval, *_: interval is not None)

    async def scenario():
        if stop in {"training", "publication"}:
            with pytest.raises(RuntimeError, match=f"{stop} failed"):
                await driver.train(args)
        else:
            await driver.train(args)

    asyncio.run(asyncio.wait_for(scenario(), 3))
    if stop is None:
        assert [(e[0], e[1]) for e in events if e[0] in {"prepare", "train", "consume"}] == [
            ("prepare", 1),
            ("train", 0),
            ("consume", 1),
            ("prepare", 2),
            ("train", 1),
            ("consume", 2),
            ("train", 2),
        ]
        assert ("save", 0, 1) in events  # Saving must not consume reserved work.
    if stop in {"debug", "evaluation"}:
        assert not any(e[0] == "prepare" for e in events)
    assert any(e[0] == "cancel" for e in events)
    assert events.index(next(e for e in events if e[0] == "cancel")) < events.index(("boundary",))
