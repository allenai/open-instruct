"""Exercise diagnostic weight mutation while requests remain in flight."""

import asyncio
import sys
from contextlib import nullcontext
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct.miles.execution import publication_checks


def controller(*, failure=None):
    events = []
    engines = []
    for index in range(2):
        engine = SimpleNamespace(paused=False, requests=[{"tokens": [11, 12], "logprobs": [-0.2, -0.3]}])

        async def pause(engine=engine, index=index):
            events.append(("pause", index))
            if failure == "pause" and index == 1:
                raise RuntimeError("pause failed")
            engine.paused = True

        engine.pause_generation = AsyncMock(side_effect=pause)
        engines.append(engine)

    async def check(action, **kwargs):
        events.append((action, kwargs))
        assert all(engine.paused for engine in engines), "live decoding could observe reset weights"
        if failure == action:
            raise RuntimeError(f"{action} failed")

    info = SimpleNamespace(rollout_engines=engines, snapshot_cell_id_to_hashes={"engine": "incarnation"})
    inference = SimpleNamespace(
        start_update_weights=AsyncMock(return_value=info),
        end_update_weights=AsyncMock(),
        abort_update_weights=AsyncMock(),
        check_weights=AsyncMock(side_effect=check),
    )
    return inference, engines, events


def test_refresh_round_trip_pauses_every_engine_before_mutating_weights():
    inference, engines, events = controller()
    requests = [engine.requests for engine in engines]
    asyncio.run(publication_checks.prepare_round_trip(inference, refresh=True, selector="main", skip_list=["aux"]))
    assert [event[0] for event in events] == ["pause", "pause", "snapshot", "reset_tensors"]
    inference.end_update_weights.assert_awaited_once_with({"engine": "incarnation"})
    inference.abort_update_weights.assert_not_awaited()
    assert events[-1][1] == {"selector": "main", "skip_list": ["aux"]}
    for engine, original in zip(engines, requests, strict=True):
        assert engine.paused  # Only the successful publisher may resume decoding.
        assert engine.requests is original
        assert engine.requests == [{"tokens": [11, 12], "logprobs": [-0.2, -0.3]}]


@pytest.mark.parametrize("failure", ["pause", "snapshot", "reset_tensors"])
def test_failed_round_trip_never_resumes_or_leaves_controller_locked(failure):
    inference, engines, events = controller(failure=failure)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(publication_checks.prepare_round_trip(inference, refresh=True, selector="all", skip_list=None))
    if failure == "pause":
        inference.abort_update_weights.assert_awaited_once()
        inference.end_update_weights.assert_not_awaited()
        inference.check_weights.assert_not_awaited()
    else:
        inference.end_update_weights.assert_awaited_once()
        inference.abort_update_weights.assert_not_awaited()
        assert all(engine.paused for engine in engines)


def test_drained_barrier_round_trip_keeps_existing_check_path():
    inference, engines, events = controller()
    for engine in engines:
        engine.paused = True
    asyncio.run(publication_checks.prepare_round_trip(inference, refresh=False, selector="all", skip_list=None))
    assert [event[0] for event in events] == ["snapshot", "reset_tensors"]
    inference.start_update_weights.assert_not_awaited()
    inference.end_update_weights.assert_not_awaited()


def test_pause_failure_preserves_primary_error_if_unlock_also_fails():
    inference, _, _ = controller(failure="pause")
    inference.abort_update_weights.side_effect = RuntimeError("unlock failed")
    with pytest.raises(RuntimeError, match="pause failed") as caught:
        asyncio.run(publication_checks.prepare_round_trip(inference, refresh=True, selector="all", skip_list=None))
    assert "unlock failed" in caught.value.__notes__[0]
    inference.check_weights.assert_not_awaited()


def test_controller_release_failure_does_not_reset_or_attempt_second_unlock():
    inference, engines, _ = controller()
    inference.end_update_weights.side_effect = RuntimeError("controller release failed")
    with pytest.raises(RuntimeError, match="controller release failed"):
        asyncio.run(publication_checks.prepare_round_trip(inference, refresh=True, selector="all", skip_list=None))
    inference.abort_update_weights.assert_not_awaited()
    inference.check_weights.assert_not_awaited()
    assert all(engine.paused for engine in engines)


@pytest.mark.parametrize("diagnostic_interval", [0, 1])
def test_real_driver_keeps_live_requests_safe_during_optional_round_trip(monkeypatch, diagnostic_interval):
    inference, engines, events = controller()
    inference.prepare_rollout = AsyncMock()
    inference.dispose = AsyncMock()
    for engine in engines:
        engine.valid_weights = True

    async def check(action, **kwargs):
        if action in {"snapshot", "reset_tensors"}:
            assert all(engine.paused for engine in engines), "diagnostic raced with live decoding"
        if action == "reset_tensors":
            for engine in engines:
                engine.valid_weights = False
        if action == "compare":
            assert all(engine.valid_weights for engine in engines)
        events.append((action, kwargs))

    inference.check_weights.side_effect = check

    async def update_weights(*args, **kwargs):
        # Model the native publisher's pause -> load -> resume contract. The
        # driver must pause again before any subsequent diagnostic mutation.
        for engine in engines:
            engine.paused = True
            engine.valid_weights = True
            engine.paused = False

    manager = SimpleNamespace(
        get=SimpleNamespace(remote=AsyncMock(return_value={})),
        dispose=SimpleNamespace(remote=AsyncMock()),
        core_publication_boundary=SimpleNamespace(remote=AsyncMock()),
    )
    learner = SimpleNamespace(train=AsyncMock(), dispose=AsyncMock(), execute_workers=AsyncMock())
    workers = SimpleNamespace(dispose=SimpleNamespace(remote=AsyncMock()))
    placement = SimpleNamespace(
        create_rollout_components=AsyncMock(return_value=(inference, manager, 1)),
        create_training_models=AsyncMock(return_value=(learner, None)),
        update_weights=AsyncMock(side_effect=update_weights),
    )

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)

    module("miles")
    module(
        "miles.ray",
        placement_group=placement,
        wiring=SimpleNamespace(launch_worker_manager=Mock(return_value=workers)),
    )
    module("miles.ray.rollout")
    module("miles.ray.rollout.eval_dispatch", EvalDispatcher=Mock(return_value=SimpleNamespace(drain=AsyncMock())))
    module("miles.utils", object_store=SimpleNamespace(init_instance=Mock()))
    module(
        "miles.utils.compiler_cache",
        startup_cache=SimpleNamespace(
            prepare=Mock(), finish=AsyncMock(), configure_specs=Mock(), publish_progress=Mock()
        ),
    )
    module("miles.utils.data", remove_rollout_data_refs=Mock())
    module("miles.utils.hf_config", HF_EXPORT_COMPLETE_MARKER=".complete")
    module("miles.utils.misc", should_run_periodic_action=Mock(return_value=False))
    module("miles.utils.tracking_utils")
    module(
        "miles.utils.tracking_utils.tracking",
        define_step_key_metric_group=Mock(),
        finish_tracking=Mock(),
        init_tracking=Mock(),
    )
    module("miles.backends.core_utils.publication.rolling_publication", RollingPublication=Mock())
    spec = importlib_util.spec_from_file_location(
        "test_publication_driver", Path(publication_checks.__file__).with_name("driver.py")
    )
    driver = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, "stage", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(driver.throughput, "report", lambda *a: {"warnings": []})
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            publication_mode="refresh",
            diagnostic_interval=diagnostic_interval,
            engine_update_timeout=1,
            engine_drain_timeout=1,
        ),
        fully_async=True,
        offload_rollout=False,
        check_weight_update_equal=True,
        check_weight_update_selector="all",
        check_weight_update_skip_list=None,
        check_weight_update_allow_quant_error=False,
        start_rollout_id=0,
        num_rollout=1,
        eval_interval=None,
        save_trigger_sentinel=None,
        save_interval=None,
        debug_exit_after_rollout=None,
    )
    result = asyncio.run(driver.train(args))
    assert result["completed_rollout_ids"] == [0]
    assert placement.update_weights.await_count == 2 + diagnostic_interval
    assert [event[0] for event in events].count("reset_tensors") == diagnostic_interval
    assert inference.start_update_weights.await_count == diagnostic_interval
    # No synchronous drain was substituted for a refresh publication. The last
    # boundary belongs to ordinary final shutdown after the training loop.
    boundaries = manager.core_publication_boundary.remote.await_args_list
    assert all(call.kwargs == {"refresh": True} for call in boundaries[:-1])
    assert boundaries[-1].kwargs == {}
    assert all(engine.valid_weights and not engine.paused for engine in engines)
