"""Distributed serving groups must retire while both peers still exist."""

import asyncio
from types import SimpleNamespace

import pytest

from open_instruct.miles import actor, driver
from open_instruct.miles.config import CoreConfig


@pytest.mark.parametrize("fully_async", [False, True])
@pytest.mark.parametrize("diagnostic_interval", [0, 1])
def test_driver_quiesces_then_retires_transport_before_engines(monkeypatch, fully_async, diagnostic_interval):
    events = []

    async def event(name):
        events.append(name)

    manager = SimpleNamespace(
        check_weights=SimpleNamespace(remote=lambda **kwargs: event("checked")),
        generate=SimpleNamespace(remote=lambda rollout_id: event("generated")),
        dispose=SimpleNamespace(remote=lambda: event("engines-disposed")),
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
        check_weight_update_selector=None,
        check_weight_update_skip_list=[],
        olmo_core=CoreConfig(diagnostic_interval=diagnostic_interval),
        save_trigger_sentinel=None,
        save_interval=None,
        update_weights_interval=1,
        debug_exit_after_rollout=None,
        eval_interval=None,
        start_rollout_id=0,
        num_rollout=1,
    )
    asyncio.run(driver.train(args))
    assert events.count("checked") == 1 + int(diagnostic_interval > 0)
    assert events.index("published") < events.index("checked") < events.index("generated")
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
