"""The pinned native router owns transport isolation; no driver monkeypatch."""

from types import SimpleNamespace

from miles.ray.rollout import router_manager
from miles.router import router

from open_instruct.miles import data_source


def test_data_source_preserves_native_router_spawn_target(monkeypatch):
    monkeypatch.setattr(data_source, "RolloutDataSourceWithBuffer", lambda args: object())
    for enabled in (False, True):
        data_source.DashboardDrainingRolloutDataSource(SimpleNamespace(use_miles_router=enabled))
        assert router_manager.run_miles_router is router.run_router
