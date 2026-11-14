"""Offline validation of diagnostic replay's native-config identity contract."""

from types import SimpleNamespace

import pytest
import torch
from scripts.miles import diagnose_math_kda_dispatch as replay


def test_restore_dispatch_uses_runtime_config_objects_and_unwraps(monkeypatch):
    config = SimpleNamespace(kwargs={"BV": 32}, num_warps=2, num_stages=3)
    tuner = SimpleNamespace(configs=[config], cache={"stale": None})
    module = SimpleNamespace(kernel=SimpleNamespace(fn=tuner))
    monkeypatch.setattr(replay.importlib, "import_module", lambda name: module if name == "fla.example" else None)
    count = replay.restore_dispatch(
        {
            "fla.example.kernel.fn": {
                "cache": {"(8, 'torch.bfloat16')": {"kwargs": {"BV": 32}, "num_warps": 2, "num_stages": 3}}
            }
        }
    )
    assert count == {"fla.example.kernel.fn": 1}
    assert tuner.cache == {(8, "torch.bfloat16"): config}
    assert next(iter(tuner.cache.values())) is config


def test_restore_dispatch_rejects_config_absent_from_pinned_runtime(monkeypatch):
    tuner = SimpleNamespace(configs=[SimpleNamespace(kwargs={"BV": 32})], cache={})
    monkeypatch.setattr(replay.importlib, "import_module", lambda name: SimpleNamespace(kernel=tuner))
    with pytest.raises(ValueError, match="not uniquely present"):
        replay.restore_dispatch({"fla.example.kernel": {"cache": {"(8,)": {"kwargs": {"BV": 128}}}}})
    assert tuner.cache == {}


def test_output_comparison_preserves_actual_differences_and_finiteness():
    a = torch.tensor([0, 1], dtype=torch.bfloat16)
    b = torch.tensor([0, 1.5], dtype=torch.bfloat16)
    assert replay.difference(a, b) == {
        "equal": False,
        "finite": True,
        "max_abs": 0.5,
        "changed_elements": 1,
        "elements": 2,
    }
    assert not replay.difference(a, b * float("nan"))["finite"]
    with pytest.raises(ValueError, match="layout"):
        replay.difference(a, b.float())
