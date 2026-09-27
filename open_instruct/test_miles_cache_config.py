"""CPU-only cache configuration and startup timing contracts."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.execution.timing import startup_stage


def test_startup_timer_records_success_and_failure(tmp_path):
    args = SimpleNamespace(save=str(tmp_path), rank=2)
    device = mock.Mock()
    with startup_stage(args, "build", device=device):
        pass
    with pytest.raises(ValueError), startup_stage(args, "restore"):
        raise ValueError("injected")
    device.synchronize.assert_called_once()
    rows = [json.loads(s) for s in (tmp_path / "startup_rank2.jsonl").read_text().splitlines()]
    assert [r["passed"] for r in rows] == [True, False]
    assert all(r["seconds"] >= 0 for r in rows)


def test_cache_controls_validate_types():
    for key in ("compiler_cache", "compiler_cache_restore", "compiler_cache_diagnostics"):
        with pytest.raises(ValueError, match=key):
            CoreConfig(**{key: "false"})


def test_cache_is_opt_out():
    assert CoreConfig().compiler_cache is True
    assert CoreConfig(compiler_cache=False).compiler_cache is False


@pytest.mark.parametrize(
    "root", ("relative/cache", "/weka", "/weka/oe-training-default/cache", "/weka/tmp-0d/cache", "", 42)
)
def test_invalid_cache_root_rejected_before_launch(root):
    with pytest.raises(ValueError):
        CoreConfig(compiler_cache_root=root)
