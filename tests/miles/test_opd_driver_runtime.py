"""Invoke the pinned native drivers through the OPD entrypoint before model startup."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_instruct.miles.distillation import opd_timing, opd_train


@pytest.mark.parametrize("fully_async,timing", [(False, False), (True, False), (True, True)])
def test_opd_entrypoint_supplies_native_disposer_and_cleans_on_failure(monkeypatch, tmp_path, fully_async, timing):
    if timing:
        monkeypatch.setenv(opd_timing.ENV, str(tmp_path))
    else:
        monkeypatch.delenv(opd_timing.ENV, raising=False)
    args = SimpleNamespace(fully_async=fully_async, colocate=False)
    monkeypatch.setattr(opd_train, "parse_args", lambda: args)
    monkeypatch.setattr(opd_train.ray, "init", Mock())
    monkeypatch.setattr(opd_train.ray, "shutdown", Mock())
    monkeypatch.setattr(opd_train, "finish_tracking", Mock())
    load = opd_train.runpy.run_path
    events = []

    async def cleanup():
        events.append("cleanup")

    def startup(args, *, disposer):
        events.append("startup")
        disposer.add(cleanup)
        raise RuntimeError("stop before model startup")

    def native_driver(path):
        namespace = load(path)
        globals = namespace["train"].__globals__
        globals["init_orchestration_script"] = startup
        globals["validate_async_off_policy_correction"] = lambda args: None
        return namespace

    monkeypatch.setattr(opd_train.runpy, "run_path", native_driver)
    with pytest.raises(RuntimeError, match="stop before model startup"):
        opd_train.main()
    assert events == ["startup", "cleanup"]
    opd_train.ray.shutdown.assert_called_once()
    opd_train.finish_tracking.assert_called_once()
    rows = [json.loads(line) for path in tmp_path.glob("events-*.jsonl") for line in path.read_text().splitlines()]
    assert [row["stage"] for row in rows] == (["shutdown"] if timing else [])


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_timed_disposer_preserves_return_and_cleanup_failure(monkeypatch, tmp_path, fail_cleanup):
    monkeypatch.setenv(opd_timing.ENV, str(tmp_path))
    events = []

    async def cleanup():
        events.append("cleanup")
        if fail_cleanup:
            raise RuntimeError("cleanup failed")

    async def train(args, *, disposer):
        disposer.add(cleanup)
        return args

    if fail_cleanup:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            asyncio.run(opd_train.run_training(train, "result"))
    else:
        assert asyncio.run(opd_train.run_training(train, "result")) == "result"
    assert events == ["cleanup"]
    rows = [json.loads(line) for path in tmp_path.glob("events-*.jsonl") for line in path.read_text().splitlines()]
    assert [(row["stage"], row["outcome"]) for row in rows] == [
        ("shutdown", "RuntimeError" if fail_cleanup else "completed")
    ]
