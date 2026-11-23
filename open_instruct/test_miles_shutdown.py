import asyncio
import importlib.util
import json
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct.miles import __main__ as cli
from open_instruct.miles.execution import shutdown


class Exited(BaseException):
    pass


def test_run_bounded_abandons_a_blocked_operation(monkeypatch):
    release = threading.Event()
    dumps = []
    monkeypatch.setattr(shutdown.faulthandler, "dump_traceback", lambda **kw: dumps.append(kw))
    started = time.monotonic()
    try:
        assert shutdown.run_bounded(lambda: release.wait(30), "blocked", timeout=0.2) is False
    finally:
        release.set()
    assert time.monotonic() - started < 5
    assert dumps == [{"all_threads": True}]


def test_run_bounded_logs_operation_errors():
    def fail():
        raise RuntimeError("wandb unavailable")

    assert shutdown.run_bounded(fail, "failing", timeout=5) is True
    assert shutdown.run_bounded(lambda: None, "quick", timeout=5) is True


def load_driver(monkeypatch, finish_tracking):
    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)

    module("ray", is_initialized=lambda: True)
    module("miles")
    module("miles.ray", placement_group=SimpleNamespace(), wiring=SimpleNamespace())
    module("miles.ray.rollout")
    module("miles.ray.rollout.eval_dispatch", EvalDispatcher=Mock())
    module("miles.utils", object_store=SimpleNamespace())
    module("miles.utils.compiler_cache", startup_cache=SimpleNamespace(prepare=Mock(), finish=AsyncMock()))
    module("miles.utils.data", remove_rollout_data_refs=lambda *a: None)
    module("miles.utils.hf_utils.config", HF_EXPORT_COMPLETE_MARKER=".complete")
    module("miles.utils.misc", should_run_periodic_action=lambda *a: False)
    module("miles.utils.tracking_utils")
    module(
        "miles.utils.tracking_utils.tracking",
        define_step_key_metric_group=lambda *a: None,
        finish_tracking=finish_tracking,
        init_tracking=lambda args: None,
        log=lambda *a, **kw: None,
    )
    module(
        "miles.backends.core_utils.rollout",
        prepared_batch=SimpleNamespace(enabled=lambda args: False, disabled_reason=lambda args: "test"),
        errors=SimpleNamespace(),
    )
    module("miles.backends.core_utils.publication.rolling_publication", RollingPublication=Mock())
    spec = importlib.util.spec_from_file_location("test_shutdown_driver", Path(shutdown.__file__).parent / "driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, "stage", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(driver.throughput, "report", lambda *a: {"warnings": []})
    return driver


def test_driver_returns_when_wandb_finish_blocks(monkeypatch):
    release, entered = threading.Event(), threading.Event()

    def blocked_finish():
        entered.set()
        release.wait(30)

    driver = load_driver(monkeypatch, blocked_finish)
    monkeypatch.setattr(driver.shutdown, "TRACKING_CLOSE_TIMEOUT", 0.2)
    monkeypatch.setattr(driver.shutdown.faulthandler, "dump_traceback", lambda **kw: None)
    monkeypatch.setattr(driver.startup, "wait_ready", AsyncMock(side_effect=RuntimeError("ray unavailable")))
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(publication_mode="barrier"), save=None, start_rollout_id=0, fully_async=False
    )
    started = time.monotonic()
    try:
        # The original failure still surfaces after the bounded W&B close.
        with pytest.raises(RuntimeError, match="ray unavailable"):
            asyncio.run(driver.train(args))
    finally:
        release.set()
    assert entered.is_set()
    assert time.monotonic() - started < 5


def write_run(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"miles": {"hf_checkpoint": "model", "global_batch_size": 8}}))
    return path


def test_successful_train_exits_process(tmp_path, monkeypatch):
    exits = []

    def fake_exit(code):
        exits.append(code)
        raise Exited

    monkeypatch.setattr(shutdown.os, "_exit", fake_exit)
    monkeypatch.setattr(sys, "argv", ["miles", "train", str(write_run(tmp_path))])
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: SimpleNamespace(train_config=lambda c: {}))
    with pytest.raises(Exited):
        cli.main()
    assert exits == [0]


def test_failed_train_keeps_its_exception(tmp_path, monkeypatch):
    exit_hook = Mock()
    monkeypatch.setattr(shutdown.os, "_exit", exit_hook)
    monkeypatch.setattr(sys, "argv", ["miles", "train", str(write_run(tmp_path))])

    def broken(config):
        raise RuntimeError("training failed")

    monkeypatch.setattr(cli.importlib, "import_module", lambda name: SimpleNamespace(train_config=broken))
    with pytest.raises(RuntimeError, match="training failed"):
        cli.main()
    exit_hook.assert_not_called()
