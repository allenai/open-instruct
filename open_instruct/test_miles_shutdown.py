import asyncio
import importlib.util
import json
import subprocess
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
    monkeypatch.setattr(shutdown, "_abandoned", [])
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
    assert shutdown.abandoned() == ["blocked"]


def test_run_bounded_logs_operation_errors(monkeypatch):
    monkeypatch.setattr(shutdown, "_abandoned", [])

    def fail():
        raise RuntimeError("wandb unavailable")

    assert shutdown.run_bounded(fail, "failing", timeout=5) is True
    assert shutdown.run_bounded(lambda: None, "quick", timeout=5) is True
    assert shutdown.abandoned() == []


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
    monkeypatch.setattr(driver.shutdown, "_abandoned", [])
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
    # The CLI sees the abandoned close and forces the failure exit.
    assert driver.shutdown.abandoned() == ["tracking close"]


def write_run(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"miles": {"hf_checkpoint": "model", "global_batch_size": 8}}))
    return path


class FakeRay:
    def __init__(self):
        self.calls = []

    def is_initialized(self):
        return True

    def shutdown(self):
        self.calls.append("shutdown")


@pytest.fixture
def exit_hooks(monkeypatch):
    """Record forced exits, Ray shutdowns and watchdogs instead of performing them."""
    hooks = SimpleNamespace(exits=[], watchdogs=[], ray=FakeRay())

    def fake_exit(code):
        hooks.exits.append(code)
        raise Exited

    monkeypatch.setattr(shutdown, "_abandoned", [])
    monkeypatch.setattr(shutdown, "_exit_watchdog_enabled", True)
    monkeypatch.setattr(shutdown.os, "_exit", fake_exit)
    monkeypatch.setattr(shutdown, "arm_exit_watchdog", hooks.watchdogs.append)
    monkeypatch.setitem(sys.modules, "ray", hooks.ray)
    return hooks


def run_cli(tmp_path, monkeypatch, train):
    monkeypatch.setattr(sys, "argv", ["miles", "train", str(write_run(tmp_path))])
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: SimpleNamespace(train_config=train))
    cli.main()


def test_successful_train_exits_normally_with_a_watchdog(tmp_path, monkeypatch, exit_hooks):
    run_cli(tmp_path, monkeypatch, lambda config: {})
    assert exit_hooks.exits == [] and exit_hooks.ray.calls == []
    assert exit_hooks.watchdogs == [0]


def test_successful_train_with_abandoned_close_shuts_down_ray_then_exits(tmp_path, monkeypatch, exit_hooks):
    def train(config):
        shutdown._abandoned.append("tracking close")
        return {}

    with pytest.raises(Exited):
        run_cli(tmp_path, monkeypatch, train)
    assert exit_hooks.ray.calls == ["shutdown"]
    assert exit_hooks.exits == [0]


def test_failed_train_keeps_its_exception(tmp_path, monkeypatch, exit_hooks):
    def broken(config):
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        run_cli(tmp_path, monkeypatch, broken)
    assert exit_hooks.exits == []
    assert exit_hooks.watchdogs == [1]


def test_failed_train_with_abandoned_close_exits_nonzero(tmp_path, monkeypatch, exit_hooks, capsys):
    def broken(config):
        shutdown._abandoned.append("tracking close")
        raise RuntimeError("training failed")

    with pytest.raises(Exited):
        run_cli(tmp_path, monkeypatch, broken)
    assert exit_hooks.ray.calls == ["shutdown"]
    assert exit_hooks.exits == [1]
    assert "training failed" in capsys.readouterr().err


def test_watchdog_is_off_unless_the_entry_point_enables_it(monkeypatch):
    armed = []
    monkeypatch.setattr(shutdown, "_abandoned", [])
    monkeypatch.setattr(shutdown, "_exit_watchdog_enabled", False)
    monkeypatch.setattr(shutdown, "arm_exit_watchdog", armed.append)
    shutdown.finish_process(0)
    assert armed == []


def test_watchdog_forces_exit_with_the_recorded_code(monkeypatch):
    exited = threading.Event()
    codes = []

    def fake_exit(code):
        codes.append(code)
        exited.set()

    monkeypatch.setattr(shutdown.os, "_exit", fake_exit)
    monkeypatch.setattr(shutdown.os, "write", lambda fd, data: len(data))
    monkeypatch.setattr(shutdown.faulthandler, "dump_traceback", lambda **kw: None)
    timer = shutdown.arm_exit_watchdog(3, timeout=0.05)
    assert timer.daemon
    assert exited.wait(5)
    assert codes == [3]


def test_watchdog_ends_a_real_process_held_by_a_non_daemon_thread():
    script = """
import threading
from open_instruct.miles.execution import shutdown
shutdown.EXIT_WATCHDOG_SECONDS = 0.5
shutdown.enable_exit_watchdog()
threading.Thread(target=threading.Event().wait, daemon=False).start()
shutdown.finish_process(4)
"""
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parent.parent, timeout=60)
    assert result.returncode == 4
    assert time.monotonic() - started < 30


HELD_LOG_LOCK = """
import logging, sys, threading
from open_instruct.miles.execution import shutdown
handler = logging.StreamHandler(sys.stderr)
logging.root.addHandler(handler)
shutdown.logger.addHandler(handler)
shutdown.LOCKED_IO_TIMEOUT = 0.2
held = threading.Event()

def hold():
    handler.acquire()  # A stuck writer keeps the handler lock forever.
    held.set()
    threading.Event().wait()

threading.Thread(target=hold, daemon=False).start()
held.wait()
"""


@pytest.mark.parametrize(
    ("scenario", "code"),
    [
        # The watchdog fires while interpreter shutdown waits for the stuck thread.
        ("shutdown.EXIT_WATCHDOG_SECONDS = 0.3\nshutdown.enable_exit_watchdog()\nshutdown.finish_process(7)", 7),
        # An abandoned tracking close forces the exit at once.
        ("shutdown._abandoned.append('tracking close')\nshutdown.finish_process(5)", 5),
    ],
)
def test_forced_exit_survives_a_held_logging_lock(scenario, code):
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", HELD_LOG_LOCK + scenario], cwd=Path(__file__).parent.parent, timeout=60
    )
    assert result.returncode == code
    assert time.monotonic() - started < 30


FULL_STDERR = """
import fcntl, os, threading
from open_instruct.miles.execution import shutdown
shutdown.LOCKED_IO_TIMEOUT = 0.2
read, write = os.pipe()
fcntl.fcntl(write, fcntl.F_SETFL, os.O_NONBLOCK)
try:
    while True:
        os.write(write, b"x" * 65536)  # Fill the pipe; nobody reads it.
except BlockingIOError:
    pass
fcntl.fcntl(write, fcntl.F_SETFL, 0)
os.dup2(write, 2)
threading.Thread(target=threading.Event().wait, daemon=False).start()
"""


@pytest.mark.parametrize(
    ("scenario", "code"),
    [
        ("shutdown.EXIT_WATCHDOG_SECONDS = 0.3\nshutdown.enable_exit_watchdog()\nshutdown.finish_process(7)", 7),
        (
            "shutdown.run_bounded(threading.Event().wait, 'tracking close', timeout=0.1)\n"
            "try:\n    raise RuntimeError('training failed')\nexcept RuntimeError:\n"
            "    shutdown.print_exception()\n    shutdown.finish_process(1)",
            1,
        ),
    ],
)
def test_forced_exit_survives_a_full_stderr_pipe(scenario, code):
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", FULL_STDERR + scenario], cwd=Path(__file__).parent.parent, timeout=60
    )
    assert result.returncode == code
    assert time.monotonic() - started < 30
