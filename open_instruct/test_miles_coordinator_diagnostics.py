"""Verify diagnostic sampling survives a blocked actor loop and preserves errors."""

import faulthandler
import subprocess
import sys
from unittest import mock

import pytest

from open_instruct.miles.infrastructure import coordinator_diagnostics


def test_python_sampler_observes_blocked_event_loop():
    code = """
import asyncio
import time
from open_instruct.miles.infrastructure import coordinator_diagnostics
async def main():
    task = coordinator_diagnostics.start(interval=0.02)
    time.sleep(0.15)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "Python stack sample" in result.stderr
    assert "in main" in result.stderr


def test_phase_preserves_failure():
    error = RuntimeError("original failure")
    with (
        mock.patch.object(coordinator_diagnostics.logger, "exception") as log,
        pytest.raises(RuntimeError) as caught,
        coordinator_diagnostics.phase("coordinator_ack", version=15),
    ):
        raise error
    assert caught.value is error
    log.assert_called_once()


def test_phase_logs_success():
    with (
        mock.patch.object(coordinator_diagnostics.logger, "info") as log,
        coordinator_diagnostics.phase("transfer_weights", rollout_id=1),
    ):
        pass
    assert log.call_count == 2


def test_python_stack_is_written_directly_to_run_directory(tmp_path):
    code = f"""
import asyncio
import time
from open_instruct.miles.infrastructure import coordinator_diagnostics
async def main():
    task = coordinator_diagnostics.start(interval=0.05, directory={str(tmp_path)!r})
    time.sleep(0.16)
    task.cancel()
asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    files = list((tmp_path / "coordinator-diagnostics").glob("stacks-*.log"))
    assert len(files) == 1
    assert "in main" in files[0].read_text()
    assert "Python stack sample" not in result.stderr


def test_sampler_keeps_frames_alive_and_never_uses_native_watchdog(tmp_path):
    output = tmp_path / "sample.log"
    with (
        mock.patch.object(faulthandler, "dump_traceback_later", side_effect=AssertionError("unsafe watchdog")),
        output.open("w") as stream,
    ):
        coordinator_diagnostics.sample_stacks(stream)
    assert "test_sampler_keeps_frames_alive" in output.read_text()
