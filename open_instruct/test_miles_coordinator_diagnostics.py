"""Verify diagnostic sampling survives a blocked actor loop and preserves errors."""

import subprocess
import sys
from unittest import mock

import pytest

from open_instruct.miles.infrastructure import coordinator_diagnostics


@pytest.mark.parametrize("block", ["time.sleep(0.18)", "ctypes.PyDLL(None).usleep(180000)"])
def test_native_stack_sampler_survives_blocked_loop(block):
    code = """
import ctypes
import asyncio
import faulthandler
import time
from open_instruct.miles.infrastructure import coordinator_diagnostics
async def main():
    task = coordinator_diagnostics.start(interval=0.05)
    BLOCK
    faulthandler.cancel_dump_traceback_later()
    task.cancel()
asyncio.run(main())
""".replace("BLOCK", block)
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "Timeout" in result.stderr
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


def test_native_stack_is_written_directly_to_run_directory(tmp_path):
    code = f"""
import asyncio
import faulthandler
import time
from open_instruct.miles.infrastructure import coordinator_diagnostics
async def main():
    task = coordinator_diagnostics.start(interval=0.05, directory={str(tmp_path)!r})
    time.sleep(0.16)
    faulthandler.cancel_dump_traceback_later()
    task.cancel()
asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    files = list((tmp_path / "coordinator-diagnostics").glob("stacks-*.log"))
    assert len(files) == 1
    assert "in main" in files[0].read_text()
    assert "Timeout" not in result.stderr
