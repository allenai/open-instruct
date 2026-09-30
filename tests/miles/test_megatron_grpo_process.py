"""Conversion timeout must release the distributed process group, including children."""

import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest

from open_instruct.miles import megatron_grpo_convert


def test_conversion_success_and_failure(tmp_path):
    with (tmp_path / "log").open("w") as stream:
        megatron_grpo_convert.run_conversion([sys.executable, "-c", "print('done')"], dict(os.environ), stream)
        with pytest.raises(subprocess.CalledProcessError) as exc:
            megatron_grpo_convert.run_conversion(
                [sys.executable, "-c", "raise SystemExit(7)"], dict(os.environ), stream
            )
        assert exc.value.returncode == 7
    assert "done" in (tmp_path / "log").read_text()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Distributed conversion targets POSIX")
def test_conversion_timeout_kills_worker_even_after_parent_exits(tmp_path):
    child_pid = tmp_path / "child.pid"
    survivor = tmp_path / "survived"
    code = (
        "import os,signal,time,pathlib; pid=os.fork(); "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN) if pid==0 else None; "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())) if pid==0 else None; "
        "time.sleep(2); "
        f"pathlib.Path({str(survivor)!r}).write_text('alive') if pid==0 else None; time.sleep(30)"
    )
    try:
        with (tmp_path / "log").open("w") as stream, pytest.raises(subprocess.TimeoutExpired):
            megatron_grpo_convert.run_conversion([sys.executable, "-c", code], dict(os.environ), stream, timeout=0.5)
        assert child_pid.exists(), "Child was never started"
        time.sleep(2)
        assert not survivor.exists(), "Worker survived the terminated parent and wrote after timeout"
    finally:
        if child_pid.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
