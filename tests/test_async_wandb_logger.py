"""Tests for AsyncWandbLogger — wandb publishing that can never block training."""

import logging
import threading
import time
from unittest.mock import patch

from open_instruct.utils import AsyncWandbLogger


class _FakeWandb:
    """Test double for the wandb module: records calls, can wedge on demand."""

    def __init__(self):
        self.calls = []
        self.unwedge = threading.Event()
        self.unwedge.set()  # Healthy by default.
        self.fail_next = False

    def log(self, metrics, step=None):
        self.unwedge.wait()
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic wandb failure")
        self.calls.append((metrics, step))


def test_logs_flow_through_in_order():
    fake = _FakeWandb()
    with patch("open_instruct.utils.wandb", fake):
        async_logger = AsyncWandbLogger()
        for step in range(5):
            async_logger.log({"loss": step}, step=step)
        assert async_logger.flush(timeout_s=5)
    assert [step for _, step in fake.calls] == [0, 1, 2, 3, 4]
    assert fake.calls[3][0] == {"loss": 3}


def test_wedged_wandb_never_blocks_caller_and_drops_when_full(caplog):
    fake = _FakeWandb()
    fake.unwedge.clear()  # Wedge: wandb.log blocks forever.
    with patch("open_instruct.utils.wandb", fake):
        async_logger = AsyncWandbLogger(maxsize=3)
        start = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="open_instruct.utils"):
            # 1 entry wedged in-flight + 3 queued; the rest must drop.
            for step in range(10):
                async_logger.log({"loss": step}, step=step)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"log() blocked for {elapsed:.1f}s with a wedged wandb service"
        assert "queue full" in caplog.text
        assert async_logger._dropped >= 6
        fake.unwedge.set()  # Un-wedge so the daemon thread doesn't leak into other tests.
        async_logger.flush(timeout_s=5)


def test_wandb_exception_does_not_kill_the_thread(caplog):
    fake = _FakeWandb()
    fake.fail_next = True
    with patch("open_instruct.utils.wandb", fake):
        async_logger = AsyncWandbLogger()
        with caplog.at_level(logging.WARNING, logger="open_instruct.utils"):
            async_logger.log({"loss": 1}, step=1)  # This one raises inside the thread.
            async_logger.log({"loss": 2}, step=2)  # This one must still be published.
            assert async_logger.flush(timeout_s=5)
    assert "wandb.log failed" in caplog.text
    assert [step for _, step in fake.calls] == [2]


def test_flush_reports_timeout_when_wedged(caplog):
    fake = _FakeWandb()
    fake.unwedge.clear()
    with patch("open_instruct.utils.wandb", fake):
        async_logger = AsyncWandbLogger()
        async_logger.log({"loss": 1}, step=1)
        with caplog.at_level(logging.WARNING, logger="open_instruct.utils"):
            assert not async_logger.flush(timeout_s=0.3)
        fake.unwedge.set()
    assert "flush timed out" in caplog.text
