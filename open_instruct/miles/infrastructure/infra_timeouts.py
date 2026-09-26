"""Infrastructure waits for CPU-only launch, judge and evaluation processes."""

import asyncio
import contextlib
import math
import os
import threading
import time
from urllib.parse import urlsplit

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def seconds(value):
    factor = float(os.environ.get("MILES_INFRA_TIMEOUT_MULTIPLIER", "1"))
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("MILES_INFRA_TIMEOUT_MULTIPLIER must be finite and >= 1")
    return None if value is None else value * factor


@contextlib.contextmanager
def watch(operation, *, previous_timeout=None):
    threshold = min(30.0, previous_timeout) if previous_timeout and previous_timeout > 0 else 30.0
    started = time.monotonic()
    stopped = threading.Event()

    def warn():
        delay = threshold
        while not stopped.wait(delay):
            logger.warning(
                "Slow infrastructure operation: operation=%s elapsed_s=%.1f warning_s=%s deadline_s=%s",
                operation,
                time.monotonic() - started,
                threshold,
                seconds(previous_timeout),
            )
            delay = 30.0

    thread = threading.Thread(target=warn, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        elapsed = time.monotonic() - started
        if elapsed >= threshold:
            logger.warning("Slow infrastructure operation finished: operation=%s elapsed_s=%.1f", operation, elapsed)


def request(session, method, url, *, timeout, **kwargs):
    parsed = urlsplit(url)
    operation = f"HTTP {method.upper()} {parsed.hostname}:{parsed.port}{parsed.path}"
    with watch(operation, previous_timeout=timeout):
        return getattr(session, method)(url, timeout=seconds(timeout), **kwargs)


async def wait_for(awaitable, timeout, *, operation):
    deadline = seconds(timeout)
    threshold = min(30.0, timeout) if timeout and timeout > 0 else 30.0
    started = time.monotonic()

    async def warn():
        await asyncio.sleep(threshold)
        while True:
            logger.warning(
                "Slow infrastructure operation: operation=%s elapsed_s=%.1f warning_s=%s deadline_s=%s",
                operation,
                time.monotonic() - started,
                threshold,
                deadline,
            )
            await asyncio.sleep(30.0)

    watchdog = asyncio.create_task(warn())
    try:
        return await asyncio.wait_for(awaitable, timeout=deadline)
    finally:
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog
        elapsed = time.monotonic() - started
        if elapsed >= threshold:
            logger.warning("Slow infrastructure operation finished: operation=%s elapsed_s=%.1f", operation, elapsed)
