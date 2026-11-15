"""Opt-in, bounded-frequency coordinator diagnostics for H235 failure reproduction."""

import asyncio
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


async def heartbeat(interval=30):
    previous = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        now = time.monotonic()
        logger.info(
            "Coordinator heartbeat pid=%s unix=%.3f loop_delay_s=%.3f",
            os.getpid(),
            time.time(),
            now - previous - interval,
        )
        previous = now


def sample_stacks(output):
    # Take strong frame references while holding the GIL. The native faulthandler
    # watchdog walks frames without the GIL and can segfault on this Python 3.12.3
    # runtime (CPython gh-123967). Sampling cannot observe a thread that holds the
    # GIL indefinitely; the independent process/memory watcher covers that gap.
    frames = sys._current_frames()
    try:
        output.write(f"Python stack sample unix={time.time():.3f}\n")
        for identity, frame in list(frames.items())[:100]:
            output.write(f"Thread {identity}:\n")
            output.writelines(traceback.format_stack(frame, limit=50))
        output.flush()
    finally:
        frames.clear()


def _watch(stopped, interval, path):
    output = path.open("a") if path is not None else sys.stderr
    try:
        while not stopped.wait(interval):
            sample_stacks(output)
    finally:
        if path is not None:
            output.close()


async def _heartbeat_and_stop(stopped):
    try:
        await heartbeat()
    finally:
        stopped.set()


def start(interval=60, directory=None):
    path = None
    if directory is not None:
        directory = Path(directory) / "coordinator-diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"stacks-{os.getpid()}.log"
    stopped = threading.Event()
    thread = threading.Thread(target=_watch, args=(stopped, interval, path), daemon=True)
    thread.start()
    logger.info(
        "Coordinator Python stack sampling enabled pid=%s unix=%.3f interval_s=%s", os.getpid(), time.time(), interval
    )
    task = asyncio.create_task(_heartbeat_and_stop(stopped))
    task.add_done_callback(lambda _: stopped.set())
    return task


@contextmanager
def phase(name, **fields):
    started = time.monotonic()
    logger.info("Publication phase begin: %s %s", name, fields)
    try:
        yield
    except BaseException:
        logger.exception("Publication phase failed: %s elapsed_s=%.3f %s", name, time.monotonic() - started, fields)
        raise
    else:
        logger.info("Publication phase end: %s elapsed_s=%.3f %s", name, time.monotonic() - started, fields)
