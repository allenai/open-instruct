"""Opt-in, bounded-frequency coordinator diagnostics for H235 failure reproduction."""

import asyncio
import faulthandler
import os
import time
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


_STACK_FILES = []


def start(interval=60, directory=None):
    # CPython's native watchdog can dump every Python thread even when Python
    # logging, the actor loop, or a Python watchdog cannot acquire the GIL.
    # stderr is retained in the Beaker run log. No response payloads are printed.
    output = None
    if directory is not None:
        path = Path(directory) / "coordinator-diagnostics"
        path.mkdir(parents=True, exist_ok=True)
        output = (path / f"stacks-{os.getpid()}.log").open("a")
        _STACK_FILES.append(output)
    faulthandler.enable(file=output, all_threads=True)
    faulthandler.dump_traceback_later(interval, repeat=True, file=output)
    logger.info(
        "Coordinator stack sampling enabled pid=%s unix=%.3f interval_s=%s", os.getpid(), time.time(), interval
    )
    return asyncio.create_task(heartbeat())


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
