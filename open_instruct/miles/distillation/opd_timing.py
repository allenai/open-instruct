"""Opt-in pipeline intervals. Overlapping request-seconds are not GPU busy time."""

import asyncio
import inspect
import json
import os
import socket
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles.distillation import opd_timing_driver

logger = logger_utils.setup_logger(__name__)
ENV = "OI_OPD_TIMING"
_WARNED = False


def enabled():
    return bool(os.environ.get(ENV))


def identity(sample):
    return dict(
        attempt_id=(getattr(sample, "metadata", None) or {}).get("oi_opd_selection_attempt"),
        sample_index=getattr(sample, "index", None),
        group_index=getattr(sample, "group_index", None),
    )


def write(row):
    global _WARNED
    if not enabled():
        return
    try:
        root = Path(os.environ[ENV])
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"events-{socket.gethostname()}-{os.getpid()}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(dict(schema_version=1, pid=os.getpid(), **row), allow_nan=False) + "\n")
    except Exception:
        if not _WARNED:
            logger.exception("OPD timing unavailable; continuing without this diagnostic")
            _WARNED = True


@contextmanager
def stage(name, **context):
    if not enabled():
        yield {}
        return
    start, wall = time.monotonic(), time.time()
    row = dict(stage=name, started_unix=wall, **context)
    try:
        yield row
    except BaseException as exc:
        row["outcome"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else type(exc).__name__
        raise
    else:
        row["outcome"] = "completed"
    finally:
        row["seconds"] = time.monotonic() - start
        write(row)


@asynccontextmanager
async def teacher_slot(semaphore, sample):
    with stage("teacher_admission_wait", **identity(sample)):
        await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


async def awaited(awaitable, name, rollout_id):
    with stage(name, rollout_id=rollout_id):
        return await awaitable


def start_group(group):
    if not enabled():
        return
    now = time.monotonic()
    attempt = uuid.uuid4().hex
    for sample in group:
        sample.metadata["oi_opd_timing_submitted"] = now
        # A timing-only run still needs joinable fresh attempt identities.
        if not os.environ.get("OI_OPD_SELECTION_AUDIT"):
            sample.metadata["oi_opd_selection_attempt"] = attempt


def student_admitted(sample, evaluation):
    now = time.monotonic()
    start = sample.metadata.pop("oi_opd_timing_submitted", None)
    if start is not None:
        write(
            dict(
                stage="student_admission_wait",
                started_unix=time.time() - (now - start),
                seconds=now - start,
                outcome="completed",
                evaluation=evaluation,
                **identity(sample),
            )
        )


def instrument_driver(train):
    """Wrap awaits in the pinned driver; no new tasks, barriers or scheduling decisions."""
    namespace = train.__globals__
    namespace["_opd_timed_await"] = awaited
    tree = opd_timing_driver.transform(inspect.getsource(train))
    exec(compile(tree, inspect.getsourcefile(train), "exec"), namespace)
    return namespace["train"]
