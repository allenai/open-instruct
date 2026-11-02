"""Join independent startup work and wait for attempt-scoped readiness barriers."""

import asyncio
import json
import os
import time
from pathlib import Path

from open_instruct.miles.infrastructure import infra_timeouts

REPORT_DIR_ENV = "OI_MILES_STARTUP_REPORT_DIR"


async def join(*operations):
    """On failure/cancellation, join every sibling before resource cleanup starts."""
    tasks = [asyncio.create_task(operation) for operation in operations]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def wait_ready(name):
    """Manual launches have no cluster barrier; managed launches fail closed."""
    location = os.environ.get(REPORT_DIR_ENV)
    if not location:
        return
    root = Path(location)
    state = json.loads((root / "startup.json").read_text())
    remaining = state["deadline_unix"] - time.time()
    failures = list(root.glob("failed-*.json"))
    if failures:
        raise RuntimeError(f"Cluster startup failed: {failures[0].read_text()}")
    if (root / f"{name}-ready.json").is_file():
        return

    async def wait():
        while True:
            failures = list(root.glob("failed-*.json"))
            if failures:
                raise RuntimeError(f"Cluster startup failed: {failures[0].read_text()}")
            if (root / f"{name}-ready.json").is_file():
                return
            await asyncio.sleep(0.2)

    # The cluster already applies the infrastructure multiplier to this deadline.
    with infra_timeouts.watch(f"startup barrier={name}"):
        try:
            await asyncio.wait_for(wait(), timeout=max(0.0, remaining))
        except TimeoutError as error:
            raise TimeoutError(f"Startup barrier {name} did not finish before the cluster deadline") from error
