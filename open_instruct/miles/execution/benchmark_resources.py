"""Collect benchmark resource observations independently of GPU runtime imports."""

import asyncio
import json
import os
import resource
import tempfile
import time
from contextlib import suppress

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def temporary_storage():
    """Observe the temp filesystem, without scanning files or attributing ownership."""
    path = None
    try:
        path = tempfile.gettempdir()
        usage = os.statvfs(path)
    except OSError as error:
        return {"path": path, "error_type": type(error).__name__, "error": str(error)}
    return {
        "path": path,
        "capacity_bytes": usage.f_blocks * usage.f_frsize,
        "available_bytes": usage.f_bavail * usage.f_frsize,
        "free_bytes": usage.f_bfree * usage.f_frsize,
        "inodes": usage.f_files,
        "available_inodes": usage.f_favail,
    }


async def observe_resources(root):
    """Retain failed samples and retry; cancellation still joins the active child."""
    with (root / "benchmark-resources.jsonl").open("w", buffering=1) as stream:
        while True:
            process = None
            stdout = stderr = b""
            row = {"started_unix": time.time(), "temporary_storage": temporary_storage()}
            try:
                process = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
            except OSError as error:  # Includes the sampling TimeoutError.
                row.update(error_type=type(error).__name__, error=str(error))
                logger.warning("Benchmark GPU resource sample failed (%s): %s", type(error).__name__, error)
            finally:
                if process is not None and process.returncode is None:
                    # The child can exit before the event loop records it.
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
            usage = resource.getrusage(resource.RUSAGE_SELF)
            stream.write(
                json.dumps(
                    {
                        **row,
                        "time_unix": time.time(),
                        "gpu_csv": stdout.decode(),
                        "stderr": stderr.decode(),
                        "returncode": process.returncode if process is not None else None,
                        "consumer_cpu_seconds": usage.ru_utime + usage.ru_stime,
                        "consumer_max_rss_kib": usage.ru_maxrss,
                    }
                )
                + "\n"
            )
            await asyncio.sleep(2)
