"""Opt-in durable evidence for historical OI response-work qualification.

Selected masks and engine counters remain observations, not proof that an
optimizer accepted an update or that a CUDA graph actually executed.
"""

import json
import os
import re
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path

_WRITE_LOCK = threading.Lock()


def engine_counters(engine):
    """Read counters and the native last-step flag without advancing the engine.

    DeepSpeed global_steps advances at a boundary even on overflow. Its public
    was_step_applied query reports the engine's decision, not measured tensor
    movement. Missing queries remain unknown for older installed engines.
    """
    optimizer = getattr(engine, "optimizer", None)
    query = getattr(engine, "was_step_applied", None)
    applied = query() if callable(query) else None
    if applied is not None and type(applied) is not bool:
        raise TypeError("Native was_step_applied must return bool or unknown")
    values = {
        "native_step_applied": applied,
        "global_steps": getattr(engine, "global_steps", None),
        "skipped_steps": getattr(engine, "skipped_steps", None),
        "optimizer_overflow": getattr(optimizer, "overflow", None),
    }
    return {key: value if type(value) in (int, bool) else None for key, value in values.items()}


def record(output_dir, event, payload, *, rank=None):
    """Flush each record to a per-process/rank JSONL file independent of Ray logs.

    The run output is the durable source. An optional explicit
    OI_PACKING_AUDIT_RESULT_DIR mirrors bounded records to the Beaker result
    directory immediately, including when training later fails.
    """
    if os.environ.get("OI_PACKING_AUDIT", "0") != "1":
        return None
    if not re.fullmatch(r"[a-z][a-z0-9-]*", event):
        raise ValueError("Audit event must be a safe filename component")
    if rank is not None and (type(rank) is not int or rank < 0):
        raise ValueError("Audit rank must be nonnegative")
    reserved = {"schema_version", "event", "observed_utc", "process_id", "hostname", "beaker_job_id"}
    if not isinstance(payload, dict) or reserved & set(payload):
        raise ValueError("Audit payload must not replace provenance fields")
    if "rank" in payload and payload["rank"] != rank:
        raise ValueError("Audit payload rank differs from its writer")
    host = socket.gethostname()
    process_id = os.getpid()
    data = payload | {
        "schema_version": 1,
        "event": event,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": process_id,
        "hostname": host,
        "beaker_job_id": os.environ.get("BEAKER_JOB_ID"),
        "rank": rank,
    }
    line = json.dumps(data, sort_keys=True, allow_nan=False) + "\n"
    safe_host = re.sub(r"[^A-Za-z0-9_.-]", "_", host)
    owner = "preparation" if rank is None else f"rank-{rank}"
    name = f"{event}-{owner}-{safe_host}-{process_id}.jsonl"
    directories = [Path(output_dir) / "response-work-audit"]
    mirror = os.environ.get("OI_PACKING_AUDIT_RESULT_DIR")
    if mirror:
        directory = Path(mirror)
        if not directory.is_absolute():
            raise ValueError("Audit result mirror must be an absolute directory")
        if directory.resolve() != directories[0].resolve():
            directories.append(directory)
    with _WRITE_LOCK:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / name).open("a") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
    return data
