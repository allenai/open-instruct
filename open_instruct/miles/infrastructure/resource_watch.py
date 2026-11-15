"""Preserve process/memory evidence independently of a stalled rollout actor."""

import contextlib
import json
import os
import signal
import time
from pathlib import Path

_LIMIT = 2 * 1024 * 1024


def memory_snapshot(proc=Path("/proc"), cgroup=Path("/sys/fs/cgroup")):
    processes = []
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().split(b"\0", 1)[0]
            if b"RolloutExecutor" not in command:
                continue
            fields = {}
            for line in (path / "status").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmSize", "VmHWM", "Threads"}:
                    fields[key] = value.strip()
            processes.append({"pid": int(path.name), **fields})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    memory = {}
    for name in ["memory.current", "memory.max", "memory.peak", "memory.events"]:
        with contextlib.suppress(FileNotFoundError, PermissionError):
            memory[name] = (cgroup / name).read_text().strip()
    return {"unix": time.time(), "coordinators": processes, "cgroup": memory}


def preserve_logs(output, pids, logs=Path("/tmp/ray/session_latest/logs")):
    target = output / "ray-worker-diagnostics"
    target.mkdir(exist_ok=True)
    candidates = list(logs.glob("raylet.*"))
    for pid in pids:
        candidates.extend(logs.glob(f"worker*-{pid}.*"))
        candidates.extend(logs.glob(f"python-core-worker*_{pid}.log"))
    for source in sorted(set(candidates))[:16]:
        if not source.is_file():
            continue
        try:
            with source.open("rb") as stream:
                stream.seek(max(0, source.stat().st_size - _LIMIT))
                (target / source.name).write_bytes(stream.read(_LIMIT))
        except (FileNotFoundError, PermissionError):
            continue


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    seen = set()
    try:
        with (output / "coordinator-memory.jsonl").open("a", buffering=1) as stream:
            while not stopped:
                record = memory_snapshot()
                current = {p["pid"] for p in record["coordinators"]}
                stream.write(json.dumps(record) + "\n")
                if seen - current:
                    preserve_logs(output, seen)
                seen.update(current)
                for _ in range(10):
                    if stopped:
                        break
                    time.sleep(1)
    finally:
        preserve_logs(output, seen)


if __name__ == "__main__":
    run(Path(os.environ["H235_RESOURCE_DIAGNOSTICS"]))
