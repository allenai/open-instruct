"""Read-only NVML sampling for topology qualification; no CUDA context is created."""

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

FIELDS = ("index", "uuid", "utilization.gpu", "utilization.memory", "memory.used", "memory.total")


def sample():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    devices = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != len(FIELDS):
            raise ValueError("Unexpected nvidia-smi response shape")
        record = dict(zip(FIELDS, row, strict=True))
        for key in FIELDS:
            if key != "uuid":
                try:
                    record[key] = float(record[key])
                except ValueError:
                    record[key] = None
        devices.append(record)
    return devices


def triton_artifacts(parent):
    """Observe node-local worker caches without changing compiler dispatch.

    Counts include restored artifacts; changes after startup indicate writes,
    not compiler CPU duration. This does not observe non-Triton compilers.
    """
    workers = []
    for root in sorted(parent.glob("core-triton-*")):
        if root.is_symlink() or not root.is_dir():
            continue
        mtimes = [path.stat().st_mtime for path in (root / "triton").rglob("*.cubin") if path.is_file()]
        workers.append(dict(root=str(root), cubins=len(mtimes), newest_write_unix=max(mtimes, default=None)))
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=10800)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--triton-cache-parent", type=Path, help="Also count worker cubins under this local directory")
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.seconds, args.interval)) or args.seconds <= 0 or args.interval < 1:
        parser.error("duration must be positive and interval at least one second")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + args.seconds
    with args.output.open("a", buffering=1) as stream:
        while time.monotonic() < end:
            record = dict(time_unix=time.time(), host=os.uname().nodename)
            try:
                record["devices"] = sample()
                if args.triton_cache_parent is not None:
                    record["triton_artifacts"] = triton_artifacts(args.triton_cache_parent)
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                record["error"] = str(error)
            stream.write(json.dumps(record) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
