"""Reproduce cache-only SFT startup on CPU, with periodic Python stack dumps."""

import argparse
import faulthandler
import json
import os
import runpy
import sys
import time
from pathlib import Path

from scripts.train.atomic_think import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    # Restrict process affinity as well as library threads: datasets derives its
    # multiprocessing count from available CPUs, not OMP_NUM_THREADS.
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[:2])
    print(f"CPU startup diagnostic; affinity={available[:2]}", flush=True)
    collect.collect(args.root, Path("/output/gate-artifacts"), collect.ARMS[:3], [])
    command = json.loads((args.root / "gate-reference" / "command-node0.json").read_text())
    script = "open_instruct/olmo_core_finetune.py"
    offset = command.index(script)
    sys.argv = command[offset:] + ["--cache_dataset_only"]
    os.environ["EXPECTED_NUMPY_CACHE"] = "062b8a3d20-6068a350"
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    started = time.monotonic()
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        faulthandler.cancel_dump_traceback_later()
    result = {"cache_only_startup_seconds": time.monotonic() - started, "source_command": command}
    Path("/output/startup-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Cache-only startup completed in {result['cache_only_startup_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
