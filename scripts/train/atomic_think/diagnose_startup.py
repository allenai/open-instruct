"""Reproduce cache-only SFT startup on CPU, with periodic Python stack dumps."""

import argparse
import faulthandler
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

from scripts.train.atomic_think import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prepare-root", type=Path)
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
    base_command = command[offset:] + ["--cache_dataset_only"]
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    started = time.monotonic()
    try:
        if args.prepare_root:
            args.prepare_root.mkdir(parents=True, exist_ok=False)
        variants = ("original", "atomic") if args.prepare_root else ("original",)
        for kind in variants:
            os.environ["EXPECTED_NUMPY_CACHE"] = "062b8a3d20-6068a350" if kind == "original" else "3f12323c3b-6068a350"
            variant = (
                []
                if kind == "original"
                else ["--atomic_audit_mode", "atomic-safe", "--reserved_slot_tokens", "<think>", "</think>"]
            )
            manifest = args.prepare_root / f"dataset-{kind}.json" if args.prepare_root else None
            sys.argv = base_command + variant
            if manifest:
                sys.argv += ["--write_sft_dataset_manifest", str(manifest)]
            runpy.run_path(script, run_name="__main__")
            if manifest:
                # Verify the cache-only read path too; this must not load raw Arrow data.
                sys.argv = base_command + variant + ["--sft_dataset_manifest", str(manifest)]
                runpy.run_path(script, run_name="__main__")
                shutil.copyfile(manifest, Path("/output") / manifest.name)
        if args.prepare_root:
            for arm in collect.ARMS:
                kind = "atomic" if arm.startswith("atomic-") else "original"
                source = args.root / f"packing-{kind}" / "dataset-common"
                if not source.is_dir():
                    raise FileNotFoundError(source)
                destination = args.prepare_root / arm
                destination.mkdir()
                (destination / "dataset-common").symlink_to(source, target_is_directory=True)
            shutil.copyfile(args.root / "packing-receipt.json", args.prepare_root / "packing-receipt.json")
    finally:
        faulthandler.cancel_dump_traceback_later()
    result = {
        "cache_only_startup_seconds": time.monotonic() - started,
        "source_command": command,
        "prepared_root": str(args.prepare_root) if args.prepare_root else None,
    }
    Path("/output/startup-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Cache-only startup completed in {result['cache_only_startup_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
