"""Hardlink every committed Core RL checkpoint of the given runs before checkpoint_keep_last prunes it.

Each checkpoint becomes PRESERVE/<run>/u<update>/ = {core-latest.json, core/rollout_<id>, rollout/global_dataset_state_dict_<id>.pt},
a self-contained `miles.load` root (export it to HF with an empty-loop run, num_rollouts = id + 1). Hardlinks cost no space
until the run deletes its own copy. Polls until killed or --hours elapse.

Usage: python preserve_checkpoints.py PRESERVE_ROOT RUN_ROOT [RUN_ROOT ...] [--interval 60 --hours 12]
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def preserve(run, target_root):
    checkpoints = run / "checkpoints"
    done = []
    for source in sorted((checkpoints / "core").glob("rollout_*")):
        if not (source / "complete.json").is_file():
            continue
        rollout_id = int(source.name.split("_")[1])
        final = target_root / run.name / f"u{rollout_id + 1:04d}"
        if final.exists():
            continue
        cursor = checkpoints / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        staging = final.with_name(final.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        try:
            (staging / "core").mkdir(parents=True)
            (staging / "rollout").mkdir()
            subprocess.run(["cp", "-al", str(source), str(staging / "core" / source.name)], check=True)
            os.link(cursor, staging / "rollout" / cursor.name)
            (staging / "core-latest.json").write_text(json.dumps({"rollout_id": rollout_id}) + "\n")
            staging.rename(final)
            done.append(final)
        except (OSError, subprocess.CalledProcessError) as error:  # pruned mid-link: retry next poll
            shutil.rmtree(staging, ignore_errors=True)
            print(f"{time.strftime('%H:%M:%S')} retry {source}: {error}", flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preserve", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    stop = time.time() + args.hours * 3600
    while time.time() < stop:
        for run in args.runs:
            for path in preserve(run, args.preserve):
                print(f"{time.strftime('%H:%M:%S')} preserved {path}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
