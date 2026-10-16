"""Continue matched H077 SFT to the original fixed 11768-update horizon."""

import argparse
import os
import re
from pathlib import Path

from scripts.train.atomic_think import run, run_cached

from open_instruct import atomic_think

HORIZON = 11768
SAMPLES = 188544
CHECKPOINTS = (*range(1024, HORIZON, 1024), HORIZON)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "atomic-safe"))
    parser.add_argument("root", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("resume", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    match = re.fullmatch(r"step(\d+)", args.resume.name)
    if match is None or (start := int(match[1])) not in (512, *CHECKPOINTS[:-1]):
        parser.error("Resume must be an audited step512 or intermediate continuation checkpoint")
    if args.root.resolve() in (args.data_root.resolve(), args.resume.parent.parent.resolve()):
        parser.error("Use a fresh output root and preserve the source checkpoint tree")
    if args.resume.parent.name != args.arm:
        parser.error("Resume checkpoint belongs to a different arm")
    local = Path("/tmp") / f"h077-full-cache-{os.environ['BEAKER_JOB_ID']}"
    receipt = run_cached.prepare(
        args.root,
        args.arm,
        args.data_root,
        args.cache,
        local,
        expected_samples=SAMPLES,
        manifest_sha256=args.manifest_sha256,
    )
    rank = os.environ["BEAKER_REPLICA_RANK"]
    atomic_think.write_json(args.root / args.arm / f"cache-staging-node{rank}.json", receipt)
    checkpoints = [step for step in CHECKPOINTS if step > start]
    batch_steps = sorted({*range(start + 1, min(start + 13, HORIZON + 1)), *checkpoints, HORIZON - 2, HORIZON - 1})
    run.train(
        args.root,
        args.arm,
        args.arm,
        HORIZON,
        [start, *checkpoints],
        1024,
        resume=args.resume,
        state_steps=checkpoints,
        offline=True,
        packed_cache=local,
        data_root=args.data_root,
        verify_resume=True,
        batch_steps=batch_steps,
    )


if __name__ == "__main__":
    main()
