"""Run one warmed SFT treatment or its precision/resume qualification."""

import argparse
import json
import os
from pathlib import Path

from scripts.train.atomic_think import run, run_cached

from open_instruct import atomic_think

BASE_ROOT = Path("/weka/oe-training-default/robertb/atomic-think-h077")
CACHE_HASHES = {
    "atomic-safe": "37150f9cfd71928ae19eb2d29e92548bfcd508258a571e348bb36c679183beca",
    "control": "b812522e98114e7c5c76dbe62d09bef483eb9fcf9de833ba09c38e3ea80e692e",
}


def launch(root, arm, manifest, manifest_sha, steps, resume=None):
    data_root = BASE_ROOT / "20261015d"
    cache = data_root / ("sample-cache-atomic-r2" if arm == "atomic-safe" else "sample-cache-original")
    local = Path("/tmp") / f"warmed-samples-{os.environ['BEAKER_JOB_ID']}"
    receipt = run_cached.prepare(root, arm, data_root, cache, local, manifest_sha256=CACHE_HASHES[arm])
    rank = os.environ["BEAKER_REPLICA_RANK"]
    atomic_think.write_json(root / arm / f"cache-staging-node{rank}.json", receipt)
    run.train(
        root,
        arm,
        arm,
        steps,
        [steps],
        steps,
        resume=resume,
        state_steps=[steps],
        offline=True,
        packed_cache=local,
        data_root=data_root,
        verify_resume=resume is not None,
        batch_steps=sorted({1, 2, 3, 4, 10, 100, steps}),
        warmup_manifest=manifest,
        warmup_manifest_sha256=manifest_sha,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "atomic-safe"))
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("manifest_sha256")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    if not args.qualify:
        launch(args.root, args.arm, args.manifest, args.manifest_sha256, 512)
        return
    launch(args.root / "fresh", args.arm, args.manifest, args.manifest_sha256, 2)
    launch(
        args.root / "resumed",
        args.arm,
        args.manifest,
        args.manifest_sha256,
        4,
        resume=args.root / "fresh" / args.arm / "step2",
    )
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    for worker in range(rank * 8, rank * 8 + 8):
        source = args.root / "fresh" / args.arm / "atomic-audit"
        destination = args.root / "resumed" / args.arm / "atomic-audit"
        receipt = json.loads((source / f"warmup-handoff-rank{worker}.json").read_text())
        if receipt["manifest_sha256"] != args.manifest_sha256:
            raise ValueError("Wrong initialization identity")
        if (destination / f"warmup-handoff-rank{worker}.json").exists():
            raise ValueError("Resume reinjected warmup")
        restored = json.loads((destination / f"restore-step2-rank{worker}.json").read_text())
        if not restored["trainer_state_including_rng_exact"] or not restored["optimizer_auxiliary_state_exact"]:
            raise ValueError("Resume qualification failed")
    atomic_think.write_json(
        args.root / f"qualification-node{rank}.json",
        {
            "manifest_sha256": args.manifest_sha256,
            "checks": [
                "FP32 handoff exact",
                "unrelated optimizer states unchanged",
                "native 16-rank resume exact",
                "no reinjection",
            ],
        },
    )


if __name__ == "__main__":
    main()
