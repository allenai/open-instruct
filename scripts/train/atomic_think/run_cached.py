"""Run one fresh H077 arm with verified packed samples staged on each node."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from scripts.train.atomic_think import run

from open_instruct import atomic_think, packed_sample_cache


def prepare(root, arm, data_root, cache, local, *, expected_samples=8448, manifest_sha256=None):
    output = root / arm
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("step*")) or list(output.glob("completion-node*.json")):
        raise FileExistsError("Fresh cached run refuses existing checkpoints or completion records")
    atomic = arm.startswith("atomic-")
    kind = "atomic" if atomic else "original"
    manifest_bytes = (cache / "manifest.json").read_bytes()
    if expected_samples != 8448 and not manifest_sha256:
        raise ValueError("Longer prefixes require a registered manifest hash")
    if manifest_sha256 is not None and hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("Cache differs from registered manifest hash")
    manifest = json.loads(manifest_bytes)
    if (
        manifest["provenance"]["kind"] != kind
        or manifest["provenance"]["epoch"] != 1
        or manifest["provenance"]["data_loader_seed"] != 34521
        or manifest["provenance"]["global_batch_tokens"] != 1048576
        or len(manifest["rows"]) != expected_samples
    ):
        raise ValueError("Sample cache differs from registered H077 training prefix")
    source = (data_root / ("atomic-safe" if atomic else "control") / "dataset-common").resolve(strict=True)
    target = output / "dataset-common"
    try:
        target.symlink_to(source, target_is_directory=True)
    except FileExistsError:
        if not target.is_symlink() or target.resolve() != source:
            raise ValueError("Existing packing differs from the validated source") from None
    # Preserve local copies of both dataset manifests for later provenance collection.
    for name in ("dataset-original.json", "dataset-atomic.json"):
        content = (data_root / name).read_bytes()
        destination = root / name
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Existing dataset manifest differs: {destination}")
        temporary = root / f".{name}.{os.environ.get('BEAKER_JOB_ID', 'local')}.{os.getpid()}"
        temporary.write_bytes(content)
        temporary.replace(destination)
    started = time.monotonic()
    receipt = packed_sample_cache.stage_cache(cache, local)
    receipt["seconds"] = time.monotonic() - started
    receipt["kind"] = kind
    receipt["dataset_manifest_sha256"] = hashlib.sha256((data_root / f"dataset-{kind}.json").read_bytes()).hexdigest()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=run.ARMS)
    parser.add_argument("root", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    if args.root.resolve() == args.data_root.resolve():
        parser.error("Use a fresh output root; preserve the earlier incomplete control")
    local = Path("/tmp") / f"h077-sample-cache-{os.environ['BEAKER_JOB_ID']}"
    receipt = prepare(args.root, args.arm, args.data_root, args.cache, local)
    rank = os.environ["BEAKER_REPLICA_RANK"]
    atomic_think.write_json(args.root / args.arm / f"cache-staging-node{rank}.json", receipt)
    mode = "control" if args.arm == "control-repeat" else args.arm
    run.train(
        args.root,
        args.arm,
        mode,
        512,
        [0, 1, 2, 10, 100, 353, 512],
        512,
        state_steps=[256, 512],
        offline=True,
        packed_cache=local,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
