"""Copy completed H077 audit artifacts from WEKA into a Beaker result dataset.

Run only after the selected training jobs have stopped writing. Native training
checkpoints stay on WEKA; this collects diagnostic files, not model weights.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ARMS = (
    "gate-reference",
    "gate-probed",
    "gate-resumed",
    "control",
    "rounding-only",
    "atomic-safe",
    "atomic-historical",
    "control-repeat",
    "gate-reference-repeat",
    "gate-resumed-unprobed",
)


def collect(root, output, arms, steps):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Collection destination must be separate from the source tree")
    selected = [root / "packing-receipt.json"]
    selected.extend(root.glob("dataset-*.json"))
    selected.extend(root.glob("gate-passed-node*.json"))
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        source = root / arm
        if not source.is_dir():
            raise FileNotFoundError(source)
        selected.extend(source.glob("command-node*.json"))
        selected.extend(source.glob("completion-node*.json"))
        selected.extend(source.glob("train-node*.log"))
        audit = source / "atomic-audit"
        selected.extend(audit.glob("*.json"))
        for step in steps:
            selected.extend(audit.glob(f"probe-step{step}-rank*.pt"))
    files = []
    for source in sorted(set(selected)):
        relative = source.relative_to(root)
        if root not in source.resolve().parents:
            raise ValueError(f"Artifact leaves the run tree: {source}")
        before = source.stat()
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Artifact changed during collection: {source}")
        with source.open("rb") as stream:
            original_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        with target.open("rb") as stream:
            copied_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if original_hash != copied_hash:
            raise RuntimeError(f"Artifact copy mismatch: {source}")
        files.append({"path": str(relative), "bytes": after.st_size, "sha256": copied_hash})
    receipt = {
        "source": str(root),
        "arms": list(arms),
        "probe_steps": list(steps),
        "files": files,
        "scope": "Artifact snapshot; completeness and training success must be verified separately",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "collection-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--arms", choices=ARMS, nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="*", default=[])
    args = parser.parse_args()
    receipt = collect(args.root, args.output, args.arms, args.steps)
    print(f"Collected {len(receipt['files'])} artifacts; checkpoints remain on WEKA", flush=True)


if __name__ == "__main__":
    main()
