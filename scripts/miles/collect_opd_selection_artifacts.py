"""Archive the bounded OPD pair's analysis inputs without checkpoint weights."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def collect(root, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"root": str(root), "arms": {}, "files": []}
    patterns = (
        "*.json",
        "training.log",
        "runtime-tests.log",
        "selection-events.jsonl",
        "eval-capture/*.jsonl",
        "debug/train_data/0_0.pt",
        "debug/train_data/7_0.pt",
        "hf-7/.complete",
        "hf-7/opd-export.json",
        "checkpoints/latest_checkpointed_iteration.txt",
    )
    with tarfile.open(output / "opd-selection-artifacts.tar.gz", "w:gz") as archive:
        for arm in ("drop-8u", "retry-8u"):
            source = root / arm
            manifest["arms"][arm] = {
                "export_complete": (source / "hf-7/.complete").is_file(),
                "initial_eval_exists": (source / "eval-capture/eval-0.jsonl").is_file(),
                "final_eval_exists": (source / "eval-capture/eval-7.jsonl").is_file(),
            }
            paths = sorted({path for pattern in patterns for path in source.glob(pattern)})
            for path in paths:
                if not path.is_file() or path.is_symlink():
                    continue
                relative = str(path.relative_to(root))
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                manifest["files"].append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
                archive.add(path, arcname=relative, recursive=False)
    (output / "capture-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
