"""Snapshot small timing artifacts from live runs without reading model tensors."""

import argparse
import hashlib
import json
import time
from pathlib import Path

PATTERNS = (
    "driver_timing.jsonl",
    "startup_rank*.jsonl",
    "publication.jsonl",
    "pipeline_occupancy.jsonl",
    "rollout_errors.jsonl",
    "pipeline_lifecycle.jsonl",
    "engine_occupancy*.jsonl",
    "gpu_usage_node*.jsonl",
)


def collect(roots, output):
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for root in roots:
        destination = output / root.name
        destination.mkdir(exist_ok=True)
        for directory in (root, root / "checkpoints"):
            for pattern in PATTERNS:
                for source in sorted(directory.glob(pattern)):
                    if not source.is_file():
                        continue
                    target = destination / source.name
                    with source.open("rb") as stream:
                        # Snapshot only the size seen at open; producers may keep appending.
                        size = source.stat().st_size
                        if size > 256 * 1024 * 1024:
                            raise ValueError(f"Timing artifact exceeds snapshot budget: {source}")
                        data = stream.read(size)
                    # A concurrent final JSONL record may be incomplete.
                    end = data.rfind(b"\n") + 1
                    target.write_bytes(data[:end])
                    inventory.append(
                        {
                            "source": str(source),
                            "file": str(target.relative_to(output)),
                            "bytes": end,
                            "sha256": hashlib.sha256(data[:end]).hexdigest(),
                        }
                    )
        complete = []
        for source in (root / "checkpoints").glob("*/complete.json"):
            complete.append({"path": str(source), "data": json.loads(source.read_text())})
        (destination / "complete-checkpoints.json").write_text(json.dumps(complete))
    (output / "inventory.json").write_text(json.dumps({"observed_unix": time.time(), "files": inventory}, indent=2))
    print(json.dumps({"artifacts": len(inventory), "bytes": sum(r["bytes"] for r in inventory)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    collect(args.roots, args.output)


if __name__ == "__main__":
    main()
