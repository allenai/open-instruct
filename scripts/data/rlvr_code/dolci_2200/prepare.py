"""Emit a sparse ground_truth overlay for the reviewed Dolci-Think-RL-7B revision."""

import argparse
import csv
import hashlib
import json
import runpy
import tempfile
from pathlib import Path

from pyarrow import parquet as pq

# Load only the repository-owned static protocol. The package initializer imports the service.
PROTOCOL = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "open_instruct/code_utils/execution_client/protocol.py")
)


def replacement(row, entry):
    (target,) = row["ground_truth"]
    if (
        row["dataset"] != ["code_stdio"]
        or row["id"] != entry["source_id"]
        or hashlib.sha256(target.encode()).hexdigest() != entry["target_sha256"]
    ):
        raise ValueError(f"Target differs from the audited release: {row['custom_id']}")
    if entry["function_name"]:
        return None

    tests = PROTOCOL["decode_tests"](target)
    changed = 0
    for test in tests:
        lines = test["output"]
        if isinstance(lines, list) and all(isinstance(line, str) for line in lines):
            test["output"] = "\n".join(lines)
            changed += 1
    PROTOCOL["validate_tests"]("stdio", tests)
    if len(tests) != int(entry["tests"]) or changed != int(entry["wrapped_outputs"]) or not changed:
        raise ValueError(f"Wrapper count differs from audit: {row['custom_id']}")
    return {
        "custom_id": row["custom_id"],
        "target_sha256": entry["target_sha256"],
        "ground_truth": [json.dumps(tests, separators=(",", ":"))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("output", type=Path, help="local JSONL overlay")
    args = parser.parse_args()
    with Path(__file__).with_name("targets.tsv").open() as f:
        entries = {entry["custom_id"]: entry for entry in csv.DictReader(f, delimiter="\t")}
    seen = set()
    overlay = []
    for index in range(9):
        shard = args.shard_dir / f"train-{index:05}-of-00009.parquet"
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=128, columns=["custom_id", "dataset", "id", "ground_truth"]
        ):
            for row in batch.to_pylist():
                key = row["custom_id"]
                if key not in entries:
                    continue
                if key in seen:
                    raise ValueError(f"Duplicate audited ID: {key}")
                seen.add(key)
                result = replacement(row, entries[key])
                if result is not None:
                    overlay.append(result)
    if seen != entries.keys() or len(seen) != 640 or len(overlay) != 136:
        raise ValueError("Expected all 640 audited IDs and exactly 136 replacements")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        path = Path(temp) / "overlay.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in overlay:
                print(json.dumps(record, separators=(",", ":")), file=f)
        # Publish only a complete overlay, and never overwrite an input or previous result.
        args.output.hardlink_to(path)


if __name__ == "__main__":
    main()
