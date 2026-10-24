"""Replay both full-corpus streams with bounded CPU and memory use."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 8
    manifest_path = Path("/corpus/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rows_per_arm"] == 2253684 and len(manifest["files"]) == 1024
    assert manifest["paired_prepacking_order_identical"]
    output = Path("/work/full-replay")
    output.mkdir(exist_ok=False)

    def run(slot, arm, host):
        affinity = ",".join(map(str, cpus[slot * 2 : slot * 2 + 2]))
        with (output / f"{arm}-h{host}.log").open("w") as log:
            subprocess.run(
                [
                    "taskset",
                    "-c",
                    affinity,
                    sys.executable,
                    "/work/replay_batches.py",
                    "--host",
                    str(host),
                    "--arm",
                    arm,
                    "--output",
                    str(output),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for arm in ("control", "uniform"):
        for first in (0, 4):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run, slot, arm, first + slot) for slot in range(4)]
                for future in futures:
                    future.result()
            print("REPLAY-GROUP-COMPLETE", arm, first, flush=True)
    receipts = {}
    for arm in ("control", "uniform"):
        for host in range(8):
            path = output / arm / f"h{host}.json"
            receipt = json.loads(path.read_text())
            assert [b["step"] for b in receipt["batches"]] == list(range(1, 21))
            receipts[f"{arm}/h{host}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "qualification.json").write_text(
        json.dumps(
            {
                "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "receipt_sha256": receipts,
                "batch_hashes": 320,
                "scope": "Full-corpus ordered batch replay; TPU hashes must still be compared",
            },
            indent=2,
        )
    )
    print("FULL-REPLAYS-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
