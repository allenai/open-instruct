"""Analyze already-launched H127 probes after both monitors report completion.

This only reads local captures and writes diagnostic artifacts. It cannot launch,
restart, stop, or qualify a TPU experiment.
"""

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

from analyze_update_audits import load_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--timeout", type=float, default=5400)
    args = parser.parse_args()
    output = args.run_root / "numerical-investigation"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        receipts = [
            json.loads((args.run_root / f"gradient-{arm}-monitor/receipt.json").read_text()) for arm in ("ep8", "cp8")
        ]
        if any(r.get("failures") or r.get("failed_pods", 0) for r in receipts):
            raise RuntimeError("A diagnostic monitor reported failure; inspect original receipts")
        if not all(
            r.get("passed") and r.get("succeeded_pods") == 32 and r.get("matched_hashes") == 24 for r in receipts
        ):
            time.sleep(15)
            continue
        directories = [args.run_root / f"gradient-{arm}-monitor/logs" for arm in ("ep8", "cp8")]
        records = [load_records(p) for p in directories]
        assert all(set(r) == {1, 2, 3} for r in records), "Missing update audit"
        for arm, arm_records in zip(("ep8", "cp8"), records, strict=True):
            for step, record in arm_records.items():
                path = output / f"gradient-{arm}-step{step}-audit.npz"
                raw = base64.b64decode(record["npz_base64"], validate=True)
                if path.exists():
                    assert path.read_bytes() == raw, f"Refusing to replace different audit: {path}"
                else:
                    path.write_bytes(raw)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("analyze_update_audits.py")),
                *map(str, directories),
                str(output / "gradient-mesh-comparison.json"),
            ],
            check=True,
        )
        print("Diagnostic comparison written; original failed qualification remains unchanged.", flush=True)
        return
    raise TimeoutError("Observation deadline expired; no job was changed or declared terminal")


if __name__ == "__main__":
    main()
