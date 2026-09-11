"""Run both native scorer/serving comparisons on CPU with retained WEKA traces."""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from scripts.miles import launch_update_zero_compare


def specification(image, core_root, megatron_root, *, wait_seconds=0, backend="both"):
    document = launch_update_zero_compare.specification(image, core_root, megatron_root, evidence_only=True)
    files = {"__init__.py": b""}
    for name in ("compare_trainer_routes.py", "compare_update_zero.py", "update_zero_capture.py"):
        files[name] = Path(__file__).with_name(name).read_bytes()
    lines = [
        "set -euo pipefail",
        "mkdir -p /tmp/trainer-analysis/scripts/miles /output",
        "touch /tmp/trainer-analysis/scripts/__init__.py",
        "cd /tmp/trainer-analysis",
        "export CUDA_VISIBLE_DEVICES=''",
        "export OMP_NUM_THREADS=1",
    ]
    for name, data in files.items():
        encoded = base64.b64encode(data).decode()
        lines.append(f"printf %s {shlex.quote(encoded)} | base64 -d > /tmp/trainer-analysis/scripts/miles/{name}")
    provenance = {
        "image": image,
        "core_root": core_root,
        "megatron_root": megatron_root,
        "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    encoded = base64.b64encode(json.dumps(provenance).encode()).decode()
    lines.append(f"printf %s {shlex.quote(encoded)} | base64 -d > /output/provenance.json")
    participants = (("olmo_core", core_root), ("megatron", megatron_root))
    for name, root in participants:
        if backend != "both" and name != backend:
            continue
        lines.append(
            shlex.join(
                [
                    "python",
                    "-m",
                    "scripts.miles.compare_trainer_routes",
                    root,
                    "--backend",
                    name,
                    "--output",
                    f"/output/{name}.json",
                    "--wait-seconds",
                    str(wait_seconds),
                ]
            )
        )
    task = document["tasks"][0]
    task["name"] = "compare-native-scorer-routes"
    task["arguments"] = ["\n".join(lines) + "\n"]
    task["resources"]["cpuCount"] = 2
    if wait_seconds:
        task["context"]["minRuntime"] = "1h"
        task["timeout"] = f"{2 * wait_seconds + 1800}s"
    document["description"] = (
        "CPU verification of actual Core/Megatron scorer routes against same-token serving traces"
    )
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--core-root", required=True)
    parser.add_argument("--megatron-root", required=True)
    parser.add_argument("--backend", choices=("both", "olmo_core", "megatron"), default="both")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="compare-native-routes-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(
            json.dumps(
                specification(
                    args.image,
                    args.core_root,
                    args.megatron_root,
                    wait_seconds=args.wait_seconds,
                    backend=args.backend,
                ),
                indent=2,
            )
            + "\n"
        )
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
