"""Launch a bounded CPU/Saturn comparison of immutable update-100 checkpoints."""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from scripts.miles import launch_diagnostic_retention

REFERENCE = "/weka/oe-training-default/robertb/olmo-miles/checkpoints/olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf"


def specification(image, olmo_miles_source):
    root = launch_diagnostic_retention.ROOT
    document = launch_diagnostic_retention.specification(image)
    files = {
        "scripts/__init__.py": b"",
        "scripts/miles/__init__.py": b"",
        "olmo_miles/__init__.py": b"",
        "olmo_miles/runtime/__init__.py": b"",
    }
    for name in (
        "compare_native_small_drift.py",
        "checkpoint_drift.py",
        "checkpoint_weights.py",
        "core_checkpoint_stream.py",
        "megatron_checkpoint_stream.py",
    ):
        files[f"scripts/miles/{name}"] = Path(__file__).with_name(name).read_bytes()
    for name in ("olmo_weight_export.py", "weight_export_lifecycle.py"):
        files[f"olmo_miles/runtime/{name}"] = (olmo_miles_source / "src/olmo_miles/runtime" / name).read_bytes()
    provenance = {
        "image": image,
        "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "scope": "CPU only; wait for completed update-100 snapshots, then read router, normalization and KDA scalar model tensors",
    }
    lines = [
        "set -euo pipefail",
        "mkdir -p /tmp/native-drift/scripts/miles /tmp/native-drift/olmo_miles/runtime /output",
        "cd /tmp/native-drift",
        "export CUDA_VISIBLE_DEVICES=''",
        "export OMP_NUM_THREADS=2",
        "export MKL_NUM_THREADS=2",
        'export PYTHONPATH="/tmp/native-drift:${PYTHONPATH:-}"',
    ]
    for name, data in files.items():
        lines.append(f"printf %s {shlex.quote(base64.b64encode(data).decode())} | base64 -d > {shlex.quote(name)}")
    lines.append(
        f"printf %s {shlex.quote(base64.b64encode(json.dumps(provenance).encode()).decode())} | base64 -d > /output/provenance.json"
    )
    lines.append(
        shlex.join(
            [
                "python",
                "-m",
                "scripts.miles.compare_native_small_drift",
                "--reference",
                REFERENCE,
                "--core",
                root + "/core/metrics/core/rollout_0000099",
                "--megatron",
                root + "/diagnostic-retained/megatron/iter_0000099",
                "--output",
                "/output/drift.json",
                "--updates",
                "100",
                "--wait-seconds",
                "7200",
            ]
        )
    )
    task = document["tasks"][0]
    task.update(
        name="native-small-parameter-drift",
        arguments=["\n".join(lines) + "\n"],
        resources={"cpuCount": 2, "memory": "32 GiB", "sharedMemory": "4 GiB"},
        context={"priority": "urgent", "minRuntime": "1h", "autoResume": False},
        timeout="3h",
    )
    document["description"] = (
        "Native update-100 router/normalization drift with canonical conversion and completed-save checks"
    )
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--olmo-miles-source", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = specification(args.image, args.olmo_miles_source)
    if args.render_only:
        print(json.dumps(document, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="native-small-drift-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
