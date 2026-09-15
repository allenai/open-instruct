"""Snapshot live timing artifacts on CPU/Saturn, without training or generation."""

import base64
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    image, *roots = sys.argv[1:]
    if not roots:
        raise ValueError("Provide at least one run root")
    source = Path(__file__).with_name("collect_timing_artifacts.py").read_bytes()
    provenance = {
        "image": image,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "audit_sha256": hashlib.sha256(source).hexdigest(),
        "roots": roots,
    }
    commands = ["set -euo pipefail", "mkdir -p /output", "cd /opt/core-rl"]
    for path, content in {
        "/tmp/collect-timings.py": source,
        "/output/provenance.json": json.dumps(provenance, indent=2).encode(),
    }.items():
        commands.append(
            f"printf %s {shlex.quote(base64.b64encode(content).decode())} | base64 -d > {shlex.quote(path)}"
        )
    commands.append(shlex.join(["python", "/tmp/collect-timings.py", *roots, "--output", "/output"]))
    spec = {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": "Read-only baseline timing artifact snapshot",
        "tasks": [
            {
                "name": "baseline-timing-artifacts-20260915",
                "image": {"beaker": image},
                "command": ["bash", "-lc"],
                "arguments": ["\n".join(commands)],
                "resources": {"cpuCount": 4, "memory": "4 GiB", "gpuCount": 0, "sharedMemory": "4 GiB"},
                "constraints": {"cluster": ["ai2/saturn"]},
                "context": {"priority": "urgent", "minRuntime": "10m", "autoResume": False},
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "timeout": "20m",
                "envVars": [{"name": "OMP_NUM_THREADS", "value": "2"}, {"name": "CUDA_VISIBLE_DEVICES", "value": ""}],
            }
        ],
    }
    with tempfile.TemporaryDirectory(prefix="engine-drain-audit-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(spec))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
