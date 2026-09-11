"""Launch CPU-only WEKA checkpoint retention on Saturn."""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260911-abhishek-500-v1"


def specification(image):
    source = (Path(__file__).parent / "retain_diagnostic_checkpoints.py").read_bytes()
    encoded = base64.b64encode(source).decode()
    command = "\n".join(
        [
            "set -euo pipefail",
            "mkdir -p /output",
            f"printf %s {shlex.quote(encoded)} | base64 -d > /tmp/retain-checkpoints.py",
            "python /tmp/retain-checkpoints.py "
            + shlex.quote(ROOT + "/megatron/checkpoints")
            + " "
            + shlex.quote(ROOT + "/diagnostic-retained/megatron")
            + " --max-hours 17 | tee /output/retained.jsonl",
        ]
    )
    return {
        "version": "v2",
        "description": "Retain completed intermediate Megatron500 saves for parameter/route drift; CPU only; "
        + hashlib.sha256(source).hexdigest(),
        "tasks": [
            {
                "name": "retain-diagnostic-checkpoints",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "resources": {"cpuCount": 1, "memory": "2 GiB"},
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "constraints": {"cluster": ["ai2/saturn"]},
                "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                "timeout": "18h",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="retain-diagnostics-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(specification(args.image), indent=2) + "\n")
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
