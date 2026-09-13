"""Launch source inventory or retained-sample audits on Saturn with exact auditor source."""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


def specification(image, mode, paths, source):
    encoded = base64.b64encode(source).decode()
    setup = (
        f"import base64,pathlib; pathlib.Path('/output/readiness_cpu.py').write_bytes(base64.b64decode({encoded!r}))"
    )
    command = "\n".join(
        (
            "set -euo pipefail",
            "cd /opt/core-rl",
            "mkdir -p /output",
            "export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=/opt/core-rl",
            "export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}",
            "python -c " + shlex.quote(setup),
            "python /output/readiness_cpu.py "
            + shlex.join(([] if mode in ("prepare-long", "services") else [mode]) + list(map(str, paths))),
        )
    )
    return {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": f"Readiness {mode}; source sha256 {hashlib.sha256(source).hexdigest()}",
        "tasks": [
            {
                "name": f"readiness-{mode}",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [
                    {"mountPath": f"/weka/{name}", "source": {"weka": name}}
                    for name in ("oe-training-default", "oe-adapt-default")
                ],
                "result": {"path": "/output"},
                "resources": {"cpuCount": 8, "memory": "64 GiB", "gpuCount": 0},
                "constraints": {"cluster": ["ai2/saturn"]},
                "context": {"priority": "urgent", "minRuntime": "10m", "autoResume": False},
                "timeout": "45m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument(
        "mode", choices=("inspect", "audit", "prepare-long", "lifecycle", "rescore", "drift", "features", "services")
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source = (
        Path(__file__)
        .with_name(
            {"prepare-long": "prepare_long_context.py", "services": "readiness_services.py"}.get(
                args.mode, "readiness_cpu.py"
            )
        )
        .read_bytes()
    )
    document = specification(args.image, args.mode, args.paths, source)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(document))
        result = subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
            capture_output=True,
            text=True,
        )
    args.receipt.write_text(json.dumps({"spec": document, "result": json.loads(result.stdout)}, indent=2))
    print(result.stdout)


if __name__ == "__main__":
    main()
