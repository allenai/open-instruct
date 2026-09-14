"""Submit the committed reload probe through the MILES image wrapper."""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


def specification(image, root, source):
    setup = (
        "import base64,pathlib; pathlib.Path('/output/reload.py').write_bytes(base64.b64decode("
        + repr(base64.b64encode(source).decode())
        + "))"
    )
    command = "\n".join(
        [
            "set -euo pipefail",
            "cd /opt/core-rl",
            "mkdir -p /output",
            "export PYTHONPATH=/opt/core-rl OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false",
            "export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models",
            "python -c " + shlex.quote(setup),
            "python /output/reload.py " + shlex.quote(str(root)),
        ]
    )
    return {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": f"Dense fresh HF reload; probe sha256 {hashlib.sha256(source).hexdigest()}",
        "tasks": [
            {
                "name": "readiness-dense-reload",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 1, "memory": "64 GiB", "sharedMemory": "16 GiB"},
                "constraints": {"cluster": ["ai2/holmes"]},
                "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                "timeout": "30m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("root", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name("readiness_reload.py").read_bytes()
    document = specification(args.image, args.root, source)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "experiment.json"
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
