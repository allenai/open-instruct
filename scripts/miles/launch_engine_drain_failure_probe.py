"""Submit an isolated real-engine retirement test with an exact harness inventory."""

import base64
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    image = sys.argv[1]
    config = Path(sys.argv[2])
    files = {
        "/tmp/engine-drain-failure.py": Path(__file__).with_name("engine_drain_failure_probe.py").read_bytes(),
        "/tmp/engine-drain-failure.toml": config.read_bytes(),
    }
    provenance = {
        "image": image,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        "scope": "test harness only; immutable image supplies the unmodified production implementation",
    }
    files["/output/provenance.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    commands = ["set -euo pipefail", "mkdir -p /output", "cd /opt/core-rl"]
    commands.extend(
        f"printf %s {shlex.quote(base64.b64encode(content).decode())} | base64 -d > {shlex.quote(name)}"
        for name, content in files.items()
    )
    commands.append("python /tmp/engine-drain-failure.py /tmp/engine-drain-failure.toml 2>&1 | tee /output/run.log")
    spec = {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": "Engine-drain fresh-process resume and deliberate owned SGLang engine loss",
        "tasks": [
            {
                "name": "engine-drain-failure-20260913",
                "image": {"beaker": image},
                "command": ["bash", "-lc"],
                "arguments": ["\n".join(commands)],
                "resources": {"gpuCount": 4, "memory": "256 GiB", "sharedMemory": "128 GiB"},
                "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "timeout": "1h",
                "result": {"path": "/output"},
                "envVars": [{"name": "NCCL_CUMEM_ENABLE", "value": "1"}, {"name": "OMP_NUM_THREADS", "value": "2"}],
            }
        ],
    }
    with tempfile.TemporaryDirectory(prefix="engine-drain-failure-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(spec))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
