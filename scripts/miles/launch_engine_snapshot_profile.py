"""Launch the isolated full-volume snapshot copy profile on one Holmes GPU."""

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
    source = Path(__file__).with_name("engine_snapshot_profile.py").read_bytes()
    provenance = {
        "image": image,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "audit_sha256": hashlib.sha256(source).hexdigest(),
    }
    commands = ["set -euo pipefail", "mkdir -p /output", "cd /opt/core-rl"]
    for path, content in {
        "/tmp/audit-engine-drain.py": source,
        "/output/provenance.json": json.dumps(provenance, indent=2).encode(),
    }.items():
        commands.append(
            f"printf %s {shlex.quote(base64.b64encode(content).decode())} | base64 -d > {shlex.quote(path)}"
        )
    commands.append("python /tmp/audit-engine-drain.py")
    spec = {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": "Isolated full-volume snapshot copy profile",
        "tasks": [
            {
                "name": "engine-snapshot-profile-20260913-a",
                "image": {"beaker": image},
                "command": ["bash", "-lc"],
                "arguments": ["\n".join(commands)],
                "resources": {"cpuCount": 8, "gpuCount": 1, "memory": "192 GiB", "sharedMemory": "96 GiB"},
                "constraints": {"cluster": ["ai2/holmes"]},
                "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                "result": {"path": "/output"},
                "timeout": "1h",
                "envVars": [{"name": "OMP_NUM_THREADS", "value": "2"}],
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
