"""Launch the committed-image two-GPU delivery probe on Holmes."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    image = sys.argv[1]
    spec = {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": "Independent Core engine delivery transport probe",
        "tasks": [
            {
                "name": "engine-delivery-transport-20260913",
                "image": {"beaker": image},
                "command": ["bash", "-lc"],
                "arguments": ["mkdir -p /output; cd /opt/core-rl; python -m scripts.miles.engine_delivery_probe"],
                "resources": {"gpuCount": 2, "memory": "64 GiB", "sharedMemory": "8 GiB"},
                "context": {"priority": "urgent", "minRuntime": "1h"},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "1h",
                "result": {"path": "/output"},
                "envVars": [{"name": "NCCL_CUMEM_ENABLE", "value": "1"}, {"name": "OMP_NUM_THREADS", "value": "2"}],
            }
        ],
    }
    with tempfile.TemporaryDirectory(prefix="engine-delivery-probe-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(spec))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
