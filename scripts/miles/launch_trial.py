"""Submit a bounded public-input MoE trial through the repository image wrapper."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


COMMAND = """set -euo pipefail
cd /opt/core-rl
export HF_HOME=/tmp/hf-cache
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
mkdir -p /output
python tests/miles/local_moe.py bootstrap /output/toy-moe
python tests/miles/local_moe.py run /output/toy-moe
python tests/miles/local_moe.py run /output/toy-moe --resume
python tests/miles/local_moe.py audit /output/toy-moe
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--cluster", default="ai2/jupiter")
    parser.add_argument("--workspace", default="ai2/open-instruct-dev")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    spec = {
        "version": "v2",
        "description": "MILES/Core toy MoE: public GSM8K, weight publication, native restart, independent audit",
        "tasks": [
            {
                "name": "toy-moe-gsm8k",
                "image": {"beaker": args.image},
                "command": ["bash", "-c"],
                "arguments": [COMMAND],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 1, "sharedMemory": "8 GiB"},
                "context": {"priority": "normal", "minRuntime": "15m", "autoResume": False},
                "constraints": {"cluster": [args.cluster]},
                "timeout": "20m",
            }
        ],
    }
    document = json.dumps(spec, indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-core-trial-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", args.workspace, "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
