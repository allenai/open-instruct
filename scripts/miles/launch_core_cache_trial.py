"""Opt-in one-GPU cold/restored actual Core update qualification."""

import argparse
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

CACHE_ROOT = "/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d"


def specification(image):
    if not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", image):
        raise ValueError("Cache fingerprint requires an immutable Beaker image ID")
    command = f"""set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled
export NCCL_CUMEM_ENABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
mkdir -p /output
trap 'if [ -d /tmp/core-cache-trial ]; then cp -r /tmp/core-cache-trial /output/; fi' EXIT
python -m scripts.miles.core_cache_trial run /tmp/core-cache-trial \\
  --image {shlex.quote(image)} --shared-root {shlex.quote(CACHE_ROOT)}
"""
    return {
        "version": "v2",
        "description": "Core EP1 fixed combined update: cold/restored compiler artifacts and exact state/probability comparison",
        "tasks": [
            {
                "name": "core-cache",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 1, "sharedMemory": "16 GiB"},
                "constraints": {"cluster": ["ai2/holmes"]},
                "context": {"priority": "urgent", "minRuntime": "30m", "autoResume": False},
                "timeout": "45m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    image = args.image
    if not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", image):
        metadata = json.loads(
            subprocess.check_output(["beaker", "image", "get", image, "--format", "json"], text=True)
        )
        image = metadata[0]["id"]
    document = json.dumps(specification(image), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="core-cache-trial-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
