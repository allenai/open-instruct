"""Launch the bounded SFT GSM8K trial through the committed image wrapper."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

COMMAND = """set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=1
export HF_HOME=/tmp/hf-cache
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/miles-core-sft/$BEAKER_EXPERIMENT_ID
mkdir -p /output "$RUN_ROOT"
# Source weights and responses remain on WEKA; the Beaker result contains reports only.
trap 'for name in preparation.json arguments.json audit.json; do if [ -f "$RUN_ROOT/$name" ]; then cp "$RUN_ROOT/$name" /output/; fi; done' EXIT
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.version.cuda)'
python scripts/miles/sft_gsm8k.py prepare "$RUN_ROOT"
python scripts/miles/sft_gsm8k.py run "$RUN_ROOT"
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    spec = {
        "version": "v2",
        "description": "Open-instruct / MILES / Core: SFT step23607 GSM8K, EP2 + one engine, two updates and held-out before/after",
        "tasks": [
            {
                "name": "sft-gsm8k",
                "image": {"beaker": args.image},
                "command": ["bash", "-c"],
                "arguments": [COMMAND],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 3, "sharedMemory": "100 GiB"},
                "context": {"priority": "normal", "minRuntime": "30m", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "45m",
            }
        ],
    }
    document = json.dumps(spec, indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-sft-trial-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
