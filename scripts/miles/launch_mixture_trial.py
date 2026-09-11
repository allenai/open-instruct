"""Launch a bounded, frozen three-source trial from existing prepared sources."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

PARITY_ROOT = "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1"
TASK_ROOT = "/weka/oe-training-default/robertb/open-instruct/miles-core-datasources/01M26GC6F3TRRQEXR9HJQR0XGG"


def specification(image, *, response_cap=8192):
    if response_cap <= 0:
        raise ValueError("response_cap must be positive")
    command = f"""set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=1
export HF_HOME=/tmp/hf-cache
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/miles-core-mixture/$BEAKER_EXPERIMENT_ID
mkdir -p /output
trap 'for name in preparation.json arguments.json audit.json; do if [ -f "$RUN_ROOT/$name" ]; then cp "$RUN_ROOT/$name" /output/; fi; done' EXIT
python scripts/miles/preflight_attention.py --backend flash_4
python -m scripts.miles.mixture_trials prepare "$RUN_ROOT" --gsm8k-root {shlex.quote(PARITY_ROOT)} --math-root {shlex.quote(TASK_ROOT + "/math")} --ifeval-root {shlex.quote(TASK_ROOT + "/ifeval")} --hf {shlex.quote(PARITY_ROOT + "/hf")} --response-cap {response_cap}
python -m scripts.miles.mixture_trials validate "$RUN_ROOT"
python -m scripts.miles.mixture_trials run "$RUN_ROOT"
"""
    return {
        "version": "v2",
        "description": f"Core EP2 + SGLang: GSM8K/math/legacy-IF mixture, two updates, {response_cap}-token responses",
        "tasks": [
            {
                "name": "mixture",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 3, "sharedMemory": "100 GiB"},
                "context": {"priority": "urgent", "minRuntime": "30m", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "60m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--response-cap", type=int, default=8192)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, response_cap=args.response_cap), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-mixture-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
