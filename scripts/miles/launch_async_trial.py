"""Launch a bounded full-SFT scheduling trial; no restart or learning claim."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.launch_gsm8k_parity import ROOT


def specification(image, *, synchronous=False):
    mode = "sync" if synchronous else "async"
    command = f"""set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=1
export HF_HOME=/tmp/hf-cache
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/miles-scheduling/$BEAKER_EXPERIMENT_ID/{mode}
mkdir -p /output
trap 'for name in arguments.json audit.json; do if [ -f "$RUN_ROOT/$name" ]; then cp "$RUN_ROOT/$name" /output/; fi; done; if [ -d "$RUN_ROOT/metrics" ]; then cp "$RUN_ROOT/metrics/"*.jsonl /output/; fi' EXIT
python scripts/miles/preflight_attention.py --backend flash_4
python -m scripts.miles.async_trial {ROOT} "$RUN_ROOT" {"--synchronous" if synchronous else ""}
"""
    return {
        "version": "v2",
        "description": f"Core full SFT {mode}: four updates, bounded lag, independent rewards and publication checks",
        "tasks": [
            {
                "name": "scheduling-" + mode,
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 3, "sharedMemory": "100 GiB"},
                "context": {"priority": "urgent", "minRuntime": "30m", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "45m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, synchronous=args.synchronous), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-scheduling-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
