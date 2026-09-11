"""Launch a bounded full-SFT scheduling trial; no restart or learning claim."""

import argparse
import json
import shlex
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


def audit_specification(image, retained_output, *, synchronous=False):
    retained = Path(retained_output)
    prefix = Path("/weka/oe-training-default/robertb/open-instruct/miles-scheduling")
    if not retained.is_relative_to(prefix) or len(retained.relative_to(prefix).parts) != 2 or ".." in retained.parts:
        raise ValueError("Use the exact retained scheduling experiment/mode directory")
    if retained.name != ("sync" if synchronous else "async"):
        raise ValueError("Retained mode and synchronous audit flag differ")
    command = f"""set -euo pipefail
cd /opt/core-rl
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/usr/local/cuda/compat:${{LD_LIBRARY_PATH:-}}
python -m scripts.miles.async_trial {shlex.quote(ROOT)} {shlex.quote(str(retained))} {"--synchronous" if synchronous else ""} --audit-only /output/audit.json
"""
    return {
        "version": "v2",
        "description": f"Independent CPU re-audit of retained scheduling trial {retained}",
        "tasks": [
            {
                "name": "scheduling-reaudit",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"cpuCount": 8, "memory": "32 GiB"},
                "context": {"priority": "urgent", "minRuntime": "20m", "autoResume": False},
                "constraints": {"cluster": ["ai2/saturn"]},
                "timeout": "30m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--audit-retained-output", type=Path)
    args = parser.parse_args()
    spec = (
        audit_specification(args.image, args.audit_retained_output, synchronous=args.synchronous)
        if args.audit_retained_output is not None
        else specification(args.image, synchronous=args.synchronous)
    )
    document = json.dumps(spec, indent=2) + "\n"
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
