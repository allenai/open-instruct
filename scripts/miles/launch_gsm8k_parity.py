"""Launch shared-data preparation or the 100-update Core GSM8K arm."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1"


def specification(image, stage, *, megatron_directory="megatron"):
    if (
        not megatron_directory
        or Path(megatron_directory).name != megatron_directory
        or megatron_directory in (".", "..")
    ):
        raise ValueError("megatron_directory must be one directory name beneath the campaign root")
    if stage != "audit" and megatron_directory != "megatron":
        raise ValueError("megatron_directory selects audit artifacts only")
    common = f"""set -euo pipefail
cd /opt/core-rl
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=1
export HF_HOME=/tmp/hf-cache
export RUN_ROOT={shlex.quote(ROOT)}
mkdir -p /output
"""
    if stage == "prepare":
        command = (
            common
            + """python scripts/miles/prepare_gsm8k_parity.py "$RUN_ROOT"
python scripts/miles/gsm8k_parity.py "$RUN_ROOT" --validate-only
cp "$RUN_ROOT/preparation.json" /output/
"""
        )
        resources = {"cpuCount": 8, "memory": "32 GiB", "gpuCount": 0, "sharedMemory": "4 GiB"}
        cluster, timeout, env = "ai2/saturn", "30m", []
    elif stage == "audit-core":
        command = (
            common
            + 'python scripts/miles/analyze_gsm8k_parity.py audit "$RUN_ROOT" --backend core\n'
            + 'cp "$RUN_ROOT/core/audit.json" /output/core-audit.json\n'
        )
        resources = {"cpuCount": 8, "memory": "32 GiB", "gpuCount": 0, "sharedMemory": "4 GiB"}
        cluster, timeout, env = "ai2/saturn", "30m", []
    elif stage == "audit":
        command = (
            common
            + f"export MEGATRON_DIRECTORY={shlex.quote(megatron_directory)}\n"
            + """python scripts/miles/analyze_gsm8k_parity.py audit "$RUN_ROOT" --backend core
python scripts/miles/analyze_gsm8k_parity.py audit "$RUN_ROOT" --backend megatron --megatron-directory "$MEGATRON_DIRECTORY"
python scripts/miles/analyze_gsm8k_parity.py compare "$RUN_ROOT" --megatron-directory "$MEGATRON_DIRECTORY"
cp "$RUN_ROOT/core/audit.json" /output/core-audit.json
cp "$RUN_ROOT/$MEGATRON_DIRECTORY/audit.json" /output/megatron-audit.json
cp "$RUN_ROOT/comparison.json" /output/
"""
        )
        resources = {"cpuCount": 8, "memory": "32 GiB", "gpuCount": 0, "sharedMemory": "4 GiB"}
        cluster, timeout, env = "ai2/saturn", "30m", []
    else:
        command = (
            common
            + """export WANDB_MODE=online
copy_reports() {
  for name in arguments.json effective.json preparation.json completion.json; do
    if [ -f "$RUN_ROOT/core/$name" ]; then cp "$RUN_ROOT/core/$name" /output/; fi
  done
  for name in publication.jsonl training_contract_rank0.jsonl training_contract_rank1.jsonl; do
    if [ -f "$RUN_ROOT/core/metrics/$name" ]; then cp "$RUN_ROOT/core/metrics/$name" /output/; fi
  done
}
trap copy_reports EXIT
python -c 'import torch; assert "B300" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name(), torch.version.cuda)'
python scripts/miles/preflight_attention.py --backend flash_4
python scripts/miles/gsm8k_parity.py "$RUN_ROOT"
"""
        )
        resources = {"gpuCount": 3, "sharedMemory": "100 GiB"}
        cluster, timeout = "ai2/holmes", "4h"
        env = [{"name": "WANDB_API_KEY", "secret": "robertb_WANDB_API_KEY"}]
    return {
        "version": "v2",
        "description": f"GSM8K Core/Megatron comparison: {stage}; 100 updates, 400 train, 128 held-out test prompts",
        "tasks": [
            {
                "name": "gsm8k-" + stage,
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "envVars": env,
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": resources,
                "context": {
                    "priority": "urgent",
                    "minRuntime": "4h" if stage == "core" else "20m",
                    "autoResume": False,
                },
                "constraints": {"cluster": [cluster]},
                "timeout": timeout,
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--stage", choices=("prepare", "core", "audit", "audit-core"), required=True)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--megatron-directory", default="megatron")
    args = parser.parse_args()
    document = (
        json.dumps(specification(args.image, args.stage, megatron_directory=args.megatron_directory), indent=2) + "\n"
    )
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-gsm8k-parity-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
