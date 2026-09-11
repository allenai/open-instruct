"""Launch verified historical light-SFT input preparation or its Core learning arm."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.light_sft_gsm8k import ROOT, history


def specification(image, stage):
    if stage not in ("prepare", "core", "audit"):
        raise ValueError("Unknown light-SFT stage")
    cpu = stage != "core"
    command = f"""set -euo pipefail
cd /opt/core-rl
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=1
export HF_HOME=/tmp/hf-cache
export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models
mkdir -p /output
export RUN_ROOT={ROOT}
"""
    if stage == "prepare":
        command += """python -m scripts.miles.light_sft_gsm8k prepare
python -m scripts.miles.light_sft_gsm8k validate
cp "$RUN_ROOT/preparation.json" /output/
cp "$RUN_ROOT/historical.json" /output/
cp "$RUN_ROOT/checkpoint-inventory.json" /output/
"""
    elif stage == "audit":
        command += """python -m scripts.miles.light_sft_gsm8k audit
cp "$RUN_ROOT/core/audit.json" /output/
"""
    else:
        command += """export WANDB_MODE=online
copy_reports() {
  for name in arguments.json effective.json preparation.json completion.json offline-0.json offline-200.json; do
    if [ -f "$RUN_ROOT/core/$name" ]; then cp "$RUN_ROOT/core/$name" /output/; fi
  done
  if [ -d "$RUN_ROOT/core/metrics" ]; then
    find "$RUN_ROOT/core/metrics" -maxdepth 1 -name '*.jsonl' -exec cp {} /output/ \\;
  fi
}
trap copy_reports EXIT
python -c 'import torch; assert torch.cuda.device_count() == 4; assert "B300" in torch.cuda.get_device_name()'
python scripts/miles/preflight_attention.py --backend flash_4
python -m scripts.miles.light_sft_gsm8k run
"""
    mounts = [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}]
    if stage == "prepare":
        mounts.append(
            {"mountPath": "/historical-offline", "source": {"beaker": history()["full_test"]["after_result"]}}
        )
    return {
        "version": "v2",
        "description": f"Light SFT1000 historical GSM8K / Core200: {stage}",
        "tasks": [
            {
                "name": "light-sft-" + stage,
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": mounts,
                "result": {"path": "/output"},
                "resources": {"cpuCount": 8, "memory": "32 GiB", "gpuCount": 0, "sharedMemory": "4 GiB"}
                if cpu
                else {"gpuCount": 4, "sharedMemory": "100 GiB"},
                "constraints": {"cluster": ["ai2/saturn" if cpu else "ai2/holmes"]},
                "context": {"priority": "urgent", "minRuntime": "20m" if cpu else "8h", "autoResume": False},
                "timeout": "30m" if cpu else "8h",
                "envVars": [] if cpu else [{"name": "WANDB_API_KEY", "secret": "robertb_WANDB_API_KEY"}],
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--stage", required=True, choices=("prepare", "core", "audit"))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, args.stage), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="light-sft-gsm8k-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
