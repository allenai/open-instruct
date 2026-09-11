"""Render or launch the bounded, two-GPU full-SFT durable-continuation gate."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

HF = "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1/hf"
COMMAND = """set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled
export NCCL_CUMEM_ENABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/miles-core-durable/$BEAKER_EXPERIMENT_ID
mkdir -p /output
retain_reports() {
  for name in preparation.json audit.json; do
    if [ -f "$RUN_ROOT/$name" ]; then cp "$RUN_ROOT/$name" /output/; fi
  done
  for phase in control split resumed; do
    mkdir -p "/output/$phase"
    if [ -d "$RUN_ROOT/$phase" ]; then
      find "$RUN_ROOT/$phase" -maxdepth 1 -type f \\( -name 'rank*.json' -o -name 'training_contract_rank*.jsonl' \\) -exec cp '{}' "/output/$phase/" \\;
    fi
  done
  for name in core-latest.json core/rollout_0000001/complete.json core/rollout_0000001/pending.json rollout/global_dataset_state_dict_1.pt; do
    if [ -f "$RUN_ROOT/split/$name" ]; then
      mkdir -p "/output/split/$(dirname "$name")"
      cp "$RUN_ROOT/split/$name" "/output/split/$name"
    fi
  done
}
trap retain_reports EXIT
python -c 'import torch; assert "B300" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name())'
python scripts/miles/preflight_attention.py --backend flash_4
python -m scripts.miles.durable_continuation prepare "$RUN_ROOT" --hf __HF__
for phase in control split resumed; do
  torchrun --standalone --nproc-per-node=2 -m scripts.miles.durable_continuation run "$RUN_ROOT" --phase "$phase" --backend flash_4
done
python -m scripts.miles.durable_continuation audit "$RUN_ROOT" --world 2
"""


def specification(image, hf=HF):
    return {
        "version": "v2",
        "description": "Core full-SFT EP2 durable continuation: fixed four-update control vs two + fresh-process two",
        "tasks": [
            {
                "name": "durable-continuation",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [COMMAND.replace("__HF__", shlex.quote(hf))],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 2, "sharedMemory": "100 GiB"},
                "constraints": {"cluster": ["ai2/holmes"]},
                "context": {"priority": "urgent", "minRuntime": "60m", "autoResume": False},
                "timeout": "90m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--hf", default=HF)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, args.hf), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-core-durable-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
