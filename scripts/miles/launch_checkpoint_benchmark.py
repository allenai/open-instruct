"""Launch bounded checkpoint profiling and exact export/resume checks on WEKA."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.launch_durable_continuation import HF

COMMAND = r"""set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled
export NCCL_CUMEM_ENABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/checkpoint-perf/$BEAKER_EXPERIMENT_ID
export TRITON_CACHE_DIR=$RUN_ROOT/triton-cache
mkdir -p /output "$RUN_ROOT"
retain_reports() {
  find "$RUN_ROOT" -maxdepth 3 -type f \( -name '*.json' -o -name '*.log' \) ! -path '*/hf-boundary/*' ! -path '*/triton-cache/*' | while read -r path; do
    relative=${path#"$RUN_ROOT/"}
    mkdir -p "/output/$(dirname "$relative")"
    cp "$path" "/output/$relative"
  done
}
trap retain_reports EXIT
python -c 'import torch; assert "B300" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name())'
python scripts/miles/preflight_attention.py --backend flash_4
# Existing checkpoint topology tests include optimizer masters, moments, steps and buffers.
python -m pytest -q /opt/core-rl/sources/olmo-core/src/test/nn/moe/v2/ep_checkpoint_reshard_test.py /opt/core-rl/sources/olmo-core/src/test/train/train_module/transformer/ddp_train_module_test.py -k 'checkpoint or roundtrip or resume' > "$RUN_ROOT/topology-tests.log" 2>&1
cp "$RUN_ROOT/topology-tests.log" /output/
for mode in __MODES__; do
  root=$RUN_ROOT/$mode
  python -m scripts.miles.checkpoint_benchmark prepare "$root" --hf __HF__
  for phase in split resumed; do
    torchrun --standalone --nproc-per-node=2 -m scripts.miles.checkpoint_benchmark run "$root" --phase "$phase" --mode "$mode" --backend flash_4
  done
  # Retain all measurement arms even when the strict exactness gate fails.
  python -m scripts.miles.checkpoint_benchmark audit "$root" --world 2 || touch "$RUN_ROOT/failed-$mode"
  retain_reports
done
if compgen -G "$RUN_ROOT/failed-*" > /dev/null; then exit 1; fi
"""


def specification(image, hf=HF, modes=("baseline", "balanced")):
    if not modes or any(mode not in ("baseline", "compact", "balanced", "processes") for mode in modes):
        raise ValueError("Unknown or empty checkpoint measurement modes")
    command = COMMAND.replace("__HF__", shlex.quote(hf)).replace("__MODES__", " ".join(map(shlex.quote, modes)))
    return {
        "version": "v2",
        "description": "Core checkpoint profile: same-boundary continuation, HF export, baseline and candidate writers",
        "tasks": [
            {
                "name": "checkpoint-profile",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 2, "sharedMemory": "100 GiB"},
                "constraints": {"cluster": ["ai2/holmes"]},
                "context": {"priority": "urgent", "minRuntime": "60m", "autoResume": False},
                "timeout": "3h",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--hf", default=HF)
    parser.add_argument("--modes", nargs="+", default=["baseline", "balanced"])
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, args.hf, args.modes), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-checkpoint-profile-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
