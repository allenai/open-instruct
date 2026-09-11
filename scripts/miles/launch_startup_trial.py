"""Grouped full-model Ray startup qualification on Holmes."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.launch_gsm8k_parity import ROOT


def specification(image):
    command = f"""set -euo pipefail
cd /opt/core-rl
export TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=1 OMP_NUM_THREADS=2 WANDB_MODE=disabled
export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/startup/$BEAKER_EXPERIMENT_ID
mkdir -p /output
retain() {{
  for arm in cold restored; do
    mkdir -p /output/$arm
    for name in run.toml run.log elapsed.json; do
      if [ -f "$RUN_ROOT/$arm/$name" ]; then cp "$RUN_ROOT/$arm/$name" /output/$arm/; fi
    done
    if [ -d "$RUN_ROOT/$arm/metrics" ]; then
      find "$RUN_ROOT/$arm/metrics" -maxdepth 1 -type f -exec cp '{{}}' /output/$arm/ \\;
    fi
  done
  if [ -f "$RUN_ROOT/comparison.json" ]; then cp "$RUN_ROOT/comparison.json" /output/; fi
}}
trap retain EXIT
python -m scripts.miles.startup_trial run "$RUN_ROOT" --campaign {ROOT}
"""
    return dict(
        version="v2",
        description="Core RL cold/restored Triton: actual Ray workers, full SFT EP2/TP1, startup phase timers",
        tasks=[
            dict(
                name="startup",
                image={"beaker": image},
                command=["bash", "-c"],
                arguments=[command],
                datasets=[dict(mountPath="/weka/oe-training-default", source={"weka": "oe-training-default"})],
                result={"path": "/output"},
                resources={"gpuCount": 3, "sharedMemory": "100 GiB"},
                constraints={"cluster": ["ai2/holmes"]},
                context={"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                timeout="2h",
            )
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    text = json.dumps(specification(args.image), indent=2) + "\n"
    if args.render_only:
        print(text)
        return
    with tempfile.TemporaryDirectory(prefix="core-startup-") as tmp:
        path = Path(tmp) / "experiment.json"
        path.write_text(text)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
