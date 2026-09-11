"""CPU-only independent audits of retained control-exercise outputs on Saturn."""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.launch_gsm8k_parity import ROOT


def specification(image, experiment, replay_only=False):
    if not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", experiment):
        raise ValueError("Use an immutable experiment ID")
    retained = f"/weka/oe-training-default/robertb/open-instruct/control-exercise/{experiment}"
    arms = "replay-admission64" if replay_only else "controls sync async"
    command = f"""set -euo pipefail
cd /opt/core-rl
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/usr/local/cuda/compat:${{LD_LIBRARY_PATH:-}}
export WANDB_MODE=disabled OMP_NUM_THREADS=2
mkdir -p /output
python - <<'INNER'
import hashlib,json
from pathlib import Path
worker=Path('scripts/miles/exercise_controls.py')
Path('/output/provenance.json').write_text(json.dumps(dict(retained_experiment='{experiment}',
    auditor_sha256=hashlib.sha256(worker.read_bytes()).hexdigest()),indent=2))
INNER
for arm in {arms}; do
  deadline=$((SECONDS + 7200))
  while [ ! -f {retained}/$arm/elapsed.json ]; do
    if (( SECONDS > deadline )); then echo "Timed out waiting for $arm"; exit 1; fi
    echo "Waiting for retained $arm execution to finish"
    sleep 30
  done
  updates=24
  if [ "$arm" = controls ]; then updates=4; fi
  if [ "$arm" = replay-admission64 ]; then updates=8; fi
  python -m scripts.miles.exercise_controls audit {ROOT} {retained}/$arm "$arm" --updates "$updates" --report /output/$arm/audit.json
done
"""
    return dict(
        version="v2",
        description=f"Independent CPU audit of Core scoring/scheduling controls {experiment}",
        tasks=[
            dict(
                name="reaudit",
                image={"beaker": image},
                command=["bash", "-c"],
                arguments=[command],
                datasets=[{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                result={"path": "/output"},
                resources={"cpuCount": 4, "memory": "32 GiB", "gpuCount": 0},
                constraints={"cluster": ["ai2/saturn"]},
                context={"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                timeout="3h",
            )
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("experiment")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, args.experiment, args.replay_only), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="control-reaudit-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
