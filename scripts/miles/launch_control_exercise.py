"""One image, three grouped allocations: scoring, matched scheduling, controls."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.launch_gsm8k_parity import ROOT


def task(image, name, command, gpus, timeout):
    common = """set -euo pipefail
cd /opt/core-rl
export TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=1 OMP_NUM_THREADS=2
export HF_HOME=/tmp/hf-cache WANDB_MODE=offline
export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/control-exercise/$BEAKER_EXPERIMENT_ID
mkdir -p /output "$RUN_ROOT"
python -c 'import torch; assert "B300" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name())'
python scripts/miles/preflight_attention.py --backend flash_4
"""
    return dict(
        name=name,
        image={"beaker": image},
        command=["bash", "-c"],
        arguments=[common + command],
        datasets=[{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
        result={"path": "/output"},
        resources={"gpuCount": gpus, "memory": "256 GiB", "sharedMemory": "100 GiB"},
        context={"priority": "urgent", "minRuntime": "1h", "autoResume": False},
        constraints={"cluster": ["ai2/holmes"]},
        timeout=timeout,
    )


def live_command(arms, updates):
    return f"""copy_reports() {{
  for arm in {arms}; do
    mkdir -p /output/$arm
    for name in audit.json arguments.json protocol.json run.toml elapsed.json plan.json validate.log run.log; do
      if [ -f "$RUN_ROOT/$arm/$name" ]; then cp "$RUN_ROOT/$arm/$name" /output/$arm/; fi
    done
    if [ -d "$RUN_ROOT/$arm/metrics" ]; then find "$RUN_ROOT/$arm/metrics" -maxdepth 1 -name '*.jsonl' -exec cp '{{}}' /output/$arm/ \\;; fi
  done
}}
trap copy_reports EXIT
status=0
for arm in {arms}; do
  export TRITON_CACHE_DIR=/tmp/control-cache/$arm/triton
  export TORCHINDUCTOR_CACHE_DIR=/tmp/control-cache/$arm/inductor
  python -m scripts.miles.exercise_controls prepare {ROOT} "$RUN_ROOT/$arm" "$arm" --updates {updates}
  python -m open_instruct.miles plan "$RUN_ROOT/$arm/run.toml" --set 'core.row_specialization="dynamic"' > "$RUN_ROOT/$arm/plan.json"
  python -m open_instruct.miles validate "$RUN_ROOT/$arm/run.toml" > "$RUN_ROOT/$arm/validate.log" 2>&1
  if python -m scripts.miles.exercise_controls train {ROOT} "$RUN_ROOT/$arm" "$arm" --updates {updates} 2>&1 | tee "$RUN_ROOT/$arm/run.log"; then
    if python -m scripts.miles.exercise_controls audit {ROOT} "$RUN_ROOT/$arm" "$arm" --updates {updates}; then :; else status=1; fi
  else status=1
  fi
  ray stop --force || true
done
exit "$status"
"""


def score_command():
    return f"""export SCORE_ROOT="$RUN_ROOT/scoring"
mkdir "$SCORE_ROOT"
trap 'cp -r "$SCORE_ROOT" /output/' EXIT
python - <<'INNER'
import hashlib,json,os
from pathlib import Path
from olmo_core.kernels import swiglu
kernel=Path(swiglu.__file__)
hash=hashlib.sha256(kernel.read_bytes()).hexdigest()
worker=Path("scripts/miles/profile_core_score_variants.py")
manifest=dict(configured_modes=True,parent_sha256=hash,candidate_sha256=hash,
              worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest())
(Path(os.environ["SCORE_ROOT"])/"manifest.json").write_text(json.dumps(manifest))
INNER
cat > /tmp/configured-score-rank.sh <<'INNER'
set -euo pipefail
arm=$1
export TRITON_CACHE_DIR=/tmp/configured-score/$arm/rank$RANK/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/configured-score/$arm/rank$RANK/inductor
exec python -m scripts.miles.profile_core_score_variants {ROOT} "$SCORE_ROOT" "$SCORE_ROOT/manifest.json" --arm "$arm"
INNER
status=0
for arm in parent candidate; do
  if python -m torch.distributed.run --master_port=29657 --nnodes=1 --nproc_per_node=2 --no-python bash /tmp/configured-score-rank.sh "$arm"; then :; else status=1; fi
done
if python -m scripts.miles.profile_core_score_variants {ROOT} "$SCORE_ROOT" "$SCORE_ROOT/manifest.json"; then :; else status=1; fi
exit "$status"
"""


def specification(image):
    return dict(
        version="v2",
        description="Pinned Core dynamic rows: exact EP2 scores; 24-update sync/async pair; grouped CLI controls",
        tasks=[
            task(image, "scores", score_command(), 2, "90m"),
            task(image, "scheduling", live_command("sync async", 24), 3, "3h"),
            task(image, "controls", live_command("controls", 4), 3, "90m"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--render-only", action="store_true")
    options = parser.parse_args()
    document = json.dumps(specification(options.image), indent=2) + "\n"
    if options.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="core-control-exercise-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
