"""Launch isolated parent/candidate EP2 scoring on successive retained batches."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.launch_frozen_core_score_profile import ROOT, encoded

IMAGE = "01M279KKMX3RGXB6AYB273DE3C"
OUTPUT = ROOT + "/score-variants-20260911-v1"


def specification(image, *, output=OUTPUT):
    if image != IMAGE:
        raise ValueError("Use the immutable parent Core290 image")
    if Path(output).parent != Path(ROOT) or not re.fullmatch(r"score-variants-[A-Za-z0-9._-]+", Path(output).name):
        raise ValueError("Use a distinct score-variants directory beneath the retained campaign")
    directory = Path(__file__).parent
    worker = (directory / "profile_core_score_variants.py").read_text()
    patch = (directory / "diagnostics/swiglu-runtime-rows.patch").read_text()
    manifest = json.loads((directory / "diagnostics/score-variants.json").read_text())
    if manifest["image"] != image or hashlib.sha256(patch.encode()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Variant patch or image differs from the committed manifest")
    manifest["worker_sha256"] = hashlib.sha256(worker.encode()).hexdigest()
    rank_script = f"""#!/bin/bash
set -euo pipefail
arm=${{1:?parent or candidate}}
export PYTHONPATH=/tmp/score-$arm/core/src:/opt/core-rl:/opt/core-rl/sources/miles
export TRITON_CACHE_DIR=/tmp/score-$arm/cache/rank${{RANK}}/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/score-$arm/cache/rank${{RANK}}/inductor
exec python /tmp/profile_core_score_variants.py {shlex.quote(ROOT)} {shlex.quote(output)} /tmp/score-variants.json --arm "$arm"
"""
    command = f"""set -euo pipefail
cd /opt/core-rl
test ! -e {shlex.quote(output)}
mkdir -p {shlex.quote(output)} /output /tmp/score-parent /tmp/score-candidate
printf %s {shlex.quote(encoded(worker))} | base64 -d > /tmp/profile_core_score_variants.py
printf %s {shlex.quote(encoded(patch))} | base64 -d > /tmp/score-variants.patch
printf %s {shlex.quote(encoded(json.dumps(manifest)))} | base64 -d > /tmp/score-variants.json
printf %s {shlex.quote(encoded(rank_script))} | base64 -d > /tmp/score-variants-rank.sh
cp -a /opt/core-rl/sources/olmo-core /tmp/score-parent/core
cp -a /opt/core-rl/sources/olmo-core /tmp/score-candidate/core
patch --batch --fuzz=0 -p1 -d /tmp/score-candidate/core < /tmp/score-variants.patch
cp /tmp/score-variants.json {shlex.quote(output)}/manifest.json
trap 'cp -r {shlex.quote(output)}/parent {shlex.quote(output)}/candidate /output/ 2>/dev/null || true; cp {shlex.quote(output)}/comparison.json /tmp/score-variants.json /output/ 2>/dev/null || true' EXIT
export NCCL_CUMEM_ENABLE=1 TOKENIZERS_PARALLELISM=false HF_HOME=/tmp/hf-cache WANDB_MODE=disabled
status=0
for arm in parent candidate; do
  if python -m torch.distributed.run --master_addr=127.0.0.1 --master_port=29657 --nnodes=1 --nproc_per_node=2 --no-python bash /tmp/score-variants-rank.sh "$arm"; then :; else status=1; fi
done
if python /tmp/profile_core_score_variants.py {shlex.quote(ROOT)} {shlex.quote(output)} /tmp/score-variants.json; then :; else status=1; fi
exit "$status"
"""
    return {
        "version": "v2",
        "description": "Core EP2 parent vs runtime-row SwiGLU: retained batches5..9, fresh isolated arms, no updates",
        "tasks": [
            {
                "name": "core-score-variants",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 2, "memory": "256 GiB", "sharedMemory": "32 GiB"},
                "context": {"priority": "urgent", "minRuntime": "30m", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "90m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, output=args.output), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="core-score-variants-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
