"""Embed a diagnostic worker in the immutable original Core100 runtime."""

import argparse
import base64
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

IMAGE = "01M26N80T0V9PREQTS87J849P8"
ROOT = "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1"
OUTPUT = ROOT + "/score-profile-20260911-v1"


def encoded(value):
    return base64.b64encode(value.encode()).decode()


def specification(image, *, output=OUTPUT, rollout=5):
    if image != IMAGE:
        raise ValueError("Use the immutable original Core100 image")
    if Path(output).parent != Path(ROOT) or not re.fullmatch(r"score-profile-[A-Za-z0-9._-]+", Path(output).name):
        raise ValueError("Use a distinct score-profile directory beneath the original campaign")
    if rollout < 0 or rollout >= 100:
        raise ValueError("Choose a retained training rollout in0..99")
    worker = Path(__file__).with_name("profile_frozen_core_scores.py").read_text()
    manifest = json.loads(Path(__file__).with_name("frozen_core_score_manifest.json").read_text())
    manifest["worker_sha256"] = hashlib.sha256(worker.encode()).hexdigest()
    rank_script = f"""#!/bin/bash
set -euo pipefail
export TRITON_CACHE_DIR=/tmp/frozen-score-cache/rank${{RANK}}/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/frozen-score-cache/rank${{RANK}}/inductor
exec python /tmp/profile_frozen_core_scores.py {shlex.quote(ROOT)} {shlex.quote(output)} /tmp/score-manifest.json --rollout {rollout}
"""
    command = f"""set -euo pipefail
cd /opt/core-rl
test ! -e {shlex.quote(output)}
mkdir -p {shlex.quote(output)} /output
printf %s {shlex.quote(encoded(worker))} | base64 -d > /tmp/profile_frozen_core_scores.py
printf %s {shlex.quote(encoded(json.dumps(manifest)))} | base64 -d > /tmp/score-manifest.json
printf %s {shlex.quote(encoded(rank_script))} | base64 -d > /tmp/score-rank.sh
cp /tmp/score-manifest.json {shlex.quote(output)}/manifest.json
trap 'cp {shlex.quote(output)}/rank[0-9].json /output/ 2>/dev/null || true; cp /tmp/score-manifest.json /output/' EXIT
export NCCL_CUMEM_ENABLE=1 TOKENIZERS_PARALLELISM=false HF_HOME=/tmp/hf-cache WANDB_MODE=disabled
python -m torch.distributed.run --master_addr=127.0.0.1 --master_port=29657 --nnodes=1 --nproc_per_node=2 --no-python bash /tmp/score-rank.sh
"""
    return {
        "version": "v2",
        "description": "Frozen Core100 EP2 scorer: first batch +3 repeats +separate warm profiler; no updates",
        "tasks": [
            {
                "name": "frozen-core-score-profile",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 2, "memory": "256 GiB", "sharedMemory": "32 GiB"},
                "context": {"priority": "urgent", "minRuntime": "30m", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "60m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--rollout", type=int, default=5)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, output=args.output, rollout=args.rollout), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="frozen-score-profile-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
