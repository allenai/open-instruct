"""Launch exhaustive native/HF conversion checks on a CPU WEKA worker."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

NATIVE = "/weka/olmo-3p5-checkpoints/production-hero-small/olmo35-small-hero-20260907/olmo35-small-hero-20260907-non-emo/step38000"
HF = "/weka/olmo-3p5-checkpoints/scratch/hero-hf-20260909/non-emo/step38000/hf"


def specification(image, *, native=NATIVE, hf=HF, inspect_only=False):
    script = "inspect_hero_checkpoint.py" if inspect_only else "validate_hero_conversion.py"
    report = "inventory.json" if inspect_only else "conversion.json"
    command = f"""set -euo pipefail
cd /opt/core-rl
mkdir -p /output
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export OLMO_USE_TORCH_GROUPED_MM=0
cp /opt/core-rl/sources/runtime.lock.json /output/
python scripts/miles/{script} --native {shlex.quote(native)} --hf {shlex.quote(hf)} --report /output/{report}
"""
    return {
        "version": "v2",
        "description": "Hero native/HF exhaustive weight interchange audit; CPU only, no RL or generation",
        "tasks": [
            {
                "name": "hero-conversion",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/olmo-3p5-checkpoints", "source": {"weka": "olmo-3p5-checkpoints"}}],
                "result": {"path": "/output"},
                "resources": {
                    "cpuCount": 2 if inspect_only else 16,
                    "memory": "4 GiB" if inspect_only else "256 GiB",
                    "gpuCount": 0,
                    "sharedMemory": "1 GiB" if inspect_only else "8 GiB",
                },
                "context": {"priority": "urgent", "minRuntime": "10m" if inspect_only else "30m", "autoResume": False},
                "constraints": {"cluster": ["ai2/saturn"]},
                "timeout": "15m" if inspect_only else "90m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--native", default=NATIVE)
    parser.add_argument("--hf", default=HF)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    document = (
        json.dumps(specification(args.image, native=args.native, hf=args.hf, inspect_only=args.inspect_only), indent=2)
        + "\n"
    )
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-hero-conversion-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
