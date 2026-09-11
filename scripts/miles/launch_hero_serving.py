"""Launch a bounded TP1 hero HF/SGLang serving check on Holmes."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.launch_hero_conversion import HF


def specification(image, *, hf=HF, diagnose_core=False):
    tool = "diagnose_hero_core.py" if diagnose_core else "qualify_hero_serving.py"
    output = "diagnosis.json" if diagnose_core else "serving.json"
    extra = (
        "--reference-report /reference/serving.json --save-activations /output/hf-activations.pt"
        if diagnose_core
        else "--core-reference --core-logprob-atol 0.1"
    )
    command = f"""set -euo pipefail
cd /opt/core-rl
mkdir -p /output
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/hf-cache
export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models
cp /opt/core-rl/sources/runtime.lock.json /output/
python -c 'import torch; assert "B300" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name())'
python /opt/core-rl/sources/olmo-sglang/tools/{tool} --model {shlex.quote(hf)} --output /output/{output} --recurrent-hf-prefill --logprob-atol 0.1 {extra}
"""
    return {
        "version": "v2",
        "description": (
            "Hero layerwise HF/Core diagnosis after failed probability gate; no training"
            if diagnose_core
            else "Hero TP1 HF/SGLang short prefill/decode/graph parity; no training"
        ),
        "tasks": [
            {
                "name": "hero-serving",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/olmo-3p5-checkpoints", "source": {"weka": "olmo-3p5-checkpoints"}}]
                + (
                    [{"mountPath": "/reference", "source": {"beaker": "01M26ZBDKYP9E2N7QC7NXN1B09"}}]
                    if diagnose_core
                    else []
                ),
                "result": {"path": "/output"},
                "resources": {"gpuCount": 1, "sharedMemory": "32 GiB"},
                "context": {"priority": "urgent", "minRuntime": "30m", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "45m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--hf", default=HF)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--diagnose-core", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, hf=args.hf, diagnose_core=args.diagnose_core), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-hero-serving-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
