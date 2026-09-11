"""Launch a bounded TP1 hero HF/SGLang serving check on Holmes."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from scripts.miles.launch_hero_conversion import HF


def specification(
    image,
    *,
    hf=HF,
    diagnose_core=False,
    mode_matrix=False,
    hf_core_moe_reference=False,
    diagnose_moe=False,
    hf_attention="eager",
):
    if hf_attention not in ("eager", "sdpa") or (hf_attention != "eager" and not diagnose_core):
        raise ValueError("HF attention control requires a layerwise diagnosis")
    if hf_core_moe_reference and not diagnose_core:
        raise ValueError("The Core-compatible HF MoE control is only a layerwise diagnostic")
    if sum((diagnose_core, mode_matrix, diagnose_moe)) > 1:
        raise ValueError("Choose one diagnostic at a time")
    tool = "diagnose_hero_core.py" if diagnose_core else "qualify_hero_serving.py"
    output = "diagnosis.json" if diagnose_core else "serving.json"
    extra = (
        "--reference-report /reference/serving.json --save-activations /output/hf-activations.pt"
        if diagnose_core
        else "--core-reference --core-logprob-atol 0.1"
    )
    if mode_matrix:
        extra = "--diagnostic-mode-matrix"
    if diagnose_core:
        extra += " --hf-attention " + hf_attention
    reference_dataset = "01M26ZBDKYP9E2N7QC7NXN1B09" if diagnose_core else None
    common_options = "--recurrent-hf-prefill --logprob-atol 0.1"
    if diagnose_moe:
        tool, output = "diagnose_hero_moe.py", "moe-operators.json"
        common_options = ""
        extra = "--activations /reference/hf-activations.pt --reference-report /reference/diagnosis.json --layer 1"
        reference_dataset = "01M270S3P7WY092EG706RZ9H21"
    reference_environment = "export OLMO_HF_MOE_CORE_REFERENCE=1" if hf_core_moe_reference else ""
    command = f"""set -euo pipefail
cd /opt/core-rl
mkdir -p /output
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/hf-cache
export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models
{reference_environment}
cp /opt/core-rl/sources/runtime.lock.json /output/
python -c 'import torch; assert "B300" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name())'
python /opt/core-rl/sources/olmo-sglang/tools/{tool} --model {shlex.quote(hf)} --output /output/{output} {common_options} {extra}
"""
    return {
        "version": "v2",
        "description": (
            "Hero isolated MoE operator diagnosis: actual no-grad and grad-enabled forwards; no optimizer"
            if diagnose_moe
            else (
                "Hero layerwise Core-compatible HF MoE control; diagnosis only"
                if hf_core_moe_reference
                else "Hero layerwise HF/Core diagnosis after failed probability gate; no training"
            )
            if diagnose_core
            else "Hero SGLang graph/chunk matrix after failed probability gate; no training"
            if mode_matrix
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
                    [{"mountPath": "/reference", "source": {"beaker": reference_dataset}}] if reference_dataset else []
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
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--diagnose-core", action="store_true")
    diagnostics.add_argument("--mode-matrix", action="store_true")
    diagnostics.add_argument("--diagnose-moe", action="store_true")
    parser.add_argument("--hf-core-moe-reference", action="store_true")
    parser.add_argument("--hf-attention", choices=("eager", "sdpa"), default="eager")
    args = parser.parse_args()
    document = (
        json.dumps(
            specification(
                args.image,
                hf=args.hf,
                diagnose_core=args.diagnose_core,
                mode_matrix=args.mode_matrix,
                diagnose_moe=args.diagnose_moe,
                hf_core_moe_reference=args.hf_core_moe_reference,
                hf_attention=args.hf_attention,
            ),
            indent=2,
        )
        + "\n"
    )
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
