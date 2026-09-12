"""Submit the full GSM8K test evaluation of a run's start and end checkpoints on eight Holmes GPUs."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

RUN_ROOT = "/weka/oe-training-default/robertb/open-instruct/runs/two-node-async-gsm8k-20260912-r2"
START = (
    "/weka/oe-training-default/robertb/olmo-miles/checkpoints/"
    "olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf"
)


def specification(
    image, *, run_root=RUN_ROOT, start=START, campaign="gsm8k-test-eval-20260912", sampled_n=8, limit=None
):
    output = f"{run_root}/{campaign}"
    env = {
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "olmo_sglang.models",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "/tmp/hf-cache",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        "python",
        "scripts/miles/gsm8k_test_eval.py",
        "--gpus",
        "8",
        "--output",
        "/output",
        "--sampled-n",
        str(sampled_n),
        "--reference-eval",
        f"{run_root}/prepared/data/eval.jsonl",
        "--checkpoint",
        f"start={start}",
        "--checkpoint",
        f"update100={run_root}/export-hf",
    ]
    if limit:
        command += ["--limit", str(limit)]
    lines = [
        "set -euo pipefail",
        "cd /opt/core-rl",
        f"test ! -e {shlex.quote(output)}",
        *[f"export {name}={shlex.quote(value)}" for name, value in env.items()],
        'export PYTHONPATH="/opt/core-rl:${PYTHONPATH:-}"',
        f"trap 'mkdir -p {shlex.quote(output)} && cp -r /output/. {shlex.quote(output)}/ || true' EXIT",
        "python -c 'import torch; assert \"B300\" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name())'",
        shlex.join(command),
    ]
    return {
        "version": "v2",
        "description": "Full GSM8K test (1319 questions) greedy and sampled scoring of the two-node run's start and update-100 checkpoints",
        "tasks": [
            {
                "name": "gsm8k-test-eval",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": ["\n".join(lines) + "\n"],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 8, "sharedMemory": "100 GiB"},
                "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "4h",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--campaign", default="gsm8k-test-eval-20260912")
    parser.add_argument("--sampled-n", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--render-only", action="store_true")
    options = parser.parse_args()
    spec = specification(options.image, campaign=options.campaign, sampled_n=options.sampled_n, limit=options.limit)
    if options.render_only:
        print(json.dumps(spec, indent=2))
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
        path = Path(handle.name)
    result = subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            str(path),
            "--workspace",
            "ai2/open-instruct-dev",
            "--name",
            options.campaign,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
