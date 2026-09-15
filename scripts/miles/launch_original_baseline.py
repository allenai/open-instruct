"""Submit the explicitly requested historical Open Instruct comparison.

The --miles image wrapper is reused only for committed-source launch discipline;
these jobs execute the original image's DeepSpeed/vLLM trainer, not MILES.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "/weka/oe-training-default/robertb/open-instruct/checkpoints/olmo3-think-sft-6ff857587e040d6d523a3d5f3a56e918f5401d66"
SOURCE = "/weka/oe-training-default/robertb/open-instruct/runs/olmo3-sft-gsm8k-core-200-32k-r2-20260914/prepared/data"
PREPARED = "/weka/oe-training-default/robertb/open-instruct/data/olmo3-sft-gsm8k-original-retrofit-20260914"


def specification(image, source_dataset, stage, name, *, keep_zero_advantage_groups=False):
    output = "/weka/oe-training-default/robertb/open-instruct/runs/" + name
    args = [
        "python",
        "/tmp/qualification/scripts/miles/original_baseline.py",
        stage,
        "--model",
        MODEL,
        "--source",
        SOURCE,
        "--prepared",
        PREPARED,
        "--output",
        output,
    ]
    if keep_zero_advantage_groups:
        args.append("--keep-zero-advantage-groups")
    command = (
        "set -euo pipefail\nmkdir -p /tmp/qualification /output\n"
        "cp /qualification-source/provenance.json /output/\n"
        "tar -xf /qualification-source/source.tar -C /tmp/qualification\n"
        "cd /stage\n" + shlex.join(args) + " 2>&1 | tee /output/run.log\n"
    )
    env = {
        "PYTHONPATH": "/stage",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "HF_HOME": "/tmp/hf-cache",
        "WANDB_RUN_GROUP": "olmo3-sft-learning-confidence-20260914",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    task = {
        "name": name,
        "image": {"beaker": image},
        "command": ["bash", "-c"],
        "arguments": [command],
        "datasets": [
            {"mountPath": "/qualification-source", "source": {"beaker": source_dataset}},
            {"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}},
            {"mountPath": "/weka/oe-adapt-default", "source": {"weka": "oe-adapt-default"}},
        ],
        "result": {"path": "/output"},
        "resources": {
            "gpuCount": 0 if stage == "prepare" else 8,
            "cpuCount": 16 if stage == "prepare" else 48,
            "memory": "64 GiB" if stage == "prepare" else "704 GiB",
            "sharedMemory": "200 GiB" if stage != "prepare" else "4 GiB",
        },
        "constraints": {"cluster": ["ai2/saturn" if stage == "prepare" else "ai2/jupiter"]},
        "context": {"priority": "urgent", "minRuntime": "30m" if stage != "train" else "4h", "autoResume": False},
        "timeout": "1h" if stage == "prepare" else ("3h" if stage == "smoke" else "48h"),
        "envVars": [{"name": k, "value": v} for k, v in env.items()],
    }
    if stage != "prepare":
        task["envVars"].append({"name": "WANDB_API_KEY", "secret": "robertb_WANDB_API_KEY"})
    return {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": "Original Open Instruct Olmo 3 SFT GSM8K control: " + stage,
        "tasks": [task],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--stage", choices=("prepare", "smoke", "train"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--keep-zero-advantage-groups", action="store_true")
    args = parser.parse_args()
    if args.image != "01KA3FGCMVYGVEX2NG7Q2JWZ8E":
        raise ValueError("Use the qualified historical Think image 01KA3FGCMVYGVEX2NG7Q2JWZ8E")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Commit changes before launching")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="original-baseline-source-") as tmp:
        directory = Path(tmp)
        raw = subprocess.check_output(["git", "archive", "HEAD", "scripts/miles/original_baseline.py"], cwd=ROOT)
        (directory / "source.tar").write_bytes(raw)
        provenance = {
            "commit": commit,
            "base_image": args.image,
            "stage": args.stage,
            "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "trainer": "original-open-instruct",
            "keep_zero_advantage_groups": args.keep_zero_advantage_groups,
        }
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
        source_dataset = "SOURCE_DATASET"
        if not args.render_only:
            name = "original-baseline-source-" + uuid.uuid4().hex[:10]
            subprocess.run(
                [
                    "beaker",
                    "dataset",
                    "create",
                    str(directory),
                    "--workspace",
                    "ai2/open-instruct-dev",
                    "--budget",
                    "ai2/oe-other",
                    "--name",
                    name,
                ],
                check=True,
            )
            user = json.loads(subprocess.check_output(["beaker", "account", "whoami", "--format", "json"], text=True))[
                0
            ]["name"]
            source_dataset = json.loads(
                subprocess.check_output(["beaker", "dataset", "get", f"{user}/{name}", "--format", "json"], text=True)
            )[0]["id"]
        spec = specification(
            args.image,
            source_dataset,
            args.stage,
            args.name,
            keep_zero_advantage_groups=args.keep_zero_advantage_groups,
        )
        if args.render_only:
            print(json.dumps(spec, indent=2))
            return
        path = directory / "experiment.json"
        path.write_text(json.dumps(spec))
        response = json.loads(
            subprocess.check_output(
                [
                    "beaker",
                    "experiment",
                    "create",
                    str(path),
                    "--workspace",
                    "ai2/open-instruct-dev",
                    "--format",
                    "json",
                ],
                text=True,
            )
        )[0]
        print(
            json.dumps(
                {"experiment": response["id"], "source_dataset": source_dataset, "name": args.name, **provenance}
            )
        )


if __name__ == "__main__":
    main()
