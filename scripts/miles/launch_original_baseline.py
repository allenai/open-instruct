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
# The four-domain basket arm reuses the frozen data the MILES dense g16 run
# prepared (same rendered prompts, labels and held-out identities) and the
# MILES prepared judge (same Qwen3-32B snapshot and no-thinking template).
PROFILES = {
    "gsm8k": {"source": SOURCE, "prepared": PREPARED, "judge_prepared": None},
    "basket": {
        "source": "/weka/oe-training-default/robertb/open-instruct/runs/olmo3-think-sft-basket-200-32k-g16-20260915/prepared/data",
        "prepared": "/weka/oe-training-default/robertb/open-instruct/data/olmo3-think-sft-basket-original-retrofit-20260916",
        "judge_prepared": "/weka/oe-adapt-default/robertb/olmo-miles/trial-data/dolci-think-20260908/judge",
    },
}


def specification(
    image,
    source_dataset,
    stage,
    name,
    *,
    keep_zero_advantage_groups=False,
    checkpoint_root=None,
    checkpoint_tag=None,
    evaluation_model=None,
    profile="gsm8k",
    steps=None,
):
    output = "/weka/oe-training-default/robertb/open-instruct/runs/" + name
    paths = PROFILES[profile]
    args = [
        "python",
        "/tmp/qualification/scripts/miles/original_baseline.py",
        stage,
        "--model",
        evaluation_model or MODEL,
        "--source",
        paths["source"],
        "--prepared",
        paths["prepared"],
        "--output",
        output,
        "--profile",
        profile,
    ]
    if steps is not None:
        args.extend(["--steps", str(steps)])
    if paths["judge_prepared"] and stage in {"train", "smoke", "resume"}:
        args.extend(["--judge-prepared", paths["judge_prepared"]])
    if stage == "export":
        if not checkpoint_root or not checkpoint_tag:
            raise ValueError("export requires checkpoint_root and checkpoint_tag")
        args.extend(["--checkpoint-root", checkpoint_root, "--checkpoint-tag", checkpoint_tag])
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
        "WANDB_RUN_GROUP": (
            "dolci-basket-32k-zero-20260914" if profile == "basket" else "olmo3-sft-learning-confidence-20260914"
        ),
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if stage in {"export", "resume"}:
        # Our own optimizer checkpoints contain trusted Python client state.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
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
            "gpuCount": 0 if stage in {"prepare", "export"} else (1 if stage == "evaluate" else 8),
            "cpuCount": 16 if stage == "prepare" else 48,
            "memory": "256 GiB"
            if stage == "export"
            else ("64 GiB" if stage in {"prepare", "evaluate"} else "704 GiB"),
            "sharedMemory": "200 GiB" if stage in {"train", "smoke", "resume"} else "4 GiB",
        },
        "constraints": {"cluster": ["ai2/saturn" if stage in {"prepare", "export"} else "ai2/jupiter"]},
        "context": {
            "priority": "urgent",
            "minRuntime": "4h" if stage in {"train", "resume"} else "30m",
            "autoResume": stage == "resume",
        },
        "timeout": "1h" if stage in {"prepare", "export"} else ("3h" if stage in {"smoke", "evaluate"} else "48h"),
        "envVars": [{"name": k, "value": v} for k, v in env.items()],
    }
    if stage in {"train", "smoke", "resume"}:
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
    parser.add_argument(
        "--stage", choices=("prepare", "smoke", "train", "export", "evaluate", "resume"), required=True
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--keep-zero-advantage-groups", action="store_true")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--checkpoint-tag")
    parser.add_argument("--evaluation-model", help="Public HF model for independent evaluation")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gsm8k")
    parser.add_argument("--steps", type=int, help="Driver steps for train/resume (default: profile budget)")
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
            "profile": args.profile,
            "steps": args.steps,
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
            checkpoint_root=args.checkpoint_root,
            checkpoint_tag=args.checkpoint_tag,
            evaluation_model=args.evaluation_model,
            profile=args.profile,
            steps=args.steps,
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
