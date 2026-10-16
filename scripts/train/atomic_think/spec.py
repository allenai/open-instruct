"""Render an allocated urgent two-node H077 job; submission is separate."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=(
            "gate",
            "gate-repeat",
            "control",
            "rounding-only",
            "atomic-safe",
            "atomic-historical",
            "control-repeat",
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--minutes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 0 < args.minutes <= 120
    assert all(c.isalnum() or c in "-_" for c in args.run_id)
    root = f"/weka/oe-training-default/robertb/atomic-think-h077/{args.run_id}"
    spec = {
        "version": "v2",
        "description": f"H077 {args.arm}: historical SFT runtime; FP32 master-rounding audit",
        "tasks": [
            {
                "name": "train",
                "image": {"beaker": args.image},
                "replicas": 2,
                "leaderSelection": True,
                "command": ["python", "scripts/train/atomic_think/run.py"],
                "arguments": [args.arm, root],
                "envVars": [
                    {"name": "PYTHONPATH", "value": "/stage"},
                    {"name": "NCCL_DEBUG", "value": "ERROR"},
                    {"name": "OLMO_SHARED_FS", "value": "1"},
                    {"name": "HF_HOME", "value": "/weka/oe-adapt-default/allennlp/.cache/huggingface"},
                    {"name": "HF_DATASETS_CACHE", "value": "/weka/oe-adapt-default/allennlp/.cache/huggingface"},
                    {"name": "HF_HUB_CACHE", "value": "/weka/oe-adapt-default/allennlp/.cache/hub"},
                    {"name": "NCCL_SOCKET_IFNAME", "value": "ib"},
                    {"name": "NCCL_IB_HCA", "value": "^=mlx5_bond_0"},
                    {"name": "HF_TOKEN", "secret": "robertb_HF_TOKEN"},
                    {"name": "WANDB_API_KEY", "secret": "robertb_WANDB_API_KEY"},
                    {"name": "BEAKER_TOKEN", "secret": "robertb_BEAKER_TOKEN"},
                ],
                "datasets": [
                    {"mountPath": f"/weka/{name}", "source": {"weka": name}}
                    for name in ("oe-adapt-default", "oe-training-default")
                ],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 8, "sharedMemory": "10240 MB"},
                "context": {"priority": "urgent", "minRuntime": f"{args.minutes}m", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "hostNetworking": True,
                "propagateFailure": True,
                "propagatePreemption": True,
                "synchronizedStartTimeout": "10m",
                "timeout": f"{args.minutes}m",
            }
        ],
        "retry": {"allowedTaskRetries": 0},
    }
    args.output.write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
