"""Submit the full-model publication profile on three Holmes B300 GPUs."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1"


def specification(image, *, campaign="publication-profile-20260912-v1"):
    output = ROOT + "/" + campaign
    env = {
        "OI_PUBLICATION_PROFILE_ROOT": ROOT,
        "OI_PUBLICATION_PROFILE_OUTPUT": output,
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "olmo_sglang.models",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_CUMEM_ENABLE": "1",
        # Record which transport NCCL chose for every communicator, including the
        # trainer-to-engine weight update group, in per-process files under the result.
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,P2P,SHM,NET,GRAPH",
        "NCCL_DEBUG_FILE": "/output/nccl/%h-%p.log",
        "HF_HOME": "/tmp/hf-cache",
        "PYTHONUNBUFFERED": "1",
    }
    lines = [
        "set -euo pipefail",
        "cd /opt/core-rl",
        f"test ! -e {shlex.quote(output)}",
        "mkdir -p /output/nccl",
        *[f"export {name}={shlex.quote(value)}" for name, value in env.items()],
        'export PYTHONPATH="/opt/core-rl:${PYTHONPATH:-}"',
        "trap 'cp -r "
        + shlex.quote(output)
        + "/profile.json "
        + shlex.quote(output)
        + "/final-weight-comparison.json "
        + shlex.quote(output)
        + "/cleanup.json "
        + shlex.quote(output)
        + "/metrics/publication.jsonl /output/ 2>/dev/null || true' EXIT",
        "python -c 'import torch; assert \"B300\" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name(), torch.version.cuda)'",
        "python scripts/miles/publication_profile.py",
    ]
    return {
        "version": "v2",
        "description": "Full-model Core publication profile: broadcast versus engine load per bucket, bucket-size sweep, NCCL transport",
        "tasks": [
            {
                "name": "publication-profile",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": ["\n".join(lines) + "\n"],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 3, "sharedMemory": "100 GiB"},
                "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "90m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--campaign", default="publication-profile-20260912-v1")
    parser.add_argument("--render-only", action="store_true")
    options = parser.parse_args()
    spec = specification(options.image, campaign=options.campaign)
    if options.render_only:
        print(json.dumps(spec, indent=2))
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
        path = Path(handle.name)
    result = subprocess.run(
        ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)


if __name__ == "__main__":
    main()
