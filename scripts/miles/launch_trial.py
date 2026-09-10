"""Submit a bounded public-input MoE trial through the repository image wrapper."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

COMMAND = """set -euo pipefail
cd /opt/core-rl
export HF_HOME=/tmp/hf-cache
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=1
mkdir -p /output
python -c 'import torch; assert torch.cuda.is_available(), "Pinned runtime requires a CUDA 13-compatible driver"; print(torch.cuda.get_device_name(), torch.version.cuda)'
python tests/miles/local_moe.py bootstrap /output/toy-moe
python tests/miles/local_moe.py run /output/toy-moe
python tests/miles/local_moe.py run /output/toy-moe --resume
python tests/miles/local_moe.py audit /output/toy-moe
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--cluster", default="ai2/holmes")
    parser.add_argument("--workspace", default="ai2/open-instruct-dev")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--disaggregated", action="store_true")
    args = parser.parse_args()
    command = COMMAND
    if args.disaggregated:
        command = (
            COMMAND.split("python tests/miles/local_moe.py bootstrap")[0]
            + """
python tests/miles/local_moe.py bootstrap /output/input
for mode in baseline streaming; do
  mkdir -p /output/$mode
  cp -r /output/input/. /output/$mode/
  flags="--disaggregated"
  if [ "$mode" = baseline ]; then flags="$flags --legacy-export --per-tensor"; fi
  python tests/miles/local_moe.py run /output/$mode $flags
  python tests/miles/local_moe.py run /output/$mode $flags --resume
  python tests/miles/local_moe.py audit /output/$mode
done
"""
        )
    spec = {
        "version": "v2",
        "description": "MILES/Core toy MoE: public GSM8K, weight publication, native restart, independent audit",
        "tasks": [
            {
                "name": "toy-moe-gsm8k",
                "image": {"beaker": args.image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 2 if args.disaggregated else 1, "sharedMemory": "8 GiB"},
                "context": {"priority": "normal", "minRuntime": "15m", "autoResume": False},
                "constraints": {"cluster": [args.cluster]},
                "timeout": "25m" if args.disaggregated else "20m",
            }
        ],
    }
    document = json.dumps(spec, indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-core-trial-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", args.workspace, "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
