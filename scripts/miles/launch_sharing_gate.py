"""Qualify a sharing image with runtime tests, dense resume and live EP2 MoE replay."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

COMMAND = """set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 NCCL_CUMEM_ENABLE=1
export PYTHONPATH=/opt/core-rl/tests/miles:$PYTHONPATH
mkdir -p /output
# This optional cross-backend diagnostic requires a separate olmo-miles comparison
# checkout. It is not a dependency of the standalone training product.
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/miles -q \\
  --ignore=tests/miles/test_core_policy_contract.py --junitxml=/output/runtime-tests.xml \\
  2>&1 | tee /output/runtime-tests.log
CUDA_VISIBLE_DEVICES=0 python tests/miles/smoke.py /output/dense --model-type olmo3
CUDA_VISIBLE_DEVICES=0 python tests/miles/smoke.py /output/dense --model-type olmo3 --resume
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \\
  -m pytest tests/miles/test_parameter_probe.py -q 2>&1 | tee /output/fsdp-tests.log
python tests/miles/packing_contract.py bootstrap /output/packing
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \\
  tests/miles/packing_contract.py run /output/packing --checkpointing
python tests/miles/local_moe.py bootstrap /output/moe
python tests/miles/local_moe.py run /output/moe --disaggregated --expert-parallel-size 2 --routing-replay
python tests/miles/local_moe.py run /output/moe --disaggregated --expert-parallel-size 2 --routing-replay --resume
python tests/miles/local_moe.py audit /output/moe
python -c 'import json; from pathlib import Path; Path("/output/complete.json").write_text(json.dumps({"passed": True, "scope": "runtime tests; dense synthetic resume; FSDP diagnostics; EP2 packing; live MoE replay/resume"}))'
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    spec = {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": "MILES/Core sharing candidate: gated tests, dense resume, EP2 packing and live MoE replay/resume",
        "tasks": [
            {
                "name": "sharing-gate",
                "image": {"beaker": args.image},
                "command": ["bash", "-c"],
                "arguments": [COMMAND],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 3, "sharedMemory": "16 GiB"},
                "constraints": {"cluster": ["ai2/holmes"]},
                "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                "timeout": "2h",
            }
        ],
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "spec.json"
        path.write_text(json.dumps(spec))
        result = json.loads(
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
        )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps({"spec": spec, "result": result}, indent=2) + "\n")
    print(json.dumps({"experiment": result[0]["id"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
