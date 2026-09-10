"""Two-GPU native EP contract comparison using random local fixtures only."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

COMMAND = """set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled
export NCCL_CUMEM_ENABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
mkdir -p /output
python tests/miles/ep_contract.py bootstrap /tmp/contract
for world in 1 2; do
  for mode in policy auxiliary combined; do
    torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=$world tests/miles/ep_contract.py run /tmp/contract --mode $mode
    torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=$world tests/miles/ep_contract.py run /tmp/contract --mode $mode --checkpointing
  done
done
python tests/miles/ep_contract.py compare /tmp/contract
cp /tmp/contract/ep-contract.json /output/
for directory in /tmp/contract/ep*-ac*; do
  if [ -d "$directory" ]; then
    mkdir -p "/output/$(basename "$directory")"
    cp "$directory"/training_contract_rank*.jsonl "/output/$(basename "$directory")/"
  fi
done
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    spec = dict(
        version="v2",
        description="Core RL contract: matched EP1/EP2 moments, replay, auxiliary objectives and recomputation",
        tasks=[
            dict(
                name="core-contract",
                image={"beaker": args.image},
                command=["bash", "-c"],
                arguments=[COMMAND],
                result={"path": "/output"},
                resources={"gpuCount": 2, "sharedMemory": "16 GiB"},
                constraints={"cluster": ["ai2/holmes"]},
                context={"priority": "normal", "minRuntime": "20m", "autoResume": False},
                timeout="35m",
            )
        ],
    )
    document = json.dumps(spec, indent=2) + "\n"
    if args.render_only:
        print(document)
        return
    with tempfile.TemporaryDirectory(prefix="core-contract-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
