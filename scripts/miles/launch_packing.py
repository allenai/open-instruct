"""Small fixed-input packing gate: KDA, full attention, latent MoE, EP and replay."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

COMMAND = """set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 NCCL_CUMEM_ENABLE=1
mkdir -p /output
python tests/miles/packing_contract.py bootstrap /output/packing
for world in 1 2; do
  torchrun --standalone --nproc-per-node=$world tests/miles/packing_contract.py run /output/packing
  torchrun --standalone --nproc-per-node=$world tests/miles/packing_contract.py run /output/packing --checkpointing
done
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    args = parser.parse_args()
    document = dict(
        version="v2",
        budget="ai2/oe-other",
        description="Core packed versus unpacked KDA/latent MoE: EP1/EP2, replay, recomputation and gradients",
        tasks=[
            dict(
                name="core-packing",
                image={"beaker": args.image},
                command=["bash", "-c"],
                arguments=[COMMAND],
                result={"path": "/output"},
                resources={"gpuCount": 2, "sharedMemory": "16 GiB"},
                constraints={"cluster": ["ai2/holmes"]},
                context={"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                timeout="90m",
            )
        ],
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "spec.json"
        path.write_text(json.dumps(document))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
