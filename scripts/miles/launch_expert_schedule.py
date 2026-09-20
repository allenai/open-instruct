"""Launch the committed-image four-GPU EP2 expert scheduling qualification."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

COMMAND = """set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 NCCL_CUMEM_ENABLE=1
python -m pytest -q open_instruct/test_miles_expert_search.py tests/miles/test_expert_schedule_runtime.py tests/miles/test_expert_schedule_distributed.py tests/miles/test_packing_loss.py tests/miles/test_options.py
python tests/miles/packing_contract.py bootstrap /output/expert-schedule
for mode in plain recompute; do
  extra=()
  if [ "$mode" = recompute ]; then extra+=(--checkpointing); fi
  torchrun --standalone --nproc-per-node=4 tests/miles/expert_schedule_contract.py /output/expert-schedule "${extra[@]}"
done
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    args = parser.parse_args()
    document = dict(
        version="v2",
        budget="ai2/oe-other",
        description="Replay expert-aware packing: four trainer GPUs, EP2, fixed scores/gradients/Adam, recomputation on/off",
        tasks=[
            dict(
                name="expert-schedule-gate",
                image={"beaker": args.image},
                command=["bash", "-c"],
                arguments=[COMMAND],
                result={"path": "/output"},
                resources={"gpuCount": 4, "sharedMemory": "16 GiB"},
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
