"""Run a native Olmicro smoke in two separate training processes and prove resume."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def checkpoint(root, update):
    checkpoints = root / "checkpoints"
    rollout = update - 1
    manifest = json.loads((checkpoints / "core" / f"rollout_{rollout:07d}" / "complete.json").read_text())
    cursor = checkpoints / "rollout" / f"global_dataset_state_dict_{rollout}.pt"
    if hashlib.sha256(cursor.read_bytes()).hexdigest() != manifest["cursor_sha256"]:
        raise RuntimeError("Smoke checkpoint cursor differs from its committed manifest")
    if manifest["clock"]["completed_steps"] != update or manifest["world_size"] != 2:
        raise RuntimeError(f"Unexpected native checkpoint boundary: {manifest['clock']}")
    return manifest


def main():
    config = Path(sys.argv[1])
    document = json.loads(config.read_text())
    if document["training"]["num_rollouts"] != 3 or document["miles"]["debug_exit_after_rollout"] != 2:
        raise ValueError("Native smoke expects three updates and a debug stop after two")
    root = Path(document["output"]["root"])
    # Only the small evaluation slice is changed; training uses the real GSM8K data.
    source = Path(document["launch"]["env"]["OI_NATIVE_SMOKE_EVAL_SOURCE"])
    target = Path(document["data"]["eval_prompt_data"][1])
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = source.read_text().splitlines()
    target.write_text("\n".join(rows[:8]) + "\n")
    for attempt in range(2):
        subprocess.run([sys.executable, "-m", "open_instruct.miles", "train", str(config)], check=True)
        state = json.loads((root / "workflow.json").read_text())
        result = state["result"]
        if attempt == 0:
            checkpoint(root, 2)
            if state["status"] != "stopped" or result["completed_rollout_ids"] != [0, 1]:
                raise RuntimeError(f"First smoke process did not stop at checkpoint two: {state}")
            logger.info("NATIVE_SMOKE_CHECKPOINT_TWO_COMMITTED; starting separate resume process")
        else:
            manifest = checkpoint(root, 3)
            if (
                state["status"] != "complete"
                or result["start_rollout_id"] != 2
                or result["completed_rollout_ids"] != [2]
            ):
                raise RuntimeError(f"Second smoke process did not restore and advance: {state}")
            receipt = {"status": "passed", "restored_update": 2, "completed_updates": 3, "clock": manifest["clock"]}
            (root / "native-smoke-complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
            logger.info("NATIVE_SMOKE_RESUME_PASSED %s", receipt)


if __name__ == "__main__":
    main()
