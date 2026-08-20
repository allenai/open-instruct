"""Convert an AppWorld split into an open-instruct RL (RLVR) dataset.

Produces one row per AppWorld task with the columns the GRPO pipeline routes on:

    messages     : [system (with supervisor creds), user (task instruction)]
    ground_truth : task_id (the AppWorldEnv verifies internally via world.evaluate)
    dataset      : "passthrough" (no extra verifier; the env emits the reward)
    tools        : ["execute_python"] (the tool *call name*; selects the schema injected
                   into the prompt and gates dispatch — pair with --tool_call_names execute_python)
    env_config   : {"env_configs": [{"env_name": "appworld", "task_id": ...}], "max_steps": N}

Reads AppWorld task data straight off disk (split file + per-task ``specs.json``) so it
does NOT import the (pydantic-1) ``appworld`` package — it runs in the open-instruct venv.
Point --data_root at an AppWorld data root (the dir containing ``data/tasks`` and
``data/datasets``), e.g. one produced by ``appworld download data``.

Example:
    uv run python scripts/data/convert_appworld_to_rl.py \
        --data_root /weka/.../appworld_root --splits train,dev,test_normal,test_challenge --private \
        --push_to_hub <org>/appworld-train-rl
"""

import argparse
import json
import os

from datasets import Dataset, DatasetDict

from open_instruct import logger_utils
from open_instruct.environments.appworld_env import build_prompt_messages

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, help="AppWorld data root (contains data/tasks, data/datasets).")
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated split files under data/datasets (e.g. train,dev,test_normal,test_challenge). "
        "Each becomes a named split in the pushed dataset.",
    )
    parser.add_argument("--max_steps", type=int, default=50, help="Per-rollout interaction budget (env max_steps).")
    parser.add_argument("--push_to_hub", default="", help="HF repo id to push to (optional).")
    parser.add_argument("--private", action="store_true", help="Push as a private dataset (AppWorld license).")
    parser.add_argument("--output_parquet", default="", help="Local parquet path for the first split (optional).")
    parser.add_argument("--limit", type=int, default=0, help="Keep only the first N tasks per split (0 = all).")
    return parser.parse_args()


def load_task_ids(data_root: str, split: str) -> list[str]:
    split_path = os.path.join(data_root, "data", "datasets", f"{split}.txt")
    with open(split_path, encoding="utf-8") as fh:
        # Lines may carry an optional "tag:task_id" prefix; keep the task_id.
        return [line.strip().split(":")[-1] for line in fh if line.strip()]


def build_row(data_root: str, task_id: str, max_steps: int) -> dict:
    specs_path = os.path.join(data_root, "data", "tasks", task_id, "specs.json")
    with open(specs_path, encoding="utf-8") as fh:
        specs = json.load(fh)
    return {
        "messages": build_prompt_messages(specs["instruction"], specs["supervisor"]),
        "ground_truth": task_id,
        "dataset": "passthrough",
        "tools": ["execute_python"],
        # env_name must match the pool key, which is the *call name* (execute_python),
        # not the registry config_name (appworld) — otherwise tool discovery creates a
        # duplicate pool and per-row task_id routing misses. Pair with:
        #   --tools appworld --tool_call_names execute_python
        "env_config": {"env_configs": [{"env_name": "execute_python", "task_id": task_id}], "max_steps": max_steps},
    }


def build_split(data_root: str, split: str, max_steps: int, limit: int) -> Dataset:
    task_ids = load_task_ids(data_root, split)
    if limit:
        task_ids = task_ids[:limit]
    logger.info(f"Building AppWorld split '{split}': {len(task_ids)} tasks.")
    return Dataset.from_list([build_row(data_root, task_id, max_steps) for task_id in task_ids])


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    dataset_dict = DatasetDict({split: build_split(args.data_root, split, args.max_steps, args.limit) for split in splits})
    logger.info(f"Built DatasetDict with splits {list(dataset_dict)}; columns {dataset_dict[splits[0]].column_names}")

    if args.output_parquet:
        dataset_dict[splits[0]].to_parquet(args.output_parquet)
        logger.info(f"Wrote {args.output_parquet} (split '{splits[0]}')")
    if args.push_to_hub:
        dataset_dict.push_to_hub(args.push_to_hub, private=args.private)
        logger.info(f"Pushed to hub: {args.push_to_hub} (private={args.private})")
    if not args.output_parquet and not args.push_to_hub:
        logger.warning("Neither --output_parquet nor --push_to_hub set; dataset was built but not saved.")


if __name__ == "__main__":
    main()
