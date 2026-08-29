"""Log the initial dataset-derived pass@k baselines for the AIME quartiles dataset to W&B.

Example:
  uv run python scripts/data/rlvr/log_aime_quartile_baselines_to_wandb.py \
    --dataset mnoukhov/aime2024-25-rlvr-olmo3-7b-base-pass64-quartiles \
    --split test \
    --wandb-project open_instruct \
    --wandb-name aime-quartile-baselines
"""

import argparse
import re
from collections import defaultdict
from typing import Any

import numpy as np
import wandb
from datasets import Dataset, load_dataset

from open_instruct import logger_utils
from open_instruct.grpo_utils import estimate_pass_at_k
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and log the initial pass@k baselines for an AIME quartiles dataset."
    )
    parser.add_argument(
        "--dataset",
        default="mnoukhov/aime2024-25-rlvr-olmo3-7b-base-pass64-quartiles",
        help="HF dataset repo containing the quartile split.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to summarize.")
    parser.add_argument("--wandb-project", default="open_instruct", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity. Defaults to the configured user.")
    parser.add_argument("--wandb-name", default="aime-quartile-baselines", help="W&B run name.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B run group.")
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags.")
    return parser.parse_args()


def _sanitize_metric_suffix(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


def _extract_pass_count(row: dict[str, Any]) -> int:
    pass_count = row.get("pass_count")
    if isinstance(pass_count, (int, np.integer)):
        return int(pass_count)

    pass_rate = row.get("pass_rate")
    if isinstance(pass_rate, str) and "/" in pass_rate:
        left, _ = pass_rate.split("/", 1)
        return int(left.strip())

    raise ValueError("Row is missing pass_count and parseable pass_rate")


def _extract_num_samples(row: dict[str, Any]) -> int:
    num_samples = row.get("num_samples")
    if isinstance(num_samples, (int, np.integer)):
        return int(num_samples)

    raise ValueError("Row is missing num_samples")


def compute_initial_dataset_pass_at_k_metrics(eval_dataset: Dataset | None) -> dict[str, float]:
    if eval_dataset is None or len(eval_dataset) == 0:
        return {}
    required_columns = {"dataset", "num_samples"}
    if not required_columns.issubset(eval_dataset.column_names):
        return {}
    if "pass_count" not in eval_dataset.column_names and "pass_rate" not in eval_dataset.column_names:
        return {}

    rows = eval_dataset.to_list()
    grouped_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    all_rows: list[tuple[int, int]] = []
    for row in rows:
        dataset_name = _sanitize_metric_suffix(str(row["dataset"]))
        pass_count = _extract_pass_count(row)
        num_samples = _extract_num_samples(row)
        grouped_rows[dataset_name].append((pass_count, num_samples))
        all_rows.append((pass_count, num_samples))

    metrics: dict[str, float] = {}

    def add_pass_at_k_metrics(prefix: str, rows_with_counts: list[tuple[int, int]]) -> None:
        min_num_samples = min(num_samples for _, num_samples in rows_with_counts)
        pass_at_ks = [2**i for i in range(min_num_samples.bit_length()) if 2**i <= min_num_samples]
        for k in pass_at_ks:
            pass_at_values = [
                estimate_pass_at_k(num_samples=num_samples, num_correct=pass_count, k=k)
                for pass_count, num_samples in rows_with_counts
            ]
            metrics[f"{prefix}/pass_at_{k}"] = float(np.mean(pass_at_values))
        metrics[f"{prefix}/num_prompts"] = float(len(rows_with_counts))

    add_pass_at_k_metrics("eval", all_rows)
    for dataset_name, rows_with_counts in grouped_rows.items():
        add_pass_at_k_metrics(f"eval/{dataset_name}", rows_with_counts)

    return metrics


def main() -> None:
    args = parse_args()
    logger.info("Loading %s[%s]", args.dataset, args.split)
    eval_dataset = load_dataset(args.dataset, split=args.split, num_proc=max_num_processes())

    metrics = compute_initial_dataset_pass_at_k_metrics(eval_dataset)
    if not metrics:
        raise ValueError("No pass@k metrics could be computed from the dataset.")

    wandb.login()
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config={"dataset": args.dataset, "split": args.split},
    )
    wandb.log(metrics, step=0)
    logger.info("Logged %d metrics to W&B run %s.", len(metrics), run.url)
    wandb.finish()


if __name__ == "__main__":
    main()
