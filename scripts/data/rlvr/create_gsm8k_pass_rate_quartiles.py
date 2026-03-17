"""Create one GSM8K dataset repo with quartile splits ordered by pass_count.

Example:
HF_TOKEN=... uv run python scripts/data/rlvr/create_gsm8k_pass_rate_quartiles.py \
  --input-dataset mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-userprompt \
  --split test \
  --output-dataset mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-userprompt-quartiles
"""

import argparse

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from open_instruct import logger_utils
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pass-count quartile splits and push as one dataset repo.")
    parser.add_argument(
        "--input-dataset",
        default="mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-userprompt",
        help="Dataset with pass_count/num_samples or pass_rate columns.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-dataset",
        default="mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-userprompt-quartiles",
        help="Target HF dataset repo to push to.",
    )
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def extract_pass_count(row: dict) -> int:
    pass_count = row.get("pass_count")
    if isinstance(pass_count, int):
        return int(pass_count)

    pass_rate = row.get("pass_rate")
    if isinstance(pass_rate, str) and "/" in pass_rate:
        left, _ = pass_rate.split("/", 1)
        return int(left.strip())

    raise ValueError("Row is missing pass_count and parseable pass_rate")


def with_dataset_name(ds: Dataset, dataset_name: str) -> Dataset:
    if "dataset" in ds.column_names:
        ds = ds.remove_columns(["dataset"])
    return ds.add_column("dataset", [dataset_name] * len(ds))


def main() -> None:
    args = parse_args()

    logger.info("Loading %s[%s]", args.input_dataset, args.split)
    ds = load_dataset(args.input_dataset, split=args.split, num_proc=max_num_processes())

    rows = ds.to_list()
    pass_counts = np.array([extract_pass_count(row) for row in rows], dtype=int)
    indices = np.arange(len(ds), dtype=int)
    # Highest pass_count first; preserve original order for ties.
    sorted_indices = indices[np.lexsort((indices, -pass_counts))]
    quartile_indices = np.array_split(sorted_indices, 4)

    quartile_outputs: list[tuple[str, Dataset]] = []
    for quartile_id, quartile_idx in enumerate(quartile_indices):
        quartile_ds = with_dataset_name(ds.select(quartile_idx.tolist()), f"gsm8k_quartile{quartile_id}")
        split_name = f"test_{quartile_id}"
        quartile_outputs.append((split_name, quartile_ds))
        logger.info(
            "Prepared %s: %d rows, pass_count range [%d, %d]",
            split_name,
            len(quartile_ds),
            int(np.min(pass_counts[quartile_idx])),
            int(np.max(pass_counts[quartile_idx])),
        )

    for split_name, quartile_ds in quartile_outputs:
        logger.info("Pushing split %s to %s", split_name, args.output_dataset)
        quartile_ds.push_to_hub(args.output_dataset, split=split_name, private=args.private)

    concatenated = concatenate_datasets([quartile_ds for _, quartile_ds in quartile_outputs])
    logger.info("Pushing concatenated split test (%d rows) to %s", len(concatenated), args.output_dataset)
    concatenated.push_to_hub(args.output_dataset, split="test", private=args.private)

    logger.info("Done")


if __name__ == "__main__":
    main()
