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


def normalize_schema(ds: Dataset) -> Dataset:
    def normalize_row(row: dict) -> dict:
        ground_truth = row.get("ground_truth")
        if isinstance(ground_truth, list):
            if len(ground_truth) != 1:
                raise ValueError(f"Expected exactly one ground_truth value, got {ground_truth}")
            row["ground_truth"] = str(ground_truth[0])
        elif ground_truth is not None:
            row["ground_truth"] = str(ground_truth)

        row.pop("completions", None)
        return row

    ds = ds.map(normalize_row, num_proc=max_num_processes())
    if "completions" in ds.column_names:
        ds = ds.remove_columns(["completions"])
    return ds


def split_to_dataset_name(quartile_id: int) -> str:
    return f"gsm8k_quartile{quartile_id}"


def main() -> None:
    args = parse_args()

    logger.info("Loading %s[%s]", args.input_dataset, args.split)
    ds = load_dataset(args.input_dataset, split=args.split, num_proc=max_num_processes())
    ds = normalize_schema(ds)

    rows = ds.to_list()
    pass_counts = np.array([extract_pass_count(row) for row in rows], dtype=int)
    indices = np.arange(len(ds), dtype=int)
    # Highest pass_count first; preserve original order for ties.
    sorted_indices = indices[np.lexsort((indices, -pass_counts))]
    quartile_indices = np.array_split(sorted_indices, 4)

    quartile_outputs: list[tuple[str, Dataset]] = []
    for quartile_id, quartile_idx in enumerate(quartile_indices):
        quartile_ds = with_dataset_name(ds.select(quartile_idx.tolist()), split_to_dataset_name(quartile_id))
        quartile_split_name = f"{args.split}_{quartile_id}"
        quartile_outputs.append((quartile_split_name, quartile_ds))
        logger.info(
            "Prepared %s: %d rows, pass_count range [%d, %d]",
            quartile_split_name,
            len(quartile_ds),
            int(np.min(pass_counts[quartile_idx])),
            int(np.max(pass_counts[quartile_idx])),
        )

    for quartile_split_name, quartile_ds in quartile_outputs:
        logger.info("Pushing split %s to %s", quartile_split_name, args.output_dataset)
        quartile_ds.push_to_hub(args.output_dataset, split=quartile_split_name, private=args.private)

    concatenated = concatenate_datasets([quartile_ds for _, quartile_ds in quartile_outputs])
    logger.info("Pushing concatenated split %s (%d rows) to %s", args.split, len(concatenated), args.output_dataset)
    concatenated.push_to_hub(args.output_dataset, split=args.split, private=args.private)

    logger.info("Done")


if __name__ == "__main__":
    main()
