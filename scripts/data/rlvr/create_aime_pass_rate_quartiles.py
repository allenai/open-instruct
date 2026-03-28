"""Create quartile splits for AIME pass-rate datasets, preserving year-specific splits.

Example:
HF_TOKEN=... uv run python scripts/data/rlvr/create_aime_pass_rate_quartiles.py \
  --input-dataset <entity>/aime2024-25-rlvr-olmo3-7b-base-pass32 \
  --splits test_2024 test_2025 \
  --output-dataset <entity>/aime2024-25-rlvr-olmo3-7b-base-pass32-quartiles
"""

import argparse

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from open_instruct import logger_utils
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pass-count quartile splits for AIME splits.")
    parser.add_argument(
        "--input-dataset",
        required=True,
        help="Dataset repo containing pass_count/num_samples or pass_rate columns.",
    )
    parser.add_argument("--splits", nargs="+", default=["test_2024", "test_2025"])
    parser.add_argument(
        "--output-dataset",
        required=True,
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


def split_to_dataset_name(split_name: str, quartile_id: int) -> str:
    suffix = split_name.removeprefix("test_")
    return f"math_aime_{suffix}_quartile{quartile_id}"


def with_dataset_name(ds: Dataset, dataset_name: str) -> Dataset:
    if "dataset" in ds.column_names:
        ds = ds.remove_columns(["dataset"])
    return ds.add_column("dataset", [dataset_name] * len(ds))


def push_quartiles_for_split(input_dataset: str, split_name: str, output_dataset: str, private: bool) -> None:
    logger.info("Loading %s[%s]", input_dataset, split_name)
    ds = load_dataset(input_dataset, split=split_name, num_proc=max_num_processes())

    rows = ds.to_list()
    pass_counts = np.array([extract_pass_count(row) for row in rows], dtype=int)
    indices = np.arange(len(ds), dtype=int)
    sorted_indices = indices[np.lexsort((indices, -pass_counts))]
    quartile_indices = np.array_split(sorted_indices, 4)

    quartile_outputs: list[tuple[str, Dataset]] = []
    for quartile_id, quartile_idx in enumerate(quartile_indices):
        quartile_ds = with_dataset_name(ds.select(quartile_idx.tolist()), split_to_dataset_name(split_name, quartile_id))
        quartile_split_name = f"{split_name}_{quartile_id}"
        quartile_outputs.append((quartile_split_name, quartile_ds))
        logger.info(
            "Prepared %s: %d rows, pass_count range [%d, %d]",
            quartile_split_name,
            len(quartile_ds),
            int(np.min(pass_counts[quartile_idx])),
            int(np.max(pass_counts[quartile_idx])),
        )

    for quartile_split_name, quartile_ds in quartile_outputs:
        logger.info("Pushing split %s to %s", quartile_split_name, output_dataset)
        quartile_ds.push_to_hub(output_dataset, split=quartile_split_name, private=private)

    concatenated = concatenate_datasets([quartile_ds for _, quartile_ds in quartile_outputs])
    logger.info("Pushing concatenated split %s (%d rows) to %s", split_name, len(concatenated), output_dataset)
    concatenated.push_to_hub(output_dataset, split=split_name, private=private)


def main() -> None:
    args = parse_args()
    for split_name in args.splits:
        push_quartiles_for_split(args.input_dataset, split_name, args.output_dataset, args.private)
    logger.info("Done")


if __name__ == "__main__":
    main()
