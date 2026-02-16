"""Create two GSM8K datasets bucketed by pass@32 rate (e.g., 25% and 75%).

Example:
HF_TOKEN=... uv run python scripts/data/rlvr/create_gsm8k_pass_rate_buckets.py \
  --input-dataset mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct \
  --split test \
  --num-per-bucket 16 \
  --bucket-low 0.25 \
  --bucket-high 0.75 \
  --output-prefix mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct
"""

import argparse

import numpy as np
from datasets import Dataset, load_dataset

from open_instruct import logger_utils
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 25%/75% pass-rate bucket datasets.")
    parser.add_argument(
        "--input-dataset",
        default="mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct",
        help="Dataset with pass_count/num_samples or pass_rate columns.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-per-bucket", type=int, default=16)
    parser.add_argument("--bucket-low", type=float, default=0.25)
    parser.add_argument("--bucket-high", type=float, default=0.75)
    parser.add_argument(
        "--output-prefix",
        default="mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct",
        help="Outputs are <prefix>-25 and <prefix>-75",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true", help="Push to hub (default False)")
    return parser.parse_args()


def extract_pass_rate(row: dict) -> float:
    pass_count = row.get("pass_count")
    num_samples = row.get("num_samples")
    if isinstance(pass_count, int) and isinstance(num_samples, int) and num_samples > 0:
        return float(pass_count) / float(num_samples)

    pass_rate = row.get("pass_rate")
    if isinstance(pass_rate, str) and "/" in pass_rate:
        left, right = pass_rate.split("/", 1)
        numerator = int(left.strip())
        denominator = int(right.strip())
        if denominator <= 0:
            raise ValueError(f"Invalid pass_rate denominator: {pass_rate}")
        return float(numerator) / float(denominator)

    raise ValueError("Row is missing pass_count/num_samples and parseable pass_rate")


def select_closest_indices(pass_rates: np.ndarray, target: float, k: int, unavailable: set[int]) -> list[int]:
    candidates = [i for i in range(len(pass_rates)) if i not in unavailable]
    if len(candidates) < k:
        raise ValueError(f"Not enough candidates for target {target}: requested {k}, have {len(candidates)}")

    candidates_arr = np.array(candidates, dtype=int)
    distances = np.abs(pass_rates[candidates_arr] - target)
    order = np.argsort(distances)
    return candidates_arr[order[:k]].tolist()


def with_dataset_name(ds: Dataset, dataset_name: str) -> Dataset:
    if "dataset" in ds.column_names:
        ds = ds.remove_columns(["dataset"])
    return ds.add_column("dataset", [dataset_name] * len(ds))


def main() -> None:
    args = parse_args()
    logger.info("Loading %s[%s]", args.input_dataset, args.split)
    ds = load_dataset(args.input_dataset, split=args.split, num_proc=max_num_processes())

    rows = ds.to_list()
    pass_rates = np.array([extract_pass_rate(row) for row in rows], dtype=float)

    unavailable: set[int] = set()
    low_indices = select_closest_indices(pass_rates, args.bucket_low, args.num_per_bucket, unavailable)
    unavailable.update(low_indices)
    high_indices = select_closest_indices(pass_rates, args.bucket_high, args.num_per_bucket, unavailable)

    ds_low = ds.select(low_indices)
    ds_high = ds.select(high_indices)

    low_name = f"{args.output_prefix}-25"
    high_name = f"{args.output_prefix}-75"

    ds_low = with_dataset_name(ds_low, "gsm8k_pass25")
    ds_high = with_dataset_name(ds_high, "gsm8k_pass75")

    logger.info(
        "Selected low bucket: %d samples, avg pass_rate=%.4f (target %.2f)",
        len(ds_low),
        np.mean([extract_pass_rate(r) for r in ds_low.to_list()]),
        args.bucket_low,
    )
    logger.info(
        "Selected high bucket: %d samples, avg pass_rate=%.4f (target %.2f)",
        len(ds_high),
        np.mean([extract_pass_rate(r) for r in ds_high.to_list()]),
        args.bucket_high,
    )

    if args.push:
        logger.info("Pushing low bucket to %s", low_name)
        ds_low.push_to_hub(low_name, split="train", private=args.private)
        logger.info("Pushing high bucket to %s", high_name)
        ds_high.push_to_hub(high_name, split="train", private=args.private)

    logger.info("Done")


if __name__ == "__main__":
    main()
