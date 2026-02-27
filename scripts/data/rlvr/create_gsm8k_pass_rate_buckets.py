"""Create one GSM8K dataset repo with bucketed pass@k selections.

Example:
HF_TOKEN=... uv run python scripts/data/rlvr/create_gsm8k_pass_rate_buckets.py \
  --input-dataset mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct \
  --split test \
  --num-per-bucket 16 \
  --k 32 \
  --buckets 0% 5% 10% 25% \
  --output-dataset mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-buckets \
  --push-layout all
"""

import argparse
import math

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from open_instruct import grpo_utils
from open_instruct import logger_utils
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pass-rate bucket subsets and push as one dataset repo.")
    parser.add_argument(
        "--input-dataset",
        default="mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct",
        help="Dataset with pass_count/num_samples or pass_rate columns.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-per-bucket", type=int, default=16)
    parser.add_argument("--k", type=int, default=32, help="k to use for pass@k estimation.")
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["25%", "75%"],
        help="Target pass@k buckets as percentages (e.g., 0%% 5%% 10%% 25%% or 0,5,10,25).",
    )
    parser.add_argument("--output-dataset", required=True, help="Target HF dataset repo to push to.")
    parser.add_argument(
        "--push-layout",
        choices=["all", "bucket_splits", "test_concat"],
        default="all",
        help=(
            "Push layout for the single output dataset: "
            "`all` -> one split per bucket plus concatenated split `test`, "
            "`bucket_splits` -> one split per bucket (e.g., bucket_5), "
            "`test_concat` -> all selected rows concatenated into split `test`."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true", help="Push to hub (default False)")
    return parser.parse_args()


def parse_bucket_percent(value: str) -> float:
    text = value.strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    pct = float(text)
    if pct < 0 or pct > 100:
        raise ValueError(f"Bucket percentage must be in [0, 100], got: {value}")
    return pct / 100.0


def parse_buckets(values: list[str]) -> list[float]:
    parsed: list[float] = []
    for raw in values:
        for token in raw.split(","):
            stripped = token.strip()
            if stripped:
                parsed.append(parse_bucket_percent(stripped))
    if not parsed:
        raise ValueError("At least one bucket must be provided")
    return parsed


def extract_num_correct_and_samples(row: dict) -> tuple[int, int]:
    pass_count = row.get("pass_count")
    num_samples = row.get("num_samples")
    if isinstance(pass_count, int) and isinstance(num_samples, int) and num_samples > 0:
        return int(pass_count), int(num_samples)

    pass_rate = row.get("pass_rate")
    if isinstance(pass_rate, str) and "/" in pass_rate:
        left, right = pass_rate.split("/", 1)
        numerator = int(left.strip())
        denominator = int(right.strip())
        if denominator <= 0:
            raise ValueError(f"Invalid pass_rate denominator: {pass_rate}")
        return numerator, denominator

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


def bucket_suffix(bucket: float) -> str:
    pct = bucket * 100.0
    rounded = round(pct)
    if math.isclose(pct, rounded, rel_tol=0.0, abs_tol=1e-9):
        return str(int(rounded))
    return f"{pct:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    buckets = parse_buckets(args.buckets)
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got: {args.k}")

    logger.info("Loading %s[%s]", args.input_dataset, args.split)
    ds = load_dataset(args.input_dataset, split=args.split, num_proc=max_num_processes())

    rows = ds.to_list()
    pass_counts_and_samples = [extract_num_correct_and_samples(row) for row in rows]
    pass_rates = np.array(
        [grpo_utils.estimate_pass_at_k(num_samples=n, num_correct=c, k=args.k) for c, n in pass_counts_and_samples],
        dtype=float,
    )

    unavailable: set[int] = set()
    bucket_outputs: list[tuple[str, Dataset]] = []
    for bucket in buckets:
        indices = select_closest_indices(pass_rates, bucket, args.num_per_bucket, unavailable)
        unavailable.update(indices)

        suffix = bucket_suffix(bucket)
        bucket_ds = with_dataset_name(ds.select(indices), f"gsm8k_pass{suffix}")
        split_name = f"bucket_{suffix}"
        bucket_outputs.append((split_name, bucket_ds))

        logger.info(
            "Selected bucket %s%%: %d samples, avg pass@%d=%.4f (target %.2f%%)",
            suffix,
            len(bucket_ds),
            args.k,
            float(np.mean(pass_rates[np.array(indices, dtype=int)])),
            bucket * 100.0,
        )

    if args.push:
        if args.push_layout in {"all", "bucket_splits"}:
            for split_name, bucket_ds in bucket_outputs:
                logger.info("Pushing split %s to %s", split_name, args.output_dataset)
                bucket_ds.push_to_hub(args.output_dataset, split=split_name, private=args.private)
        if args.push_layout in {"all", "test_concat"}:
            concatenated = concatenate_datasets([bucket_ds for _, bucket_ds in bucket_outputs])
            logger.info("Pushing concatenated split test (%d rows) to %s", len(concatenated), args.output_dataset)
            concatenated.push_to_hub(args.output_dataset, split="test", private=args.private)

    logger.info("Done")


if __name__ == "__main__":
    main()
