#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic held-out split of a DAPO math dataset.")
    parser.add_argument("--dataset", default="hamishivi/DAPO-Math-17k-Processed_filtered")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    if args.eval_size <= 0 or args.eval_size >= len(dataset):
        raise ValueError(f"eval-size must be between 1 and {len(dataset) - 1}, got {args.eval_size}")

    splits = dataset.train_test_split(test_size=args.eval_size, seed=args.seed, shuffle=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    splits["train"].to_json(train_path)
    splits["test"].to_json(eval_path)

    metadata = {
        "source_dataset": args.dataset,
        "source_split": args.split,
        "seed": args.seed,
        "train_size": len(splits["train"]),
        "eval_size": len(splits["test"]),
        "train_file": train_path.name,
        "eval_file": eval_path.name,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Created deterministic DAPO split: %s", metadata)


if __name__ == "__main__":
    main()
