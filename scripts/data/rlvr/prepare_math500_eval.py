#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

DEFAULT_DATASET = "HuggingFaceH4/MATH-500"


def format_math500_row(
    row: dict[str, Any], dataset_label: str, source_dataset: str = DEFAULT_DATASET
) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": row["problem"]}],
        "ground_truth": str(row["answer"]),
        "dataset": dataset_label,
        "source_dataset": source_dataset,
        "source_id": row["unique_id"],
        "subject": row["subject"],
        "level": row["level"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MATH-500 into OpenInstruct's RLVR evaluation format.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-label", default="math_500")
    parser.add_argument("--expected-size", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    if len(dataset) != args.expected_size:
        raise ValueError(f"Expected {args.expected_size} MATH-500 rows, found {len(dataset)}")

    formatted = dataset.map(
        lambda row: format_math500_row(row, dataset_label=args.dataset_label, source_dataset=args.dataset),
        remove_columns=dataset.column_names,
        desc="Formatting MATH-500 for OpenInstruct evaluation",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    eval_path = args.output_dir / "eval.jsonl"
    formatted.to_json(eval_path)

    metadata = {
        "source_dataset": args.dataset,
        "source_split": args.split,
        "eval_size": len(formatted),
        "eval_dataset_label": args.dataset_label,
        "eval_file": eval_path.name,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Created MATH-500 evaluation dataset: %s", metadata)


if __name__ == "__main__":
    main()
