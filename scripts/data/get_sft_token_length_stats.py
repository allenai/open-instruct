#!/usr/bin/env python3
"""Measure chat-templated token lengths for an SFT dataset without truncation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from pathlib import Path

import numpy as np

from open_instruct import dataset_transformation, utils

PERCENTILES = (50, 75, 90, 95, 99, 99.5, 99.9, 100)
THRESHOLDS = (32_768, 65_536, 131_072, 262_144)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--chat-template", default="olmo_thinker")
    parser.add_argument("--num-proc", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = utils.get_datasets(
        {args.dataset: 1.0}, splits=[args.split], columns_to_keep=["messages"], shuffle=False
    )["train"]
    tokenizer = dataset_transformation.TokenizerConfig(
        tokenizer_name_or_path=args.tokenizer, chat_template_name=args.chat_template
    ).tokenizer

    def get_length(row: dict) -> dict[str, int]:
        token_ids = tokenizer.apply_chat_template(
            conversation=row["messages"], tokenize=True, add_generation_prompt=False, return_dict=False
        )
        return {"token_length": len(token_ids), "message_count": len(row["messages"])}

    num_proc = args.num_proc or int(float(multiprocessing.cpu_count()))
    measured = dataset.map(
        get_length,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"Measuring token lengths for {args.dataset}",
    )
    lengths = np.asarray(measured["token_length"], dtype=np.int64)
    message_counts = np.asarray(measured["message_count"], dtype=np.int64)

    role_counts: Counter[str] = Counter()
    for row in dataset:
        role_counts.update(message["role"] for message in row["messages"])

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "chat_template": args.chat_template,
        "instances": int(len(lengths)),
        "total_tokens": int(lengths.sum()),
        "mean_tokens": float(lengths.mean()),
        "min_tokens": int(lengths.min()),
        "max_tokens": int(lengths.max()),
        "token_percentiles": {str(p): float(np.percentile(lengths, p)) for p in PERCENTILES},
        "instances_over_threshold": {str(t): int((lengths > t).sum()) for t in THRESHOLDS},
        "max_messages": int(message_counts.max()),
        "message_percentiles": {str(p): float(np.percentile(message_counts, p)) for p in PERCENTILES},
        "role_counts": dict(sorted(role_counts.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
