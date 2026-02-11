#!/usr/bin/env python3
"""
Tokenize a HuggingFace dataset's "messages" column using a chat template,
and cache the resulting token lengths to a JSON file for later analysis.

Usage:
    python tokenize_dataset.py <dataset_name> <output_path> [--tokenizer <tokenizer>] [--split <split>] [--subset <subset>]

Examples:
    python tokenize_dataset.py HuggingFaceH4/ultrachat_200k lengths/ultrachat.json
    python tokenize_dataset.py my-org/my-dataset lengths/mine.json --tokenizer meta-llama/Llama-3.1-8B-Instruct --split train
    python tokenize_dataset.py my-org/my-dataset lengths/mine.json --subset default --split train
"""

import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize dataset messages and cache lengths.")
    parser.add_argument("dataset", type=str, help="HuggingFace dataset name or path")
    parser.add_argument("output", type=str, help="Output path for the JSON file")
    parser.add_argument("--tokenizer", type=str, default="allenai/OLMo-2-0325-32B-Instruct",
                        help="Tokenizer to use (default: allenai/OLMo-2-0325-32B-Instruct)")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to use. If not specified, uses the first available split.")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset/config name (e.g. 'default')")
    parser.add_argument("--messages-column", type=str, default="messages",
                        help="Name of the messages column (default: 'messages')")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of samples to process (default: all)")
    parser.add_argument("--num-proc", type=int, default=None,
                        help="Number of processes for dataset loading")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if tokenizer.chat_template is None:
        raise ValueError(f"Tokenizer {args.tokenizer} does not have a chat_template set.")

    print(f"Loading dataset: {args.dataset}" + (f" (subset={args.subset})" if args.subset else ""))
    load_kwargs = {}
    if args.subset:
        load_kwargs["name"] = args.subset
    if args.split:
        load_kwargs["split"] = args.split
    if args.num_proc:
        load_kwargs["num_proc"] = args.num_proc

    ds = load_dataset(args.dataset, **load_kwargs)

    # If no split was specified, load_dataset returns a DatasetDict — grab the first split
    if hasattr(ds, "keys"):
        available_splits = list(ds.keys())
        print(f"Available splits: {available_splits}")
        split_name = args.split if args.split else available_splits[0]
        print(f"Using split: {split_name}")
        ds = ds[split_name]

    if args.messages_column not in ds.column_names:
        raise ValueError(
            f"Column '{args.messages_column}' not found. "
            f"Available columns: {ds.column_names}"
        )

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Processing {len(ds)} samples...")

    lengths = []
    errors = 0
    start = time.time()

    for i, row in enumerate(tqdm(ds, desc="Tokenizing")):
        messages = row[args.messages_column]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(token_ids))
        except Exception as e:
            if errors < 5:
                print(f"  Warning: skipping row {i}: {e}")
            elif errors == 5:
                print("  (suppressing further warnings)")
            errors += 1

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s. {len(lengths)} succeeded, {errors} errors.")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "num_samples": len(lengths),
        "num_errors": errors,
        "lengths": lengths,
    }

    with open(output_path, "w") as f:
        json.dump(result, f)

    print(f"Saved {len(lengths)} lengths to {output_path}")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f} tokens")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")


if __name__ == "__main__":
    main()
