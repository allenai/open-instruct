#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List

from datasets import DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Hugging Face Dataset from local JSONL files with specified splits and upload it as a private dataset."
    )
    parser.add_argument(
        "--local_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of local JSONL file paths. Each path corresponds to the same-index split in --splits.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        required=True,
        help="List of split names matching --local_paths (e.g., train validation test).",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        required=True,
        help="Target Hugging Face dataset repo id, e.g. your-entity/your-dataset-name",
    )
    return parser.parse_args()


def validate_inputs(local_paths: List[str], splits: List[str]) -> None:
    if len(local_paths) != len(splits):
        raise ValueError(
            f"local_paths and splits must have the same length: {len(local_paths)} != {len(splits)}"
        )
    for p in local_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Local path does not exist: {p}")
        if not p.endswith(".json") and not p.endswith(".jsonl"):
            raise ValueError(f"Expected a JSON/JSONL file, got: {p}")


def build_dataset_dict(local_paths: List[str], splits: List[str]) -> DatasetDict:
    split_to_parts: Dict[str, List] = {}
    for path, split in zip(local_paths, splits):
        ds = load_dataset("json", data_files=path, split="train")
        split_to_parts.setdefault(split, []).append(ds)

    split_to_dataset = {}
    for split, parts in split_to_parts.items():
        if len(parts) == 1:
            split_to_dataset[split] = parts[0]
        else:
            split_to_dataset[split] = concatenate_datasets(parts)

    dsd = DatasetDict(split_to_dataset)
    return dsd


def ensure_private_repo(repo_id: str) -> None:
    api = HfApi()
    # Create the dataset repo if it doesn't exist; ensure it's private.
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_inputs(args.local_paths, args.splits)

    print(f"🔧 Building DatasetDict from {len(args.local_paths)} files...")
    dsd = build_dataset_dict(args.local_paths, args.splits)

    counts = {k: len(v) for k, v in dsd.items()}
    print(f"📊 Split sizes: {counts}")

    print(f"🛡️ Ensuring private dataset repo exists: {args.hf_repo}")
    ensure_private_repo(args.hf_repo)

    print("🚀 Pushing dataset to Hugging Face Hub (private)...")
    dsd.push_to_hub(args.hf_repo)
    print(f"✅ Done. View at: https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()


