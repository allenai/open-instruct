#!/usr/bin/env python3
"""
Script to:
1) Load a base dataset from the Hugging Face Hub.
2) Remove rows whose `source` field matches any unwanted sources (if any).
3) Load one or more additional datasets from the Hub (if any) and concatenate them all.
4) Remove any unwanted columns.
5) Optionally push the combined dataset back to the Hub.

Usage example:

python scripts/data/filtering_and_updates/update_subsets.py \
    --base_ds allenai/tulu-3-sft-olmo-2-mixture-filter-datecutoff \
    --remove_sources ai2-adapt-dev/personahub_math_v5_regen_149960 allenai/tulu-3-sft-personas-math-grade \
    --add_ds allenai/tulu-3-sft-personas-math-filtered allenai/tulu-3-sft-personas-math-grade-filtered \
    --remove_keys prompt dataset \
    --push_to_hub \
    --repo_id allenai/tulu-3-sft-olmo-2-mixture-0225
"""

import argparse

from datasets import Dataset, concatenate_datasets, load_dataset

import open_instruct.utils as open_instruct_utils


def remove_specified_columns(dataset: Dataset, columns_to_remove: list) -> Dataset:
    """Removes specified columns from the dataset."""
    existing_columns = dataset.column_names
    columns_to_remove = [col for col in columns_to_remove if col in existing_columns]

    if columns_to_remove:
        print(f"Removing specified columns: {columns_to_remove}")
        dataset = dataset.remove_columns(columns_to_remove)
    else:
        print("No matching columns found to remove.")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Filter and merge Hugging Face datasets.")

    # Base dataset
    parser.add_argument(
        "--base_ds",
        default="allenai/tulu-3-sft-mixture-filter-datecutoff",
        help="Name or path of the base dataset to load (from HF Hub or local).",
    )

    # Remove sources
    parser.add_argument(
        "--remove_sources",
        nargs="*",
        default=[],
        help="List of sources to remove from the base dataset. If empty, no removal is done.",
    )

    # Add datasets (list)
    parser.add_argument(
        "--add_ds",
        nargs="*",
        default=[],
        help="Name(s) or path(s) of one or more datasets to load from HF Hub and append. If empty, no datasets are added.",
    )

    # Push options
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether to push the combined dataset to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HF Hub repo ID to push the final dataset. Required if --push_to_hub is used.",
    )

    # Remove column with keys
    parser.add_argument(
        "--remove_keys", nargs="*", default=[], help="List of columns to remove from the final dataset."
    )

    args = parser.parse_args()

    # 1. Load the base dataset
    print(f"Loading base dataset: {args.base_ds}")
    base_ds = load_dataset(args.base_ds, split="train", num_proc=open_instruct_utils.max_num_processes())

    # 2. Filter out unwanted sources, if any
    if len(args.remove_sources) > 0:
        print(f"Removing rows with source in: {args.remove_sources}")

        def keep_example(example):
            return example["source"] not in args.remove_sources

        filtered_ds = base_ds.filter(keep_example)
    else:
        print("No sources to remove; skipping filter step.")
        filtered_ds = base_ds

    # 3. Load the 'add' datasets (could be zero, one, or multiple)
    if len(args.add_ds) > 0:
        print(f"Loading and concatenating additional datasets: {args.add_ds}")
        # Start with the filtered base dataset
        combined_ds = filtered_ds

        for ds_name in args.add_ds:
            add_ds = load_dataset(ds_name, split="train", num_proc=open_instruct_utils.max_num_processes())
            # add column to add_ds where 'source' is the name of the column (it shouldnt be in it for subset)
            source = [ds_name] * len(add_ds)
            add_ds = add_ds.add_column("source", source)

            combined_ds = concatenate_datasets([combined_ds, add_ds])

        print(f"Resulting dataset size after adding: {len(combined_ds)}")
    else:
        print("No additional datasets to add.")
        combined_ds = filtered_ds

    # 4. Remove empty columns
    combined_ds = remove_specified_columns(combined_ds, args.remove_keys)

    # 5. Optionally push to the Hugging Face Hub
    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("Must provide --repo_id when --push_to_hub is used.")
        print(f"Pushing combined dataset to: {args.repo_id}")
        combined_ds.push_to_hub(args.repo_id, private=True)
        print("Push completed.")

    print("Done!")


if __name__ == "__main__":
    main()
