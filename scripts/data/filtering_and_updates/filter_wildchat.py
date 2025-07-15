#!/usr/bin/env python
"""Quick utility to filter and re‑upload Wildchat datasets.

Steps:
1. Load the source dataset from command line argument.
2. Filter out toxic examples (toxic=True).
3. Filter out redacted examples (redacted=True).
4. Keep only English examples (case‑insensitive on the `language` field).
5. Push this subset to the output dataset specified in command line argument.

Requires: `datasets` (🤗 Datasets) and an authenticated Hugging Face environment.
Run `huggingface-cli login` or set the `HF_TOKEN` env var before executing.

Usage:
python filter_wildchat.py --input-dataset <input_dataset> --output-dataset <output_dataset>
"""
import argparse
import logging
import os
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def filter_toxic(ds):
    """Filter out rows where `toxic` is True."""
    logging.info("Filtering out toxic examples…")
    initial_count = ds.num_rows
    ds_filtered = ds.filter(lambda ex: not ex.get("toxic", False))
    filtered_out = initial_count - ds_filtered.num_rows
    logging.info("Filtered out %d toxic examples (%d remaining, %.2f%%).", 
                 filtered_out, ds_filtered.num_rows, 100 * ds_filtered.num_rows / initial_count)
    return ds_filtered

def filter_redacted(ds):
    """Filter out rows where `redacted` is True."""
    logging.info("Filtering out redacted examples…")
    initial_count = ds.num_rows
    ds_filtered = ds.filter(lambda ex: not ex.get("redacted", False))
    filtered_out = initial_count - ds_filtered.num_rows
    logging.info("Filtered out %d redacted examples (%d remaining, %.2f%%).", 
                 filtered_out, ds_filtered.num_rows, 100 * ds_filtered.num_rows / initial_count)
    return ds_filtered

def filter_language(ds, lang="english"):
    """Return only rows whose `language` matches `lang` (case‑insensitive)."""
    logging.info("Filtering for language='%s' (case‑insensitive)…", lang)
    initial_count = ds.num_rows
    ds_filtered = ds.filter(lambda ex: ex.get("language", "").lower() == lang.lower())
    filtered_out = initial_count - ds_filtered.num_rows
    logging.info("Filtered out %d non-English examples (%d remaining, %.2f%%).", 
                 filtered_out, ds_filtered.num_rows, 100 * ds_filtered.num_rows / initial_count)
    return ds_filtered

def main():
    parser = argparse.ArgumentParser(description="Filter Wildchat dataset")
    parser.add_argument("--input-dataset", help="Input dataset name/path")
    parser.add_argument("--output-dataset", help="Output dataset name/path")
    args = parser.parse_args()

    logging.info("Loading dataset %s…", args.input_dataset)
    ds = load_dataset(args.input_dataset, split="train")
    logging.info("Loaded dataset with %d examples.", ds.num_rows)

    # 1) Filter out toxic examples
    ds = filter_toxic(ds)

    # 2) Filter out redacted examples
    ds = filter_redacted(ds)

    # 3) Filter by language
    ds = filter_language(ds, "english")

    # 4) Push filtered dataset
    logging.info("Pushing filtered dataset to %s…", args.output_dataset)
    ds.push_to_hub(args.output_dataset, private=False)

    logging.info("All done!")

if __name__ == "__main__":
    if os.environ.get("HF_TOKEN") is None:
        logging.warning("HF_TOKEN environment variable not set. "
                        "Run 'huggingface-cli login' or set HF_TOKEN to authenticate.")
    main()