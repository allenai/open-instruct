#!/usr/bin/env python3
"""Download a HF hub dataset split and save it as a local .jsonl file.

Small helper for scripts that need to feed a hub dataset into a script expecting
local .jsonl input (e.g. filter_existing_dataset_correctness.py).
"""

import argparse

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HF hub dataset repo id")
    parser.add_argument("--split", default="train", help="Dataset split to download")
    parser.add_argument("--output_file", required=True, help="Path to write the .jsonl file")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    ds.to_json(args.output_file)


if __name__ == "__main__":
    main()
