#!/usr/bin/env python3
"""
Filter allenai/Dolci-Think-SFT-32B to only include rows whose prompt
appears in allenai/gptoss120b-deduped, keeping one response per prompt.

Usage:
    python filter_deduped.py [--push-to-hub] [--save-local PATH]

By default just does a dry run. Pass --push-to-hub to upload.
"""

import argparse
import hashlib
import string
from datasets import load_dataset, Dataset
from tqdm import tqdm


TARGET_REPO = "jacobmorrison/Dolci-Think-SFT-32B-deduped-subset"
BIG_DATASET = "allenai/Dolci-Think-SFT-32B"
DEDUP_DATASET = "allenai/gptoss120b-deduped"


def extract_prompt(messages: list[dict]) -> str:
    """Extract the first user message as the prompt key (matches dedup logic)."""
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def normalize_prompt(prompt: str) -> str:
    """Soft normalization to match dedup logic: lowercase, strip whitespace and punctuation."""
    # lowercase
    s = prompt.lower()
    # remove all punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    s = " ".join(s.split())
    return s


def hash_prompt(prompt: str) -> str:
    """Hash for memory efficiency — prompts can be very long."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-to-hub", action="store_true", help="Upload to HuggingFace Hub")
    parser.add_argument("--save-local", type=str, default=None, help="Save to local path (arrow/parquet)")
    parser.add_argument("--big-dataset", type=str, default=BIG_DATASET)
    parser.add_argument("--dedup-dataset", type=str, default=DEDUP_DATASET)
    parser.add_argument("--target-repo", type=str, default=TARGET_REPO)
    args = parser.parse_args()

    # ---- Step 1: Load the deduped prompts into a set ----
    print(f"Loading deduped prompt set from {args.dedup_dataset}...")
    dedup_ds = load_dataset(args.dedup_dataset, split="train")
    print(f"  {len(dedup_ds):,} rows in dedup dataset")

    # Build a set of prompt hashes we're looking for
    wanted_prompts: set[str] = set()
    for row in tqdm(dedup_ds, desc="Hashing dedup prompts"):
        prompt = extract_prompt(row["messages"])
        wanted_prompts.add(hash_prompt(prompt)) # )normalize_prompt(prompt)))

    print(f"  {len(wanted_prompts):,} unique prompts to find")

    # ---- Step 2: Stream the big dataset, collect matches ----
    print(f"\nStreaming {args.big_dataset} to find matching rows...")
    big_ds = load_dataset(args.big_dataset, split="train", streaming=True)

    matched_rows = []
    seen = 0
    found_prompts: set[str] = set()

    for row in tqdm(big_ds, desc="Scanning", total=2_200_000):
        seen += 1
        prompt = extract_prompt(row["messages"])
        h = hash_prompt(prompt) # normalize_prompt(prompt))

        if h in wanted_prompts and h not in found_prompts:
            found_prompts.add(h)
            matched_rows.append(row)

            if len(found_prompts) % 10_000 == 0:
                print(f"  Found {len(found_prompts):,} / {len(wanted_prompts):,} prompts "
                      f"({seen:,} rows scanned)")

        # Early exit if we found everything
        if len(found_prompts) == len(wanted_prompts):
            print(f"  All prompts found after scanning {seen:,} rows!")
            break

    print(f"\nDone scanning. {seen:,} rows scanned.")
    print(f"  Matched: {len(matched_rows):,} / {len(wanted_prompts):,} prompts")

    missing = len(wanted_prompts) - len(found_prompts)
    if missing > 0:
        print(f"  ⚠️  {missing:,} prompts from dedup set were NOT found in the big dataset")

    # ---- Step 3: Build the output dataset ----
    print("\nBuilding output dataset...")
    out_ds = Dataset.from_list(matched_rows)
    print(f"  Output: {len(out_ds):,} rows, columns: {out_ds.column_names}")

    # ---- Step 4: Save / upload ----
    if args.save_local:
        print(f"\nSaving locally to {args.save_local}...")
        out_ds.save_to_disk(args.save_local)
        print("  Done.")

    if args.push_to_hub:
        print(f"\nPushing to {args.target_repo}...")
        out_ds.push_to_hub(args.target_repo, private=False)
        print(f"  Done! https://huggingface.co/datasets/{args.target_repo}")
    elif not args.save_local:
        print(f"\nDry run complete. Use --push-to-hub to upload to {args.target_repo}")
        print("  or --save-local PATH to save locally first.")


if __name__ == "__main__":
    main()
