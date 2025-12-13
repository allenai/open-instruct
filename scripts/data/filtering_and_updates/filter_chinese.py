#!/usr/bin/env python3

import argparse
import re

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

import open_instruct.utils as open_instruct_utils

"""
Script to remove examples containing Chinese characters from our datasets.
Uses Unicode character ranges for efficient detection without requiring language models.

Run with:
python scripts/data/filtering_and_updates/filter_chinese.py --input-dataset allenai/tulu-3-sft-mixture --threshold 0.05

Default threshold used is 5%
"""


def has_chinese_characters(text):
    """
    Detect if text contains Chinese characters using Unicode ranges.
    Returns True if Chinese characters are found, False otherwise.

    Uses the main CJK Unified Ideographs block (\u4e00-\u9fff) which covers
    ~20,000 of the most commonly used Chinese characters.
    """
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    return bool(chinese_pattern.search(text))


def get_chinese_character_ratio(text):
    """
    Calculate the ratio of Chinese characters in the text.
    Returns a value between 0.0 and 1.0.
    """
    if not text:
        return 0.0

    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    chinese_chars = chinese_pattern.findall(text)
    return len(chinese_chars) / len(text)


def extract_chinese_characters(text):
    """
    Extract Chinese characters from text for debugging purposes.
    """
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    matches = chinese_pattern.findall(text)
    return "".join(matches)


def should_be_filtered_by_chinese(example, verbose=False, filter_user_turns=False, threshold=None):
    """Filter examples containing Chinese characters"""

    messages = example["messages"]

    if filter_user_turns:
        # Look at last two messages (user + assistant)
        messages_to_check = messages[-2:] if len(messages) >= 2 else messages
    else:
        # Look only at the final assistant message
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        messages_to_check = [assistant_messages[-1]] if assistant_messages else []

    for message in messages_to_check:
        # Skip non-user/assistant messages
        if message["role"] != "assistant" and message["role"] != "user":
            continue

        content = message["content"]

        has_chinese = has_chinese_characters(content)

        # Check threshold if specified
        if threshold is not None:
            ratio = get_chinese_character_ratio(content)
            if ratio < threshold:
                has_chinese = False

        if has_chinese:
            if verbose:
                print("--------------------------------")
                print("Instance is filtered out due to Chinese characters:")
                print(f"Role: {message['role']}")
                print(f"Content: {content}")
                chinese_chars = extract_chinese_characters(content)
                ratio = get_chinese_character_ratio(content)
                print(f"Chinese characters found: '{chinese_chars}' (ratio: {ratio:.3f})")
            return True

    return False


def load_dataset_from_parquet(dataset_name):
    """Load dataset directly from parquet files."""
    # List all files in the repo
    files = list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]

    if not parquet_files:
        raise ValueError(f"No parquet files found in {dataset_name}")

    # Download and load parquet files
    dfs = []
    for file in parquet_files:
        local_file = hf_hub_download(repo_id=dataset_name, filename=file, repo_type="dataset")
        df = pd.read_parquet(local_file)
        dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Convert to HF Dataset
    from datasets import Dataset

    return Dataset.from_pandas(combined_df)


def main():
    parser = argparse.ArgumentParser(description="Filter a dataset by removing examples with Chinese characters")
    parser.add_argument("--input-dataset", required=True, help="Input dataset name")
    parser.add_argument(
        "--filter-user-turns",
        action="store_true",
        help="Also filter based on user messages (default: only filter assistant messages)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum ratio of Chinese characters to trigger filtering (0.0-1.0)",
    )
    parser.add_argument(
        "--output-entity",
        type=str,
        help="Output entity (org/user) for the filtered dataset. If not provided, uses the same entity as the input dataset.",
    )

    args = parser.parse_args()

    input_dataset = args.input_dataset
    filter_user_turns = args.filter_user_turns
    threshold = args.threshold

    # Generate output dataset name
    if args.output_entity:
        # Use custom output entity
        if "/" in input_dataset:
            _, dataset_name = input_dataset.split("/", 1)
        else:
            dataset_name = input_dataset
        output_dataset = f"{args.output_entity}/{dataset_name}-chinese-filtered"
    else:
        # Use same entity as input dataset
        if "/" in input_dataset:
            org, name = input_dataset.split("/", 1)
            output_dataset = f"{org}/{name}-chinese-filtered"
        else:
            output_dataset = f"{input_dataset}-chinese-filtered"

    # Check if output dataset name is too long (>94 characters)
    if len(output_dataset) > 94:
        print(f"Output dataset name too long ({len(output_dataset)} chars): {output_dataset}")

        # Try to shorten by replacing long suffix with shorter one
        if "/" in input_dataset:
            org, name = input_dataset.split("/", 1)

            # Replace the long suffix with shorter one
            shortened_name = name.replace("format-filtered-keyword-filtered-filter-datecutoff", "cn-fltrd-final")

            # If still too long, try more aggressive shortening
            if len(f"{org}/{shortened_name}-chinese-filtered") > 94:
                # Remove more parts
                shortened_name = shortened_name.replace("format-filtered-keyword-filtered", "cn-fltrd")
                shortened_name = shortened_name.replace("filter-datecutoff", "date-fltrd")
                shortened_name = shortened_name.replace("ngram-filtered", "ngram-fltrd")
                shortened_name = shortened_name.replace("final-content-filtered", "content-fltrd")
                shortened_name = shortened_name.replace("repetition-filter", "rep-fltrd")
                shortened_name = shortened_name.replace("domain-filtered", "domain-fltrd")

            output_dataset = f"{org}/{shortened_name}-chinese-filtered"
        else:
            # For datasets without org prefix, just truncate
            max_name_length = 94 - len("-chinese-filtered")
            output_dataset = f"{input_dataset[:max_name_length]}-chinese-filtered"

        print(f"Shortened to ({len(output_dataset)} chars): {output_dataset}")

    print(f"Loading dataset: {input_dataset}")

    try:
        # Try standard loading first
        dataset = load_dataset(input_dataset, split="train", num_proc=open_instruct_utils.max_num_processes())
    except:
        try:
            # Fallback to direct parquet loading
            print("Using direct parquet loading...")
            dataset = load_dataset_from_parquet(input_dataset)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            raise

    print(f"Dataset loaded with {len(dataset)} examples")
    print("Using standard Chinese character detection (CJK Unified Ideographs)")
    if threshold is not None:
        print(f"Using threshold: {threshold:.3f} (minimum ratio of Chinese characters)")

    # Statistics tracking
    total_examples = len(dataset)
    filtered_count = 0
    user_filtered = 0
    assistant_filtered = 0
    chinese_ratios = []

    # Keep track of filtered examples
    filtered_examples = []

    # Filter function
    def filter_fn(example):
        nonlocal filtered_count, user_filtered, assistant_filtered

        should_filter = should_be_filtered_by_chinese(
            example, verbose=False, filter_user_turns=filter_user_turns, threshold=threshold
        )

        if should_filter:
            filtered_count += 1

            # Find which message contained Chinese characters and collect stats
            messages = example["messages"]

            if filter_user_turns:
                # Look at last two messages (user + assistant)
                messages_to_check = messages[-2:] if len(messages) >= 2 else messages
            else:
                # Look only at the final assistant message
                assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
                messages_to_check = [assistant_messages[-1]] if assistant_messages else []

            for message in messages_to_check:
                # Skip non-user/assistant messages
                if message["role"] != "assistant" and message["role"] != "user":
                    continue

                content = message["content"]
                has_chinese = has_chinese_characters(content)

                # Check threshold if specified
                if threshold is not None:
                    ratio = get_chinese_character_ratio(content)
                    chinese_ratios.append(ratio)
                    if ratio < threshold:
                        has_chinese = False

                if has_chinese:
                    example["_chinese_chars"] = extract_chinese_characters(content)
                    example["_chinese_role"] = message["role"]
                    example["_chinese_ratio"] = get_chinese_character_ratio(content)

                    # Track statistics
                    if message["role"] == "user":
                        user_filtered += 1
                    elif message["role"] == "assistant":
                        assistant_filtered += 1
                    break

            if len(filtered_examples) < 5:  # Show more examples for Chinese detection
                filtered_examples.append(example)

        return not should_filter

    print("Filtering dataset...")
    filtered_dataset = dataset.filter(filter_fn)
    print(f"Filtered size: {len(filtered_dataset)}")
    print(f"Removed {len(dataset) - len(filtered_dataset)} examples")

    # Show filtered examples
    if filtered_examples:
        print("\n--- Examples that were removed ---")
        for i, example in enumerate(filtered_examples):
            print("---------------------------------")
            print(f"\nExample {i+1}:")
            if "_chinese_chars" in example:
                role = example.get("_chinese_role", "unknown")
                ratio = example.get("_chinese_ratio", 0.0)
                print(f"  Chinese characters found ({role}): '{example['_chinese_chars']}' (ratio: {ratio:.3f})")

            # Print all messages in the conversation
            messages = example.get("messages", [])
            for j, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                print(f"  {role.capitalize()}: {content}")
                # Only show first few messages to avoid overwhelming output
                if j >= 3:
                    print(f"  ... ({len(messages) - 4} more messages)")
                    break
        print("--- End of examples ---\n")

    # Print statistics
    print("\n--- Filtering Statistics ---")
    print(f"Total examples: {total_examples}")
    print(f"Examples removed: {filtered_count}")
    print(f"Removal rate: {filtered_count/total_examples*100:.2f}%")
    if filter_user_turns:
        print(f"  - User messages filtered: {user_filtered}")
        print(f"  - Assistant messages filtered: {assistant_filtered}")
    if chinese_ratios:
        print("Chinese character ratios in filtered examples:")
        print(f"  - Min: {min(chinese_ratios):.3f}")
        print(f"  - Max: {max(chinese_ratios):.3f}")
        print(f"  - Mean: {sum(chinese_ratios)/len(chinese_ratios):.3f}")
        print(f"  - Median: {sorted(chinese_ratios)[len(chinese_ratios)//2]:.3f}")

    # Upload
    full_name = f"{output_dataset}"
    print(f"Uploading to: {full_name}")
    filtered_dataset.push_to_hub(full_name, private=True)
    print("Done!")


if __name__ == "__main__":
    main()
