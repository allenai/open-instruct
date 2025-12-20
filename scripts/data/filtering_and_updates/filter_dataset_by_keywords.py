#!/usr/bin/env python3

import argparse
import re

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

import open_instruct.utils as open_instruct_utils

"""
Script to remove phrases identifying the model as a certain entity in our datasets.
Motivated by: realizing the SFT mix has lots of "I am DeepSeek" snippets.

Run with:
python scripts/data/filtering_and_updates/filter_dataset_by_keywords.py --input-dataset allenai/tulu-3-sft-mixture --column messages
"""

import os

os.environ["HF_DATASETS_DISABLE_CACHING"] = "1"

from datasets import disable_caching

disable_caching()


# Popular model providers
PROVIDERS = [
    "OpenAI",
    "Open AI",
    "Claude",
    "Gemini",
    "Qwen",
    "DeepSeek",
    "Anthropic",
    "Meta AI",
    "Meta's",
    "ChatGPT",
    "Cohere",
    "Mistral AI",
    "Mistral's",
    "xAI",
    "Perplexity",  # "Google AI", "Google's",  "Microsoft", "HuggingFace", "Hugging Face"
]

# Regex patterns for filtering (case-insensitive for common words, case-sensitive for company names)
# Regex patterns for filtering (case-insensitive for common words, case-sensitive for company names)
PATTERNS = [
    # Pattern: "I'm [model name], an AI assistant made by {provider}"
    # Kept full range, removed optional grouping that was too restrictive
    r"(?i)\bI'?m\s+("
    + "|".join(PROVIDERS)
    + r"),?\s+an?\s+AI\s+(?:assistant|model)[^.!?]{0,100}(?:made|developed|created|trained)\s+by\s+("
    + "|".join(PROVIDERS)
    + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "[Model name] is an AI assistant developed by {provider}"
    # Restored full pattern
    r"(?i)\b("
    + "|".join(PROVIDERS)
    + r")\s+is\s+an?\s+AI\s+(?:assistant|model)[^.!?]{0,100}(?:developed|created|made|trained)\s+by\s+("
    + "|".join(PROVIDERS)
    + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "as a [AI model/assistant/chatbot] ... {provider}"
    # Kept greedy to match more
    r"(?i)\bas\s+an?\s+(?:language\s+model|AI\s+model|assistant|chatbot|model)[^.!?]{0,100}\b("
    + "|".join(PROVIDERS)
    + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "as an AI developed by {provider}"
    # Kept full range
    r"(?i)\bas\s+an\s+AI\s+(?:developed|created|made|trained)\s+by\s+("
    + "|".join(PROVIDERS)
    + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "I am [model type] ... {provider}"
    # Kept greedy for full matches
    r"(?i)\bI\s+am\s+(?:a\s+)?(?:language\s+model|AI\s+model|assistant|chatbot|model)[^.!?]{0,100}\b("
    + "|".join(PROVIDERS)
    + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "I am called [provider]"
    r"(?i)\bI\s+am\s+called\s+\b(" + "|".join(PROVIDERS) + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "I'm [provider]" or "I am [provider]"
    r"(?i)\b(?:I'?m|I\s+am)\s+\b(" + "|".join(PROVIDERS) + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "trained by ... {provider}" within one sentence
    # Made middle section non-greedy but kept full ranges
    r"(?i)\btrained\s+by\s+[^.!?]{0,100}?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "developed by ... {provider}" within one sentence
    r"(?i)\bdeveloped\s+by\s+[^.!?]{0,100}?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "created by ... {provider}" within one sentence
    r"(?i)\bcreated\s+by\s+[^.!?]{0,100}?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "made by ... {provider}" within one sentence
    r"(?i)\bmade\s+by\s+[^.!?]{0,100}?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]{0,100}[.!?]",
    # Pattern: "against {provider}'s use-case policy" or similar policy references
    r"(?i)\bagainst\s+("
    + "|".join(PROVIDERS)
    + r")(?:'s|'s)?\s+(?:use-case\s+)?(?:policy|policies|guidelines|terms)[^.!?]{0,100}[.!?]",
    # Pattern: "{provider}'s policy" or "{provider}'s guidelines"
    r"(?i)\b(" + "|".join(PROVIDERS) + r")(?:'s|'s)\s+(?:policy|policies|guidelines|terms|use-case)[^.!?]{0,100}[.!?]",
    # Pattern: Any sentence containing "DeepSeek-R1" or "DeepSeek R1" (case-insensitive)
    # Less restrictive: bounded but allows more at the start
    # r"(?i)[^.!?]{0,250}?\bDeepSeek[\s-]?R1\b[^.!?]{0,100}[.!?]",
    r"(?i)[^.!?]{0,250}?\bDeepSeek\b[^.!?]{0,100}[.!?]",
    # Pattern: Anything with the word "Qwen" (case-insensitive)
    # Less restrictive: bounded but allows more at the start
    r"(?i)[^.!?]{0,250}?\bQwen\b[^.!?]{0,100}[.!?]",
    # Pattern: Any sentence containing "Alibaba Qwen" (case-insensitive) or Alibaba Cloud
    # Less restrictive: bounded but allows more at the start
    r"(?i)[^.!?]{0,250}?\bAlibaba\s+Qwen\b[^.!?]{0,100}[.!?]",
    r"(?i)[^.!?]{0,250}?\bAlibaba\s+Cloud\b[^.!?]{0,100}[.!?]",
]


def should_be_filtered_by_advanced_patterns(example, column="messages", verbose=False, filter_user_turns=False):
    """Filter by more sophisticated patterns like 'as a ... OpenAI' or 'trained by ... Google'"""

    for message in example[column]:
        # Skip user messages unless explicitly enabled
        if message["role"] == "user" and not filter_user_turns:
            continue
        if message["role"] != "assistant" and message["role"] != "user":
            continue

        content = message["content"]  # Keep original case
        # empty content check
        if content is None:
            return True
        for pattern in PATTERNS:
            if re.search(pattern, content):
                if verbose:
                    print("--------------------------------")
                    print("Instance is filtered out by advanced pattern:")
                    print(message["content"])
                    print(f"Matched pattern: {pattern}")
                return True

    return False


def should_be_filtered_combined(example, column="messages", verbose=False, filter_user_turns=False):
    """Combined filtering function"""
    return should_be_filtered_by_advanced_patterns(
        example, column=column, verbose=verbose, filter_user_turns=filter_user_turns
    )


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
    parser = argparse.ArgumentParser(description="Filter a dataset by keywords and upload to Hugging Face")
    parser.add_argument("--input-dataset", required=True, help="Input dataset name")
    parser.add_argument(
        "--filter-user-turns",
        action="store_true",
        help="Also filter based on user messages (default: only filter assistant messages)",
    )
    parser.add_argument(
        "--output-entity",
        type=str,
        help="Output entity (org/user) for the filtered dataset. If not provided, uses the same entity as the input dataset.",
    )
    parser.add_argument(
        "--column", type=str, default="messages", help="Column name containing the messages (default: messages)"
    )

    args = parser.parse_args()

    input_dataset = args.input_dataset
    filter_user_turns = args.filter_user_turns

    # Generate output dataset name
    if args.output_entity:
        # Use custom output entity
        if "/" in input_dataset:
            _, dataset_name = input_dataset.split("/", 1)
        else:
            dataset_name = input_dataset
        output_dataset = f"{args.output_entity}/{dataset_name}-keyword-filtered"
    else:
        # Use same entity as input dataset
        if "/" in input_dataset:
            org, name = input_dataset.split("/", 1)
            output_dataset = f"{org}/{name}-keyword-filtered"
        else:
            output_dataset = f"{input_dataset}-keyword-filtered"

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

    print("Filtering dataset...")
    # First filter without debugging
    filtered_dataset = dataset.filter(
        lambda ex: not should_be_filtered_combined(
            ex, column=args.column, verbose=False, filter_user_turns=filter_user_turns
        ),
        num_proc=open_instruct_utils.max_num_processes(),
    )
    print(f"Filtered size: {len(filtered_dataset)}")
    print(f"Removed {len(dataset) - len(filtered_dataset)} examples")

    # Then collect a few filtered examples in serial for inspection
    if len(dataset) > len(filtered_dataset):
        print("\nCollecting example filtered instances...")
        examples_found = 0
        print_within = min(1000, len(dataset))
        for example in dataset.select(range(print_within)):
            if should_be_filtered_combined(
                example, column=args.column, verbose=True, filter_user_turns=filter_user_turns
            ):
                # Show the example
                examples_found += 1
                if examples_found >= 10:
                    break

    # Upload
    full_name = f"{output_dataset}"
    print(f"Uploading to: {full_name}")
    filtered_dataset.push_to_hub(full_name, private=True, num_proc=open_instruct_utils.max_num_processes())
    print("Done!")


if __name__ == "__main__":
    main()
