#!/usr/bin/env python3

import sys
import argparse
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
import pandas as pd

"""
Script to remove phrases identifying the model as a certain entity in our datasets.
Motivated by: realizing the SFT mix has lots of "I am DeepSeek" snippets. 

Run with:
python scripts/data/filtering_and_updates/filter_dataset_by_keywords.py --input-dataset allenai/tulu-3-sft-mixture --column messages
"""


# Popular model providers
PROVIDERS = [
    "OpenAI", "Open AI", "Claude", "Gemini", "Qwen", "DeepSeek", "Anthropic", "Meta AI", "Meta's", "ChatGPT"
    "Cohere", "Mistral AI", "Mistral's", "xAI", "Perplexity" # "Google AI", "Google's",  "Microsoft", "HuggingFace", "Hugging Face"
]

# Regex patterns for filtering (case-insensitive for common words, case-sensitive for company names)
PATTERNS = [
    # Pattern: "I'm [model name], an AI assistant made by {provider}" 
    r"(?i)i'?m\s+(" + "|".join(PROVIDERS) + r"),?\s+an?\s+ai\s+(?:assistant|model)[^.!?]*?(?:made|developed|created|trained)\s+by\s+(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "[Model name] is an AI assistant developed by {provider}"
    r"(?i)(" + "|".join(PROVIDERS) + r")\s+is\s+an?\s+ai\s+(?:assistant|model)[^.!?]*?(?:developed|created|made|trained)\s+by\s+(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "as a [AI model/assistant/chatbot] ... {provider}" 
    r"(?i)as\s+a\s+(?:language\s+model|ai\s+model|assistant|chatbot|model)[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "as an AI developed by {provider}"
    r"(?i)as\s+an\s+ai\s+(?:developed|created|made|trained)\s+by\s+(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "I am [model type] ... {provider}"
    r"(?i)i\s+am\s+(?:a\s+)?(?:language\s+model|ai\s+model|assistant|chatbot|model)[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "I am called [provider]"
    r"(?i)i\s+am\s+called\s+\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",

    # Pattern: "I'm [provider]" or "I am [provider]"
    r"(?i)(?:i'?m|i\s+am)\s+\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",

    # Pattern: "trained by ... {provider}" within one sentence  
    r"(?i)trained\s+by\s+[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "developed by ... {provider}" within one sentence
    r"(?i)developed\s+by\s+[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "created by ... {provider}" within one sentence
    r"(?i)created\s+by\s+[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "made by ... {provider}" within one sentence
    r"(?i)made\s+by\s+[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "against {provider}'s use-case policy" or similar policy references
    r"(?i)against\s+(" + "|".join(PROVIDERS) + r")(?:'s|'s)?\s+(?:use-case\s+)?(?:policy|policies|guidelines|terms)[^.!?]*?[.!?]",
    
    # Pattern: "{provider}'s policy" or "{provider}'s guidelines"
    r"(?i)\b(" + "|".join(PROVIDERS) + r")(?:'s|'s)\s+(?:policy|policies|guidelines|terms|use-case)[^.!?]*?[.!?]",
    
    # Pattern: Any sentence containing "DeepSeek-R1" or "DeepSeek R1" (case-insensitive)
    r"(?i)[^.!?]*\bDeepSeek[\s-]?R1\b[^.!?]*?[.!?]",

    # Pattern: Anything with the word "Qwen" (case-insensitive)
    r"(?i)[^.!?]*\bQwen\b[^.!?]*?[.!?]",
    
    # Pattern: Any sentence containing "Alibaba Qwen" (case-insensitive) or Alibaba Cloud
    r"(?i)[^.!?]*\bAlibaba\s+Qwen\b[^.!?]*?[.!?]",
    r"(?i)[^.!?]*\bAlibaba\s+Cloud\b[^.!?]*?[.!?]",
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
    return should_be_filtered_by_advanced_patterns(example, column=column, verbose=verbose, filter_user_turns=filter_user_turns)

def load_dataset_from_parquet(dataset_name):
    """Load dataset directly from parquet files."""
    # List all files in the repo
    files = list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {dataset_name}")
    
    # Download and load parquet files
    dfs = []
    for file in parquet_files:
        local_file = hf_hub_download(
            repo_id=dataset_name,
            filename=file,
            repo_type="dataset"
        )
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
    parser.add_argument("--filter-user-turns", action="store_true", 
                       help="Also filter based on user messages (default: only filter assistant messages)")
    parser.add_argument("--output-entity", type=str, help="Output entity (org/user) for the filtered dataset. If not provided, uses the same entity as the input dataset.")
    parser.add_argument("--column", type=str, default="messages", 
                   help="Column name containing the messages (default: messages)")

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
        dataset = load_dataset(input_dataset, split="train")
    except:
        try:
            # Fallback to direct parquet loading
            print("Using direct parquet loading...")
            dataset = load_dataset_from_parquet(input_dataset)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            raise
    
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Keep track of filtered examples
    filtered_examples = []
    
    # Filter function
    def filter_fn(example):
        should_filter = should_be_filtered_combined(example, column=args.column, verbose=True, filter_user_turns=filter_user_turns)
        if should_filter and len(filtered_examples) < 3:
            # Find which pattern matched and extract the matching text
            for message in example[args.column]:
                # Apply same filtering logic for finding matched text
                if message["role"] == "user" and not filter_user_turns:
                    continue
                if message["role"] != "assistant" and message["role"] != "user":
                    continue
                
                content = message["content"]  # Keep original case
                
                for pattern in PATTERNS:
                    match = re.search(pattern, content)
                    if match:
                        example["_matched_text"] = match.group(0)
                        example["_matched_role"] = message["role"]
                        break
                if "_matched_text" in example:
                    break
            
            filtered_examples.append(example)
        return not should_filter
    
    print("Filtering dataset...")
    filtered_dataset = dataset.filter(filter_fn, num_proc=32)
    print(f"Filtered size: {len(filtered_dataset)}")
    print(f"Removed {len(dataset) - len(filtered_dataset)} examples")
    
    # Show a few filtered examples
    if filtered_examples:
        print("\n--- Examples that were removed ---")
        for i, example in enumerate(filtered_examples):
            print("---------------------------------")
            print(f"\nExample {i+1}:")
            if "_matched_text" in example:
                role = example.get("_matched_role", "unknown")
                print(f"  Matched text ({role}): '{example['_matched_text']}'")
            messages = example.get("args.column", [])
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    print(f"  User: {content}")
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    print(f"  Assistant: {content}")
                    break
        print("--- End of examples ---\n")
    
    # Upload
    full_name = f"{output_dataset}"
    print(f"Uploading to: {full_name}")
    filtered_dataset.push_to_hub(full_name, private=True)
    print("Done!")

if __name__ == "__main__":
    main()
