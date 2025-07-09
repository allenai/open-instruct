#!/usr/bin/env python3

import sys
import argparse
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
import pandas as pd


# Popular model providers
PROVIDERS = [
    "OpenAI", "Open AI", "Qwen", "DeepSeek", "Anthropic", "Meta AI", "Meta's", 
    "Cohere", "HuggingFace", "Hugging Face", "Mistral AI", "Mistral's", "xAI", "Perplexity" # "Google AI", "Google's",  "Microsoft",
]

# Regex patterns for filtering (case-insensitive for common words, case-sensitive for company names)
PATTERNS = [
    # Pattern: "as a [AI model/assistant/chatbot] ... {provider}" 
    r"(?i)as\s+a\s+(?:language\s+model|ai\s+model|assistant|chatbot|model)[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "I am [model type] ... {provider}"
    r"(?i)i\s+am\s+(?:a\s+)?(?:language\s+model|ai\s+model|assistant|chatbot|model)[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "trained by ... {provider}" within one sentence  
    r"(?i)trained\s+by\s+[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "developed by ... {provider}" within one sentence
    r"(?i)developed\s+by\s+[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "created by ... {provider}" within one sentence
    r"(?i)created\s+by\s+[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
    
    # Pattern: "made by ... {provider}" within one sentence
    r"(?i)made\s+by\s+[^.!?]*?\b(" + "|".join(PROVIDERS) + r")\b[^.!?]*?[.!?]",
]


def should_be_filtered_by_advanced_patterns(example, verbose=False):
    """Filter by more sophisticated patterns like 'as a ... OpenAI' or 'trained by ... Google'"""
    
    for message in example["messages"]:
        if message["role"] != "assistant":
            continue
        
        content = message["content"]  # Keep original case
        
        for pattern in PATTERNS:
            if re.search(pattern, content):
                if verbose:
                    print("--------------------------------")
                    print("Instance is filtered out by advanced pattern:")
                    print(message["content"])
                    print(f"Matched pattern: {pattern}")
                return True
    
    return False


def should_be_filtered_combined(example, verbose=False):
    """Combined filtering function"""
    return should_be_filtered_by_advanced_patterns(example, verbose)

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
    
    args = parser.parse_args()
    
    input_dataset = args.input_dataset
    # Automatically generate output name by adding -keyword-filtered
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
        should_filter = should_be_filtered_combined(example, verbose=True)
        if should_filter and len(filtered_examples) < 3:
            # Find which pattern matched and extract the matching text
            for message in example["messages"]:
                if message["role"] != "assistant":
                    continue
                
                content = message["content"]  # Keep original case
                
                for pattern in PATTERNS:
                    match = re.search(pattern, content)
                    if match:
                        example["_matched_text"] = match.group(0)
                        break
                if "_matched_text" in example:
                    break
            
            filtered_examples.append(example)
        return not should_filter
    
    print("Filtering dataset...")
    filtered_dataset = dataset.filter(filter_fn)
    print(f"Filtered size: {len(filtered_dataset)}")
    print(f"Removed {len(dataset) - len(filtered_dataset)} examples")
    
    # Show a few filtered examples
    if filtered_examples:
        print("\n--- Examples that were removed ---")
        for i, example in enumerate(filtered_examples):
            print("---------------------------------")
            print(f"\nExample {i+1}:")
            if "_matched_text" in example:
                print(f"  Matched text: '{example['_matched_text']}'")
            messages = example.get("messages", [])
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