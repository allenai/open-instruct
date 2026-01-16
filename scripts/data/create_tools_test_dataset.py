#!/usr/bin/env python3
"""
Create a test dataset based on hamishivi/tulu_3_rewritten_100k with an additional 'tools' column.

The tools column contains a list of active tool names (strings) that can be:
- "search"
- "code" (only if messages mention code/python/program)
- "browse"

This is used to test per-sample tool activation functionality.
"""

import random
import re
from datasets import load_dataset

# Tool names
SEARCH_TOOL = "search"
CODE_TOOL = "code"
BROWSE_TOOL = "browse"

# Patterns to detect code-related content
CODE_PATTERNS = [
    r'\bcode\b',
    r'\bpython\b',
    r'\bprogram\b',
    r'\bprogramming\b',
    r'\bfunction\b',
    r'\bdef\s+\w+',
    r'\bclass\s+\w+',
    r'\bimport\s+\w+',
    r'```',
    r'\bscript\b',
    r'\balgorithm\b',
]

CODE_REGEX = re.compile('|'.join(CODE_PATTERNS), re.IGNORECASE)


def messages_mention_code(messages: list[dict]) -> bool:
    """Check if any message in the conversation mentions code/python/program."""
    for msg in messages:
        content = msg.get("content", "")
        if CODE_REGEX.search(content):
            return True
    return False


def generate_tools_for_sample(messages: list[dict], rng: random.Random) -> list[str]:
    """
    Generate a random list of active tools for a sample.
    
    Rules:
    - search and browse can always be included
    - code can only be included if messages mention code/python/program
    - Each tool has a 50% chance of being included (if eligible)
    - There's a 10% chance of returning an empty list (no tools active)
    - There's a 5% chance of returning None (all tools active - represented as missing)
    """
    # 5% chance of None (all tools active)
    if rng.random() < 0.05:
        return None
    
    # 10% chance of empty list (no tools active)
    if rng.random() < 0.10:
        return []
    
    tools = []
    is_code_related = messages_mention_code(messages)
    
    # search: 50% chance
    if rng.random() < 0.5:
        tools.append(SEARCH_TOOL)
    
    # code: 50% chance, but only if code-related
    if is_code_related and rng.random() < 0.5:
        tools.append(CODE_TOOL)
    
    # browse: 50% chance
    if rng.random() < 0.5:
        tools.append(BROWSE_TOOL)
    
    return tools


def add_tools_column(example: dict, idx: int, rng: random.Random) -> dict:
    """Add tools column to a single example."""
    messages = example.get("messages", [])
    tools = generate_tools_for_sample(messages, rng)
    example["tools"] = tools
    return example


def main():
    # Set seed for reproducibility
    seed = 42
    rng = random.Random(seed)
    
    print("Loading dataset from hamishivi/tulu_3_rewritten_100k...")
    dataset = load_dataset("hamishivi/tulu_3_rewritten_100k", split="train")
    
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Original columns: {dataset.column_names}")
    
    # Take a subset for testing (e.g., 1000 samples)
    subset_size = 1000
    dataset = dataset.select(range(min(subset_size, len(dataset))))
    print(f"Using subset of {len(dataset)} samples for testing")
    
    # Add tools column
    print("Adding tools column...")
    
    def add_tools_wrapper(example, idx):
        return add_tools_column(example, idx, rng)
    
    dataset = dataset.map(add_tools_wrapper, with_indices=True)
    
    print(f"New columns: {dataset.column_names}")
    
    # Print some statistics
    tools_counts = {"search": 0, "code": 0, "browse": 0, "none": 0, "empty": 0}
    for sample in dataset:
        tools = sample.get("tools")
        if tools is None:
            tools_counts["none"] += 1
        elif len(tools) == 0:
            tools_counts["empty"] += 1
        else:
            for tool in tools:
                if tool in tools_counts:
                    tools_counts[tool] += 1
    
    print("\nTools distribution:")
    for tool, count in tools_counts.items():
        print(f"  {tool}: {count}")
    
    # Print a few examples
    print("\nExample samples:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        messages_preview = str(sample.get("messages", []))[:100] + "..."
        print(f"  Sample {i}: tools={sample.get('tools')}, messages={messages_preview}")
    
    # Save to disk
    output_path = "data/tools_test_dataset"
    print(f"\nSaving dataset to {output_path}...")
    dataset.save_to_disk(output_path)
    print("Done!")
    
    # Also save as JSON for easy inspection
    json_output_path = "data/tools_test_dataset.jsonl"
    print(f"Also saving as JSONL to {json_output_path}...")
    dataset.to_json(json_output_path)
    print("Done!")


if __name__ == "__main__":
    main()
