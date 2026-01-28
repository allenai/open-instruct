#!/usr/bin/env python3
"""
Create AIME 2024+2025 JSONL file for inference.

Usage:
    python scripts/create_aime_data.py --output_file data/aime.jsonl

Then run inference:
    python scripts/inference_with_tools.py \
        --input_file data/aime.jsonl \
        --output_file results/aime_results.jsonl \
        --model_name_or_path allenai/OLMo-2-1124-7B-Instruct
"""

import argparse
import json
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Create AIME 2024+2025 JSONL file")
    parser.add_argument("--output_file", type=str, default="data/aime.jsonl")
    parser.add_argument("--system_prompt", type=str, default="Solve this math problem. Output your final answer inside \\boxed{}.")
    args = parser.parse_args()

    data = []

    # Load AIME 2024
    print("Loading math-ai/aime24...")
    ds24 = load_dataset("math-ai/aime24", split="test")
    for i, row in enumerate(ds24):
        problem = row.get("problem") or row.get("question")
        data.append({
            "id": f"aime24_{row.get('id', i)}",
            "problem": problem,
            "answer": str(row.get("answer") or row.get("solution", "")),
            "source": "aime24",
            "messages": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": problem},
            ],
        })

    # Load AIME 2025
    print("Loading math-ai/aime25...")
    ds25 = load_dataset("math-ai/aime25", split="test")
    for i, row in enumerate(ds25):
        problem = row.get("problem") or row.get("question")
        data.append({
            "id": f"aime25_{row.get('id', i)}",
            "problem": problem,
            "answer": str(row.get("answer") or row.get("solution", "")),
            "source": "aime25",
            "messages": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": problem},
            ],
        })

    # Save
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(data)} problems to {args.output_file}")
    print(f"  - AIME 2024: {len(ds24)} problems")
    print(f"  - AIME 2025: {len(ds25)} problems")


if __name__ == "__main__":
    main()
