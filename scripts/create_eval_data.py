#!/usr/bin/env python3
"""
Create evaluation JSONL files for various benchmarks.

Usage:
    # Create all datasets
    python scripts/create_eval_data.py --datasets aime simpleqa mbpp gpqa

    # Create specific dataset
    python scripts/create_eval_data.py --datasets simpleqa --output_dir data/

    # List available datasets
    python scripts/create_eval_data.py --list
"""

import argparse
import json
import os

from datasets import load_dataset


DATASET_CONFIGS = {
    "aime": {
        "sources": [
            ("math-ai/aime24", "test", None),
            ("math-ai/aime25", "test", None),
        ],
        "instruction": "Solve this math problem step by step. The answer is an integer from 0-999. Output your final answer inside \\boxed{}.",
        "format_fn": "format_aime",
    },
    "simpleqa": {
        "sources": [
            ("basicv8vc/SimpleQA", "test", None),
        ],
        "instruction": "Answer the following question accurately and concisely.",
        "format_fn": "format_simpleqa",
    },
    "mbpp": {
        "sources": [
            ("google-research-datasets/mbpp", "test", None),
        ],
        "instruction": "Write a Python function to solve the following problem. Include the function definition and make sure it passes the given test cases.",
        "format_fn": "format_mbpp",
    },
    "gpqa": {
        "sources": [
            ("Idavidrein/gpqa", "train", "gpqa_diamond"),  # GPQA diamond config
        ],
        "instruction": "Answer this graduate-level science question. Choose the correct answer from the options provided. Output your answer as a single letter (A, B, C, or D).",
        "format_fn": "format_gpqa",
    },
}


def format_aime(row, idx, source_name, instruction):
    problem = row.get("problem") or row.get("question") or row.get("Question")
    answer = row.get("answer") or row.get("solution") or row.get("Answer", "")
    return {
        "id": f"{source_name}_{row.get('id', idx)}",
        "problem": problem,
        "answer": str(answer),
        "source": source_name,
        "messages": [
            {"role": "user", "content": f"{instruction}\n\n{problem}"},
        ],
    }


def format_simpleqa(row, idx, source_name, instruction):
    problem = row.get("problem") or row.get("question") or ""
    answer = row.get("answer") or ""

    # Parse metadata if it's a string
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            import ast
            metadata = ast.literal_eval(metadata)
        except Exception:
            metadata = {}

    return {
        "id": f"simpleqa_{idx}",
        "problem": problem,
        "answer": str(answer),
        "source": "simpleqa",
        "topic": metadata.get("topic", "") if isinstance(metadata, dict) else "",
        "messages": [
            {"role": "user", "content": f"{instruction}\n\n{problem}"},
        ],
    }


def format_mbpp(row, idx, source_name, instruction):
    task_id = row.get("task_id", idx)
    text = row.get("text", "")
    code = row.get("code", "")
    test_list = row.get("test_list", [])

    # Build the prompt with test cases
    problem = f"{text}\n\nTest cases:\n"
    for test in test_list[:3]:  # Include up to 3 test cases
        problem += f"  {test}\n"

    return {
        "id": f"mbpp_{task_id}",
        "problem": text,
        "code": code,
        "test_list": test_list,
        "source": "mbpp",
        "messages": [
            {"role": "user", "content": f"{instruction}\n\n{problem}"},
        ],
    }


def format_gpqa(row, idx, source_name, instruction):
    import random
    question = row.get("Question", "")
    correct_answer = row.get("Correct Answer", "").strip()

    # Get all answers
    answers = [
        correct_answer,
        row.get("Incorrect Answer 1", "").strip(),
        row.get("Incorrect Answer 2", "").strip(),
        row.get("Incorrect Answer 3", "").strip(),
    ]
    answers = [a for a in answers if a]  # Remove empty

    # Shuffle answers and track correct letter
    random.seed(idx)  # Reproducible shuffle
    random.shuffle(answers)
    correct_letter = chr(ord("A") + answers.index(correct_answer))

    # Build choices
    choices = [f"{chr(ord('A') + i)}. {ans}" for i, ans in enumerate(answers)]
    problem = f"{question}\n\n" + "\n".join(choices)

    return {
        "id": f"gpqa_{idx}",
        "problem": question,
        "choices": choices,
        "answer": correct_letter,
        "correct_answer_text": correct_answer,
        "source": "gpqa_diamond",
        "subdomain": row.get("Subdomain", ""),
        "messages": [
            {"role": "user", "content": f"{instruction}\n\n{problem}"},
        ],
    }


FORMAT_FNS = {
    "format_aime": format_aime,
    "format_simpleqa": format_simpleqa,
    "format_mbpp": format_mbpp,
    "format_gpqa": format_gpqa,
}


def create_dataset(name: str, output_dir: str) -> str:
    """Create a dataset and return the output path."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[name]
    format_fn = FORMAT_FNS[config["format_fn"]]
    instruction = config["instruction"]

    data = []
    total_by_source = {}

    for source_info in config["sources"]:
        source_path, split, subset = source_info[0], source_info[1], source_info[2] if len(source_info) > 2 else None
        source_name = source_path.split("/")[-1]
        if subset:
            source_name = f"{source_name}_{subset}"
        print(f"Loading {source_path} ({split}, config={subset})...")

        try:
            if subset:
                ds = load_dataset(source_path, subset, split=split)
            else:
                ds = load_dataset(source_path, split=split)
        except Exception as e:
            print(f"  Warning: Failed to load {source_path}: {e}")
            continue

        count = 0
        for i, row in enumerate(ds):
            item = format_fn(row, i, source_name, instruction)
            data.append(item)
            count += 1

        total_by_source[source_name] = count
        print(f"  Loaded {count} examples from {source_name}")

    # Save
    output_path = os.path.join(output_dir, f"{name}.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(data)} total examples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create evaluation JSONL files")
    parser.add_argument("--datasets", nargs="+", default=[], help="Datasets to create (aime, simpleqa, mbpp, gpqa)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--all", action="store_true", help="Create all datasets")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, config in DATASET_CONFIGS.items():
            sources = [s[0] for s in config["sources"]]
            print(f"  {name}: {', '.join(sources)}")
        return

    datasets = args.datasets
    if args.all:
        datasets = list(DATASET_CONFIGS.keys())

    if not datasets:
        parser.print_help()
        return

    print(f"Creating datasets: {datasets}")
    print(f"Output directory: {args.output_dir}\n")

    for name in datasets:
        try:
            create_dataset(name, args.output_dir)
        except Exception as e:
            print(f"Error creating {name}: {e}")
        print()


if __name__ == "__main__":
    main()
