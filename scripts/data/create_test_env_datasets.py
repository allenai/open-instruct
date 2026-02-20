#!/usr/bin/env python3
"""Create and upload datasets for RL environments to HuggingFace."""

import argparse
import json
from pathlib import Path

from datasets import Dataset


def create_counter_samples(num_samples: int = 100, nothink: bool = False) -> list[dict]:
    """Create samples for CounterEnv with varying target values."""
    samples = []
    suffix = " /nothink" if nothink else ""
    for i in range(num_samples):
        target = (i % 10) + 1
        samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are playing a counter game. Use the provided tools to increment or decrement the counter to reach the target value, then submit your answer.",
                },
                {
                    "role": "user",
                    "content": f"The counter starts at 0. Reach the target value of {target} and submit.{suffix}",
                },
            ],
            "ground_truth": str(target),
            "dataset": "passthrough",
            "env_config": {"task_id": str(target), "env_name": "counter"},
        })
    return samples


def create_guess_number_samples(num_samples: int = 100, nothink: bool = False) -> list[dict]:
    """Create samples for GuessNumberEnv with specific secret numbers."""
    samples = []
    user_content = "Guess the secret number. Use binary search strategy for efficiency."
    if nothink:
        user_content += " /nothink"
    for i in range(num_samples):
        secret = (i % 100) + 1
        samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are playing a number guessing game. Use the guess tool to find the secret number between 1 and 100. You will be told if your guess is too high or too low.",
                },
                {"role": "user", "content": user_content},
            ],
            "ground_truth": str(secret),
            "dataset": "passthrough",
            "env_config": {"task_id": str(secret), "env_name": "guess_number"},
        })
    return samples


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(samples)} samples to {path}")


def upload_to_huggingface(samples: list[dict], repo_id: str):
    dataset = Dataset.from_list(samples)
    dataset.push_to_hub(repo_id, private=False)
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Create and upload RL environment datasets")
    parser.add_argument("--namespace", default="hamishivi", help="HuggingFace namespace")
    parser.add_argument("--local-only", action="store_true", help="Only create local files, don't upload")
    parser.add_argument("--nothink", action="store_true", help="Add /nothink suffix to user messages (for Qwen3)")
    args = parser.parse_args()

    suffix = "-nothink" if args.nothink else ""
    env_datasets = [
        (f"rlenv-counter{suffix}", lambda n: create_counter_samples(n, nothink=args.nothink), 100),
        (f"rlenv-guess-number{suffix}", lambda n: create_guess_number_samples(n, nothink=args.nothink), 100),
    ]

    data_dir = Path(__file__).parent.parent.parent / "data" / "envs"

    for name, create_fn, num_samples in env_datasets:
        samples = create_fn(num_samples)
        local_name = name.replace("-", "_")
        save_jsonl(samples, data_dir / f"{local_name}_train.jsonl")

        if not args.local_only:
            repo_id = f"{args.namespace}/{name}"
            upload_to_huggingface(samples, repo_id)

    print("\nTo use with training:")
    if args.local_only:
        print(f"  --dataset_mixer_list {data_dir}/rlenv_counter_train.jsonl 1.0")
    else:
        print(f"  --dataset_mixer_list {args.namespace}/rlenv-counter 1.0")
    print("  --dataset_mixer_list_splits train")


if __name__ == "__main__":
    main()
