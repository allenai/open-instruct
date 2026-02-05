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
        target = (i % 10) + 1  # Targets 1-10
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
            "dataset": "env_last",
            "env_config": {"task_id": str(target)},
        })
    return samples


def create_guess_number_samples(num_samples: int = 100, nothink: bool = False) -> list[dict]:
    """Create samples for GuessNumberEnv with specific secret numbers."""
    samples = []
    user_content = "Guess the secret number. Use binary search strategy for efficiency."
    if nothink:
        user_content += " /nothink"
    for i in range(num_samples):
        secret = (i % 100) + 1  # Secrets 1-100
        samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are playing a number guessing game. Use the guess tool to find the secret number between 1 and 100. You will be told if your guess is too high or too low.",
                },
                {"role": "user", "content": user_content},
            ],
            "ground_truth": str(secret),
            "dataset": "env_last",
            "env_config": {"task_id": str(secret)},
        })
    return samples


def create_appworld_samples(tasks_dir: Path | None = None, nothink: bool = False) -> list[dict]:
    """Create samples for AppWorld environment from task specs.

    AppWorld has 733 tasks across 9 apps (Spotify, Amazon, Venmo, etc.).
    Each task has a supervisor who needs help with API-based operations.
    """
    if tasks_dir is None:
        tasks_dir = Path(__file__).parent.parent.parent / "data" / "tasks"

    if not tasks_dir.exists():
        print(f"Warning: AppWorld tasks directory not found at {tasks_dir}")
        print("Skipping appworld dataset creation.")
        return []

    samples = []
    system_prompt = """You are an AI assistant that helps users complete tasks in the AppWorld environment.

You can execute Python code using the `execute` tool. Available APIs:
- Use `apis.{app_name}.{api_name}(**params)` to call APIs
- Use `apis.api_docs.show_api_doc(app_name, api_name)` to look up API documentation
- Use `apis.supervisor.complete_task()` when done (with `answer=` if the task requires an answer)

Think step by step about what APIs you need to call, then execute code to complete the task."""
    if nothink:
        system_prompt = "/nothink\n\n" + system_prompt

    task_dirs = sorted(tasks_dir.iterdir())
    for task_dir in task_dirs:
        specs_file = task_dir / "specs.json"
        if not specs_file.exists():
            continue

        with open(specs_file) as f:
            specs = json.load(f)

        task_id = task_dir.name
        instruction = specs.get("instruction", "")
        supervisor = specs.get("supervisor", {})

        # Create user message with task details
        user_content = f"""Task: {instruction}

Supervisor: {supervisor.get('first_name', '')} {supervisor.get('last_name', '')}
Email: {supervisor.get('email', '')}

Complete this task by calling the appropriate APIs."""

        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "ground_truth": task_id,
            "dataset": "env_last",
            "env_config": {"task_id": task_id},
        })

    return samples


def create_wordle_samples(num_samples: int = 100, nothink: bool = False) -> list[dict]:
    """Create samples for Wordle environment via OpenEnv."""
    words = [
        "apple", "beach", "chair", "dance", "eagle", "flame", "grape", "house", "image", "juice",
        "knife", "lemon", "music", "night", "ocean", "piano", "queen", "river", "stone", "tiger",
        "ultra", "video", "water", "xerox", "youth", "zebra", "alarm", "brain", "crane", "dream",
        "earth", "flash", "ghost", "heart", "irony", "jelly", "karma", "light", "money", "noise",
        "olive", "peace", "quiet", "radio", "smile", "toast", "unity", "voice", "world", "young",
    ]
    samples = []
    # Instructions go in the user message
    instructions = "Wordle: Guess a 5-letter word. Output ONLY a bracketed word like [crane]. You'll get feedback showing each letter: G=correct position, Y=in word but wrong position, X=not in word. Use the feedback to make better guesses."
    for i in range(num_samples):
        word = words[i % len(words)]
        # Build messages list - system prompt only contains /nothink when enabled
        messages = []
        if nothink:
            messages.append({"role": "system", "content": "/nothink"})
        messages.append({"role": "user", "content": instructions})
        samples.append({
            "messages": messages,
            "ground_truth": word,
            "dataset": "env_last",
            "env_config": {"task_id": word},
        })
    return samples


def save_jsonl(samples: list[dict], path: Path):
    """Save samples as JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(samples)} samples to {path}")


def upload_to_huggingface(samples: list[dict], repo_id: str):
    """Upload dataset to HuggingFace Hub."""
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
    # (name, create_fn, num_samples) - num_samples=None means use all available
    datasets = [
        (f"rlenv-counter{suffix}", lambda n: create_counter_samples(n, nothink=args.nothink), 100),
        (f"rlenv-guess-number{suffix}", lambda n: create_guess_number_samples(n, nothink=args.nothink), 100),
        (f"rlenv-wordle{suffix}", lambda n: create_wordle_samples(n, nothink=args.nothink), 100),
        (f"rlenv-appworld{suffix}", lambda _: create_appworld_samples(nothink=args.nothink), None),
    ]

    data_dir = Path(__file__).parent.parent.parent / "data" / "envs"

    for name, create_fn, num_samples in datasets:
        samples = create_fn(num_samples)
        if not samples:
            continue  # Skip if no samples (e.g., appworld without task files)

        # Save locally
        local_name = name.replace("-", "_")
        save_jsonl(samples, data_dir / f"{local_name}_train.jsonl")

        # Upload to HuggingFace
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
