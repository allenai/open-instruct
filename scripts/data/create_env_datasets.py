#!/usr/bin/env python3
"""Create datasets for RL environments."""

import json
from pathlib import Path


def create_counter_dataset(output_path: Path, num_samples: int = 100):
    """Create a dataset for CounterEnv with varying target values."""
    samples = []

    for i in range(num_samples):
        target = (i % 10) + 1  # Targets 1-10
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are playing a counter game. Use the provided tools to increment or decrement the counter to reach the target value, then submit your answer.",
                },
                {
                    "role": "user",
                    "content": f"The counter starts at 0. Reach the target value of {target} and submit.",
                },
            ],
            "ground_truth": str(target),
            "dataset": "env_last",
            "env_config": {"task_id": str(target)},
        }
        samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {len(samples)} samples at {output_path}")


def create_guess_number_dataset(output_path: Path, num_samples: int = 100):
    """Create a dataset for GuessNumberEnv with specific secret numbers."""
    samples = []

    for i in range(num_samples):
        secret = (i % 100) + 1  # Secrets 1-100
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are playing a number guessing game. Use the guess tool to find the secret number between 1 and 100. You will be told if your guess is too high or too low.",
                },
                {"role": "user", "content": "Guess the secret number. Use binary search strategy for efficiency."},
            ],
            "ground_truth": str(secret),
            "dataset": "env_last",
            "env_config": {"task_id": str(secret)},
        }
        samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {len(samples)} samples at {output_path}")


def create_wordle_dataset(output_path: Path, num_samples: int = 100):
    """Create a dataset for Wordle environment via OpenEnv."""
    # Common 5-letter words for Wordle
    words = [
        "apple",
        "beach",
        "chair",
        "dance",
        "eagle",
        "flame",
        "grape",
        "house",
        "image",
        "juice",
        "knife",
        "lemon",
        "music",
        "night",
        "ocean",
        "piano",
        "queen",
        "river",
        "stone",
        "tiger",
        "ultra",
        "video",
        "water",
        "xerox",
        "youth",
        "zebra",
        "alarm",
        "brain",
        "crane",
        "dream",
        "earth",
        "flash",
        "ghost",
        "heart",
        "irony",
        "jelly",
        "karma",
        "light",
        "money",
        "noise",
        "olive",
        "peace",
        "quiet",
        "radio",
        "smile",
        "toast",
        "unity",
        "voice",
        "world",
        "young",
    ]

    samples = []
    for i in range(num_samples):
        word = words[i % len(words)]
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are playing Wordle. Guess a 5-letter word. After each guess, you'll see which letters are correct (green), in the word but wrong position (yellow), or not in the word (gray). You have 6 attempts.",
                },
                {"role": "user", "content": "Play Wordle! Start by guessing a common 5-letter word."},
            ],
            "ground_truth": word,
            "dataset": "env_last",
            "env_config": {"task_id": word},
        }
        samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {len(samples)} samples at {output_path}")


def create_appworld_dataset(output_path: Path, num_samples: int = 50):
    """Create a dataset for AppWorld environment."""
    # AppWorld task descriptions (simplified examples)
    tasks = [
        ("Send an email to john@example.com about the meeting", "email_task_1"),
        ("Schedule a reminder for tomorrow at 9am", "reminder_task_1"),
        ("Create a new note titled 'Shopping List'", "note_task_1"),
        ("Check the weather forecast for New York", "weather_task_1"),
        ("Set an alarm for 7:30 AM", "alarm_task_1"),
        ("Search for nearby coffee shops", "search_task_1"),
        ("Add milk to the shopping list", "list_task_1"),
        ("Call the nearest pizza restaurant", "call_task_1"),
        ("Book a table for 2 at 7pm", "booking_task_1"),
        ("Play some relaxing music", "music_task_1"),
    ]

    samples = []
    for i in range(num_samples):
        task_desc, task_id = tasks[i % len(tasks)]
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps users with tasks on their phone. Use the available tools to complete the user's request.",
                },
                {"role": "user", "content": task_desc},
            ],
            "ground_truth": task_id,
            "dataset": "env_last",
            "env_config": {"task_id": task_id},
        }
        samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {len(samples)} samples at {output_path}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data" / "envs"

    create_counter_dataset(data_dir / "counter_train.jsonl", num_samples=100)
    create_guess_number_dataset(data_dir / "guess_number_train.jsonl", num_samples=100)
    create_wordle_dataset(data_dir / "wordle_train.jsonl", num_samples=100)
    create_appworld_dataset(data_dir / "appworld_train.jsonl", num_samples=50)

    print(f"\nDatasets created in {data_dir}")
    print("\nTo use with training:")
    print("  --dataset_mixer_list path/to/counter_train.jsonl 1.0")
    print("  --dataset_mixer_list_splits train")
