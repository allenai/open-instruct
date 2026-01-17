"""
This script converts the General Thought dataset to a format compatible with Tulu (SFT and RL).
For now, its fairly hardcoded and in a beta form, use at your own risk.

use:
python scripts/data/convert_general_thought_to_tulu_thinker.py
"""

import random

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils

random_gen = random.Random(42)

ds = load_dataset(
    "natolambert/GeneralThought-430K-filtered", split="train", num_proc=open_instruct_utils.max_num_processes()
)
new_data = []

for sample in ds:
    question = sample["question"]
    model_reasoning = sample["model_reasoning"]
    model_answer = sample["model_answer"]
    if not question or not model_reasoning or not model_answer:
        continue
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"<think>{model_reasoning}</think><answer>{model_answer}</answer>"},
    ]
    new_data.append({"messages": messages, "ground_truth": model_answer, "dataset": "tulu_thinker"})

random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("hamishivi/GeneralThought-430K-filtered-thinker")
