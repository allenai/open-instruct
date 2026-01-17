import random

from datasets import Dataset, load_dataset
from tqdm import tqdm

import open_instruct.utils as open_instruct_utils

random_gen = random.Random(42)

dataset = load_dataset(
    "agentica-org/DeepScaleR-Preview-Dataset", split="train", num_proc=open_instruct_utils.max_num_processes()
)

# reqular dataset
new_data = []
for sample in tqdm(dataset):
    new_data.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": sample["problem"].strip()
                    + " Let's think step by step and output the final answer within \\boxed{}.",
                }
            ],
            "ground_truth": sample["answer"],
            "dataset": "math",
        }
    )
random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("ai2-adapt-dev/deepscaler-gt")

dataset = load_dataset(
    "agentica-org/DeepScaleR-Preview-Dataset", split="train", num_proc=open_instruct_utils.max_num_processes()
)
# 4k length only
new_data = []
for sample in tqdm(dataset):
    sampled_length = random_gen.sample(range(100, 4096), 1)[0]
    new_data.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": sample["problem"].strip()
                    + " Let's think step by step and output the final answer within \\boxed{}. "
                    + f"Think for {sampled_length} tokens.",
                }
            ],
            "ground_truth": str(sampled_length),
            "dataset": "max_length",
        }
    )
random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("ai2-adapt-dev/deepscaler_gt_random_max_length_only")

dataset = load_dataset(
    "agentica-org/DeepScaleR-Preview-Dataset", split="train", num_proc=open_instruct_utils.max_num_processes()
)
# 4k length and solution
new_data = []
for sample in tqdm(dataset):
    sampled_length = random_gen.sample(range(100, 4096), 1)[0]
    new_data.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": sample["problem"].strip()
                    + " Let's think step by step and output the final answer within \\boxed{}. "
                    + f"Think for {sampled_length} tokens.",
                }
            ],
            "ground_truth": [sample["answer"], str(sampled_length)],
            "dataset": ["math", "max_length"],
        }
    )
random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("ai2-adapt-dev/deepscaler_gt_with_random_max_length")

dataset = load_dataset(
    "agentica-org/DeepScaleR-Preview-Dataset", split="train", num_proc=open_instruct_utils.max_num_processes()
)
# 8k length only
new_data = []
for sample in tqdm(dataset):
    sampled_length = random_gen.sample(range(100, 8192), 1)[0]
    new_data.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": sample["problem"].strip()
                    + " Let's think step by step and output the final answer within \\boxed{}. "
                    + f"Think for {sampled_length} tokens.",
                }
            ],
            "ground_truth": str(sampled_length),
            "dataset": "max_length",
        }
    )
random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("ai2-adapt-dev/deepscaler_gt_random_max_length_only_8192")

dataset = load_dataset(
    "agentica-org/DeepScaleR-Preview-Dataset", split="train", num_proc=open_instruct_utils.max_num_processes()
)
# 8k length and solution
new_data = []
for sample in tqdm(dataset):
    sampled_length = random_gen.sample(range(100, 8192), 1)[0]
    new_data.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": sample["problem"].strip()
                    + " Let's think step by step and output the final answer within \\boxed{}. "
                    + f"Think for {sampled_length} tokens.",
                }
            ],
            "ground_truth": [sample["answer"], str(sampled_length)],
            "dataset": ["math", "max_length"],
        }
    )
random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("ai2-adapt-dev/deepscaler_gt_with_random_max_length_8192")
