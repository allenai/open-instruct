"""
This script is used to convert the countdown dataset to standard SFT/RLVR format.
Note that we don't do any special processing to answer, and we will mainly
use it for generations.

Usage: 

python scripts/data/rlvr/countdown_rlvr.py --push_to_hub
python scripts/data/rlvr/countdown_rlvr.py --push_to_hub --hf_entity ai2-adapt-dev
"""

from dataclasses import dataclass
from typing import Optional

import datasets
from huggingface_hub import HfApi
from transformers import HfArgumentParser

@dataclass
class Args:
    push_to_hub: bool = False
    hf_entity: Optional[str] = None

def main(args: Args):
    dataset = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")

    def process(example):
        # we have to make it a nested list so the length is 1
        # because open-instruct checks that len(ground_truth) matches len(dataset)
        example["ground_truth"] = [[example["target"], *example["nums"]]]
        example["dataset"] = "countdown"
        prompt = (f"Using the numbers {example['nums']}, create an equation that equals {example['target']}. "
            "You can use basic arithmetic operations (+, -, *, /) "
            "and each number can only be used once, though not all numbers need to be used.")
        example["messages"] = [
            {"role": "user", "content": prompt},
        ]
        return example
    dataset = dataset.map(process)
    for key in dataset:  # reorder columns
        dataset[key] = dataset[key].select_columns(
            ["messages", "ground_truth", "dataset"]
        )
    
    if args.push_to_hub:
        api = HfApi()
        if not args.hf_entity:
            args.hf_entity = HfApi().whoami()["name"]
        repo_id = f"{args.hf_entity}/rlvr_countdown"
        print(f"Pushing dataset to Hub: {repo_id}")
        dataset.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo="create_dataset.py",
            repo_type="dataset",
            repo_id=repo_id,
        )

if __name__ == "__main__":
    parser = HfArgumentParser((Args))
    main(*parser.parse_args_into_dataclasses())
