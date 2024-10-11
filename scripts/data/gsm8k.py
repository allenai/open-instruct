"""
This script is used to convert the GSM8K dataset to standard SFT format.
Note that we don't do any special processing to answer, and we will mainly
use it for generations.

Usage: 

python scripts/data/gsm8k.py --push_to_hub
python scripts/data/gsm8k.py --push_to_hub --hf_entity ai2-adapt-dev
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
    dataset = datasets.load_dataset("gsm8k", "main")
    
    print(dataset)
    def process(example):
        example["messages"] = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        return example
    dataset = dataset.map(process)
    for key in dataset:  # reorder columns
        dataset[key] = dataset[key].select_columns(
            ["messages", "question", "answer"]
        )
    
    if args.push_to_hub:
        api = HfApi()
        if not args.hf_entity:
            args.hf_entity = HfApi().whoami()["name"]
        repo_id = f"{args.hf_entity}/gsm8k"
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
