"""
This script is used to convert the AceCoder dataset to an RLVR format.

Usage:
python scripts/data/rlvr/acecoder.py --push_to_hub
python scripts/data/rlvr/acecoder.py --push_to_hub --hf_entity ai2-adapt-dev
"""

from dataclasses import dataclass

import datasets
from huggingface_hub import HfApi
from transformers import HfArgumentParser


@dataclass
class Args:
    push_to_hub: bool = False
    hf_entity: str | None = None


def main(args: Args):
    dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split="train", num_proc=max_num_processes())

    def process(example):
        example["messages"] = [{"role": "user", "content": example["question"]}]
        example["ground_truth"] = example["test_cases"]
        example["dataset"] = "ace_coder"
        return example

    dataset = dataset.map(process)
    # reorder columns
    dataset = dataset.select_columns(["messages", "ground_truth", "dataset"])

    if args.push_to_hub:
        api = HfApi()
        if not args.hf_entity:
            args.hf_entity = HfApi().whoami()["name"]
        repo_id = f"{args.hf_entity}/rlvr_acecoder"
        print(f"Pushing dataset to Hub: {repo_id}")
        dataset.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj=__file__, path_in_repo="create_dataset.py", repo_type="dataset", repo_id=repo_id
        )


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    main(*parser.parse_args_into_dataclasses())
