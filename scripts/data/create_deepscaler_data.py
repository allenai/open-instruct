"""
Convert agentica-org/DeepScaleR-Preview-Dataset to the OpenInstruct RLVR format.

Usage:

python scripts/data/create_deepscaler_data.py
python scripts/data/create_deepscaler_data.py --push_to_hub
python scripts/data/create_deepscaler_data.py --push_to_hub --repo_id mnoukhov/deepscaler_openinstruct
"""

from dataclasses import dataclass

import datasets
from huggingface_hub import HfApi
from transformers import HfArgumentParser

import open_instruct.utils as open_instruct_utils

SOURCE_DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
DEFAULT_REPO_ID = "mnoukhov/deepscaler_openinstruct"


@dataclass
class Args:
    push_to_hub: bool = False
    repo_id: str = DEFAULT_REPO_ID


def process(example: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": example["problem"]}],
        "ground_truth": example["answer"],
        "dataset": "math_deepscaler",
    }


def main(args: Args) -> None:
    dataset = datasets.load_dataset(SOURCE_DATASET, num_proc=open_instruct_utils.max_num_processes())
    dataset = dataset.map(
        process,
        remove_columns=dataset["train"].column_names,
        num_proc=open_instruct_utils.max_num_processes(),
        desc="Converting DeepScaleR to OpenInstruct format",
    )

    for split in dataset:
        dataset[split] = dataset[split].select_columns(["messages", "ground_truth", "dataset"])

    if args.push_to_hub:
        print(f"Pushing dataset to Hub: {args.repo_id}")
        dataset.push_to_hub(args.repo_id)
        HfApi().upload_file(
            path_or_fileobj=__file__, path_in_repo="create_dataset.py", repo_type="dataset", repo_id=args.repo_id
        )
    else:
        print(dataset)
        print(dataset["train"][0])


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    main(*parser.parse_args_into_dataclasses())
