"""Convert a random subset of agentica-org/DeepScaleR-Preview-Dataset to OpenInstruct RLVR format.

The output schema (messages / ground_truth / dataset) matches what
scripts/data/rlvr/aime_pass_at_k_dataset.py expects as input.

Usage:

python scripts/data/create_deepscaler_subset_rlvr.py
python scripts/data/create_deepscaler_subset_rlvr.py --push_to_hub
python scripts/data/create_deepscaler_subset_rlvr.py --push_to_hub \
    --repo_id mnoukhov/deepscaler-10k-rlvr --num_examples 10000 --seed 42
"""

from dataclasses import dataclass

import datasets
from huggingface_hub import HfApi
from transformers import HfArgumentParser

import open_instruct.utils as open_instruct_utils

SOURCE_DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
DEFAULT_REPO_ID = "mnoukhov/deepscaler-10k-rlvr"


@dataclass
class Args:
    push_to_hub: bool = False
    repo_id: str = DEFAULT_REPO_ID
    num_examples: int = 10000
    seed: int = 42


def process(example: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": example["problem"].strip()}],
        "ground_truth": example["answer"],
        "dataset": "math_deepscaler",
    }


def main(args: Args) -> None:
    dataset = datasets.load_dataset(SOURCE_DATASET, split="train", num_proc=open_instruct_utils.max_num_processes())

    dataset = dataset.shuffle(seed=args.seed)
    if args.num_examples is not None and args.num_examples < len(dataset):
        dataset = dataset.select(range(args.num_examples))

    dataset = dataset.map(
        process,
        remove_columns=dataset.column_names,
        num_proc=open_instruct_utils.max_num_processes(),
        desc="Converting DeepScaleR to OpenInstruct format",
    )
    dataset = dataset.select_columns(["messages", "ground_truth", "dataset"])

    if args.push_to_hub:
        print(f"Pushing {len(dataset)} examples to Hub: {args.repo_id}")
        dataset.push_to_hub(args.repo_id, split="train")
        HfApi().upload_file(
            path_or_fileobj=__file__, path_in_repo="create_dataset.py", repo_type="dataset", repo_id=args.repo_id
        )
    else:
        print(dataset)
        print(dataset[0])


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    main(*parser.parse_args_into_dataclasses())
