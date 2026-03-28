"""
Convert MathArena HMMT datasets to RLVR format.

Usage:

python scripts/data/rlvr/matharena_hmmt.py --push_to_hub
python scripts/data/rlvr/matharena_hmmt.py --push_to_hub --hf_entity ai2-adapt-dev
python scripts/data/rlvr/matharena_hmmt.py --dataset_field_mode math
"""

from dataclasses import dataclass

import datasets
from huggingface_hub import HfApi
from transformers import HfArgumentParser

from open_instruct import utils as open_instruct_utils

SOURCE_DATASETS = (
    ("MathArena/hmmt_feb_2025", "math_hmmt_feb_2025"),
    ("MathArena/hmmt_nov_2025", "math_hmmt_nov_2025"),
    ("MathArena/brumo_2025", "math_brumo_2025"),
    ("MathArena/aime_2025", "math_aime_2025"),
)
SOURCE_DATASET_MAP = {name: label for name, label in SOURCE_DATASETS}


@dataclass
class Args:
    push_to_hub: bool = False
    hf_entity: str | None = None
    dataset_field_mode: str = "source"
    repo_name: str = "rlvr_matharena_2025"
    source_dataset: str = "all"


def _convert_dataset(source_dataset_name: str, dataset_label: str, dataset_field_mode: str) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        source_dataset_name, split="train", num_proc=open_instruct_utils.max_num_processes()
    )

    def process(example):
        answer_str = str(example["answer"])
        example["messages"] = [{"role": "user", "content": example["problem"]}]
        example["ground_truth"] = answer_str
        example["answer"] = answer_str
        example["dataset"] = "math" if dataset_field_mode == "math" else dataset_label
        example["source_dataset"] = source_dataset_name
        return example

    dataset = dataset.map(process, desc=f"Converting {source_dataset_name}")
    ordered_columns = [
        "messages",
        "ground_truth",
        "dataset",
        "source_dataset",
        "problem_idx",
        "problem",
        "answer",
        "problem_type",
    ]
    return dataset.select_columns([column for column in ordered_columns if column in dataset.column_names])


def main(args: Args):
    if args.dataset_field_mode not in {"source", "math"}:
        raise ValueError("--dataset_field_mode must be one of: source, math")

    if args.source_dataset == "all":
        selected_datasets = SOURCE_DATASETS
    else:
        if args.source_dataset not in SOURCE_DATASET_MAP:
            valid = ", ".join(sorted(SOURCE_DATASET_MAP))
            raise ValueError(f"--source_dataset must be 'all' or one of: {valid}")
        selected_datasets = [(args.source_dataset, SOURCE_DATASET_MAP[args.source_dataset])]

    converted = [
        _convert_dataset(source_name, dataset_label, args.dataset_field_mode) for source_name, dataset_label in selected_datasets
    ]
    output = datasets.DatasetDict({"train": datasets.concatenate_datasets(converted)})
    print(output)

    if args.push_to_hub:
        api = HfApi()
        if not args.hf_entity:
            args.hf_entity = api.whoami()["name"]
        repo_id = f"{args.hf_entity}/{args.repo_name}"
        print(f"Pushing dataset to Hub: {repo_id}")
        output.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj=__file__, path_in_repo="create_dataset.py", repo_type="dataset", repo_id=repo_id
        )


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    main(*parser.parse_args_into_dataclasses())
