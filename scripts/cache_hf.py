from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct.utils import ArgumentParserPlus

"""
Run this file to cache models in a shared HF cache
(e.g., weka's `/weka/oe-adapt-default/allennlp/.cache/huggingface`)

python mason.py \
    --cluster ai2/jupiter ai2/saturn ai2/neptune --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority normal \
    --preemptible \
    --budget ai2/jupiter \
    --gpus 0 -- python scripts/cache_hf.py \
    --model_name_or_path "allenai/open_instruct_dev" \
    --model_revision "reward_modeling__1__1737836233" \
"""


@dataclass
class Args:
    model_name_or_path: str | None = None
    model_revision: str | None = None
    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    dataset_name: str | None = None
    """The name of the dataset to use (via the datasets library)."""
    dataset_mixer: dict | None = None
    """A dictionary of datasets (local or HF) to sample from."""
    dataset_mixer_list: list[str] | None = None
    """A list of datasets (local or HF) to sample from."""
    dataset_mix_dir: str | None = None
    """The directory to save the mixed dataset to disk."""
    dataset_config_name: str | None = None
    """The configuration name of the dataset to use (via the datasets library)."""


def main(args: Args):
    if args.dataset_name is not None:
        snapshot_download(args.dataset_name, repo_type="dataset")
    elif args.dataset_mixer is not None:
        for dataset_name in args.dataset_mixer:
            snapshot_download(dataset_name, repo_type="dataset")
    elif args.dataset_mixer_list is not None:
        for i in range(0, len(args.dataset_mixer_list), 2):
            snapshot_download(args.dataset_mixer_list[i], repo_type="dataset")
    # else:
    #     data_files = {}
    #     dataset_args = {}
    #     if args.train_file is not None:
    #         data_files["train"] = args.train_file
    #     raw_datasets = load_dataset(
    #         "json",
    #         data_files=data_files,
    #         **dataset_args,
    #     )
    # we don't tokenize the dataset here for simplicity, but we should at some point.

    if args.model_name_or_path is not None:
        AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path or args.model_name_or_path,
            revision=args.tokenizer_revision or args.model_revision,
        )
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, revision=args.model_revision, dtype=torch.bfloat16
        )


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args,))
    main(*parser.parse_args_into_dataclasses())
