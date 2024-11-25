from dataclasses import dataclass
from functools import partial
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from open_instruct.utils import ArgumentParserPlus, get_datasets
from huggingface_hub import snapshot_download


"""
Run this file to cache models in a shared HF cache
(e.g., weka's `/weka/oe-adapt-default/allennlp/.cache/huggingface`)

python mason.py \
    --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale ai2/neptune-cirrascale --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority normal \
    --preemptible \
    --budget ai2/allennlp \
    --gpus 0 -- python scripts/cache_hf.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision olmo1124_13b_4k_finetune_epoch_2_7.5e-06__42__1732416565 \
    --dataset_mixer_list ai2-adapt-dev/WildChat-prefs-280824_olmo2_7b 1.0 ai2-adapt-dev/sft_v3.9_if_taxonomy_olmo2_7b 1.0 ai2-adapt-dev/sft_v3.9_p0_olmo2_7b 1.0 ai2-adapt-dev/sft_v3.9_p1_olmo2_7b 1.0 ai2-adapt-dev/ultrafeedback_cleaned_olmo2_7b 1.0 allenai/tulu-3-pref-personas-instruction-following 1.0
"""


@dataclass
class Args:
    model_name_or_path: Optional[str] = None
    model_revision: Optional[str] = None
    dataset_name: Optional[str] = None
    """The name of the dataset to use (via the datasets library)."""
    dataset_mixer: Optional[dict] = None
    """A dictionary of datasets (local or HF) to sample from."""
    dataset_mixer_list: Optional[list[str]] = None
    """A list of datasets (local or HF) to sample from."""
    dataset_mix_dir: Optional[str] = None
    """The directory to save the mixed dataset to disk."""
    dataset_config_name: Optional[str] = None
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
            args.model_name_or_path,
            revision=args.model_revision,
        )
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            torch_dtype=torch.bfloat16,
        )
    
    

if __name__ == "__main__":
    parser = ArgumentParserPlus((Args,))
    main(*parser.parse_args_into_dataclasses())