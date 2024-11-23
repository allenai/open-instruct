from dataclasses import dataclass
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from open_instruct.utils import ArgumentParserPlus
"""
Run this file to cache models in a shared HF cache
(e.g., weka's `/weka/oe-adapt-default/allennlp/.cache/huggingface`)

python scripts/cache_model.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision olmo1124_13b_4k_finetune_epoch_2_lr_4e-6_loss_type_sum___weka_oe-training-default_ai2-llm_checkpoints_OLMo-medium_peteish13-anneal-from-596057-300B-legal-whammy-2-soup_step35773_olmo1124__42__1732223410
"""


@dataclass
class Args:
    model_name_or_path: str
    model_revision: str

def main(args: Args):
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