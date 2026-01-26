"""Convert an olmo-core distributed checkpoint to HuggingFace format.

Example usage:
    uv run python scripts/train/convert_olmo_core_to_hf.py \
        --checkpoint-dir /path/to/checkpoint/step1000 \
        --model-name allenai/OLMo-2-1124-7B \
        --output-dir /path/to/output/hf_checkpoint
"""

import argparse

import torch
import torch.distributed.checkpoint.state_dict as dcp_state_dict
import transformers

from open_instruct import olmo_core_utils


def main():
    parser = argparse.ArgumentParser(description="Convert olmo-core checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to olmo-core checkpoint directory")
    parser.add_argument("--model-name", required=True, help="HF model name or olmo-core config name")
    parser.add_argument("--output-dir", required=True, help="Where to save the HF checkpoint")
    parser.add_argument("--tokenizer-name", default=None, help="HF tokenizer name (defaults to --model-name)")
    args = parser.parse_args()

    tokenizer_name = args.tokenizer_name or args.model_name

    hf_config = transformers.AutoConfig.from_pretrained(args.model_name)
    vocab_size = hf_config.vocab_size

    model_config = olmo_core_utils.get_transformer_config(args.model_name, vocab_size)
    model = model_config.build(init_device="cpu")

    state_dict = {"model": model.state_dict()}
    dcp_state_dict.load_state_dict(state_dict, checkpoint_id=args.checkpoint_dir)
    model.load_state_dict(state_dict["model"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    olmo_core_utils.save_state_dict_as_hf(
        model_config, state_dict["model"], args.output_dir, args.model_name, tokenizer
    )
    print(f"Saved HuggingFace checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
