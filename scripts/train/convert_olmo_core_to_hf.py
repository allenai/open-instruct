"""Convert an OLMo-core distributed checkpoint to HuggingFace format.

Example usage:
    uv run python scripts/train/convert_olmo_core_to_hf.py \
        --checkpoint-dir /path/to/checkpoint/step1000/model_and_optim \
        --hf-model-name allenai/Olmo-3-1125-32B \
        --olmo-core-model-name olmo3_32B \
        --output-dir /path/to/output/hf_checkpoint
"""

import argparse
import os
import shutil

import torch
import transformers

from open_instruct import logger_utils, olmo_core_utils

logger = logger_utils.setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert OLMo-core checkpoint weights to HuggingFace format")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to OLMo-core model checkpoint directory")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Backwards-compatible alias used for both --hf-model-name and --olmo-core-model-name",
    )
    parser.add_argument("--hf-model-name", default=None, help="HF model/config name used for export")
    parser.add_argument(
        "--olmo-core-model-name", default=None, help="OLMo-core config name used to build the source model"
    )
    parser.add_argument("--output-dir", required=True, help="Where to save the HF checkpoint")
    parser.add_argument("--tokenizer-name", default=None, help="HF tokenizer name (defaults to --hf-model-name)")
    parser.add_argument("--work-dir", default="/tmp/olmo_core_to_hf", help="Scratch directory for checkpoint loading")
    parser.add_argument(
        "--init-device", default="cpu", choices=["cpu", "cuda"], help="Device for materializing weights"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args()

    hf_model_name = args.hf_model_name or args.model_name
    olmo_core_model_name = args.olmo_core_model_name or args.model_name or hf_model_name
    if hf_model_name is None:
        raise ValueError("Pass --hf-model-name or the backwards-compatible --model-name")
    tokenizer_name = args.tokenizer_name or hf_model_name

    if args.init_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--init-device cuda requested, but CUDA is not available")

    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        elif os.listdir(args.output_dir):
            raise FileExistsError(f"Output directory exists and is non-empty: {args.output_dir}")

    logger.info(f"Loading HF config from {hf_model_name}")
    hf_config = transformers.AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
    vocab_size = hf_config.vocab_size

    logger.info(f"Building OLMo-core model config {olmo_core_model_name} on {args.init_device}")
    model_config = olmo_core_utils.get_transformer_config(olmo_core_model_name, vocab_size, attn_backend="torch")
    model = model_config.build(init_device=args.init_device)

    logger.info(f"Loading model-only weights from {args.checkpoint_dir}")
    state_dict = model.state_dict()
    olmo_core_utils._load_olmo_core_model_state_dict(args.checkpoint_dir, state_dict, args.work_dir)

    logger.info(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    logger.info(f"Saving HF checkpoint to {args.output_dir}")
    olmo_core_utils.save_state_dict_as_hf(state_dict, args.output_dir, hf_model_name, tokenizer)
    logger.info(f"Saved HuggingFace checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
