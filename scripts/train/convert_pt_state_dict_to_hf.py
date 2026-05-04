"""Convert a raw OLMo-core ``model_state_dict.pt`` into HuggingFace format.

Recovers checkpoints saved by the now-removed
``save_state_dict_as_hf`` ``except NotImplementedError`` fallback, which
produced a ``model_state_dict.pt`` with raw OLMo-core parameter names
alongside the original HF ``config.json``.

Example:
    uv run python scripts/train/convert_pt_state_dict_to_hf.py \
        --src-dir /weka/.../qwen3_4b_base_dapo_20260501_153944/tmp-3m \
        --base-model Qwen/Qwen3-4B-Base
"""

import argparse
import os

import torch
import transformers

from open_instruct import logger_utils, olmo_core_utils

logger = logger_utils.setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True, help="Directory containing model_state_dict.pt.")
    parser.add_argument("--base-model", required=True, help="HF model id whose config matches the trained model.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to overwriting --src-dir.",
    )
    parser.add_argument("--tokenizer-name", default=None, help="Tokenizer to copy in. Defaults to --base-model.")
    parser.add_argument(
        "--state-dict-filename",
        default="model_state_dict.pt",
        help="Filename of the raw OLMo-core state dict inside --src-dir.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.src_dir
    state_dict_path = os.path.join(args.src_dir, args.state_dict_filename)
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"No state dict at {state_dict_path}")

    logger.info(f"Loading state dict from {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name or args.base_model)
    olmo_core_utils.save_state_dict_as_hf(
        model_config=None,
        state_dict=state_dict,
        save_dir=out_dir,
        original_model_name_or_path=args.base_model,
        tokenizer=tokenizer,
    )
    logger.info(f"Wrote HF checkpoint to {out_dir}")

    if out_dir == args.src_dir:
        os.remove(state_dict_path)
        logger.info(f"Removed legacy {state_dict_path}")


if __name__ == "__main__":
    main()
