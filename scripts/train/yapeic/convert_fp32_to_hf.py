import argparse
import os
import sys
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_state_dict(fp32_path: str) -> torch.nn.modules.module.Module:
    # PyTorch 2.6 changed default weights_only=True; allow full unpickling for trusted files
    try:
        state = torch.load(fp32_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch without weights_only kwarg
        state = torch.load(fp32_path, map_location="cpu")

    if isinstance(state, dict):
        # Common DeepSpeed/Trainer variants
        for key in ("module", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def load_model_and_tokenizer(base_model: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    config = AutoConfig.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    return model, tokenizer


def save_hf(model, tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a consolidated fp32 state dict into a Hugging Face model folder."
    )
    parser.add_argument("--fp32_path", type=str, required=True, help="Path to consolidated fp32 .pt file")
    parser.add_argument("--base_model", type=str, required=True, help="Base model id (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for HF model")
    parser.add_argument("--strict", action="store_true", help="Use strict state_dict loading (default: False)")

    args = parser.parse_args()

    if not os.path.isfile(args.fp32_path):
        print(f"Error: fp32 file not found: {args.fp32_path}")
        sys.exit(1)

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = load_model_and_tokenizer(args.base_model)

    print(f"Loading state dict from: {args.fp32_path}")
    state_dict = load_state_dict(args.fp32_path)

    print("Applying state dict to model...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=args.strict)
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}{' ...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{' ...' if len(unexpected_keys) > 5 else ''}")

    print(f"Saving HF model to: {args.out_dir}")
    save_hf(model, tokenizer, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()


