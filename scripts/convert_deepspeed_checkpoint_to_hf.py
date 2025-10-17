#!/usr/bin/env python3
"""
Utility to turn a DeepSpeed ZeRO-2/3 checkpoint produced by grpo_fast.py into a Hugging Face model folder.

Usage example:
    python scripts/convert_deepspeed_checkpoint_to_hf.py \
        --checkpoint-dir /path/to/checkpoint_state_dir \
        --output-dir /path/to/exported_model \
        --model-config /path/to/base_model_or_config \
        --tokenizer /path/to/tokenizer \
        --tag global_step16384
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
except ImportError as exc:  # pragma: no cover - makes failure mode obvious to user
    raise SystemExit(
        "Could not import DeepSpeed. Install deepspeed in the current environment before running this script."
    ) from exc

try:
    from open_instruct.dataset_transformation import CHAT_TEMPLATES
except Exception:  # pragma: no cover - script is still useful without templates
    CHAT_TEMPLATES = {}

try:
    from open_instruct.model_utils import get_olmo3_generation_config
except Exception:  # pragma: no cover - optional helper
    get_olmo3_generation_config = None


def _resolve_checkpoint_leaf(checkpoint_dir: Path, explicit_tag: Optional[str]) -> tuple[Path, Optional[str], Path]:
    """
    Determine which folder actually holds the DeepSpeed shards.

    Returns a tuple of (root_dir, tag, leaf_dir).
    """
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    tag = explicit_tag
    if tag is None:
        latest_file = checkpoint_dir / "latest"
        if latest_file.is_file():
            tag = latest_file.read_text().strip()

    # If a tag is set, assume that `checkpoint_dir/tag` holds the shard files.
    if tag:
        leaf_dir = checkpoint_dir / tag
    else:
        leaf_dir = checkpoint_dir

    if not leaf_dir.exists():
        raise FileNotFoundError(f"Derived checkpoint leaf {leaf_dir} does not exist.")

    shard = leaf_dir / "mp_rank_00_model_states.pt"
    if not shard.exists():
        # Some jobs nest the shards a level deeper, so search for a unique match.
        matches = list(leaf_dir.glob("**/mp_rank_00_model_states.pt"))
        if len(matches) == 1:
            leaf_dir = matches[0].parent
        elif len(matches) == 0:
            raise FileNotFoundError(
                f"Could not find mp_rank_00_model_states.pt under {leaf_dir}. "
                "Make sure you are pointing at a DeepSpeed ZeRO checkpoint."
            )
        else:
            raise RuntimeError(
                f"Found multiple shard folders in {leaf_dir}. "
                "Specify --tag to disambiguate which checkpoint to convert."
            )

    return checkpoint_dir, tag, leaf_dir


def _parse_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None

    normalized = dtype_str.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    if normalized not in mapping:
        valid = ", ".join(sorted(set(mapping.keys())))
        raise ValueError(f"Unknown dtype '{dtype_str}'. Expected one of: {valid}")
    return mapping[normalized]


def _load_state_dict(leaf_dir: Path, tag: Optional[str]) -> dict[str, torch.Tensor]:
    """
    Load the aggregated state dict from the DeepSpeed checkpoint.
    """
    # The DeepSpeed utility accepts either the leaf directory that holds the shards
    # (when tag=None) or the checkpoint root plus an explicit tag.
    if tag is None:
        state_dict = load_state_dict_from_zero_checkpoint(str(leaf_dir))
    else:
        state_dict = load_state_dict_from_zero_checkpoint(str(leaf_dir.parent), tag=tag)
    return state_dict


def _maybe_set_chat_template(tokenizer, chat_template_name: Optional[str]) -> None:
    if not chat_template_name:
        return

    template = None
    if chat_template_name in CHAT_TEMPLATES:
        template = CHAT_TEMPLATES[chat_template_name]
    else:
        path = Path(chat_template_name)
        if path.is_file():
            template = path.read_text()

    if template is None:
        raise ValueError(
            f"Could not resolve chat template '{chat_template_name}'. "
            "Pass a key from CHAT_TEMPLATES or a path to a .jinja template file."
        )

    tokenizer.chat_template = template


def _maybe_add_generation_config(model, tokenizer, chat_template_name: Optional[str]) -> None:
    if not chat_template_name or get_olmo3_generation_config is None:
        return
    normalized = chat_template_name.lower()
    if "olmo" in normalized:
        gen_config = get_olmo3_generation_config(tokenizer)
        model.generation_config = gen_config


def convert_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    model_config_source: str,
    tokenizer_source: Optional[str],
    tag: Optional[str],
    dtype_str: Optional[str],
    trust_remote_code: bool,
    chat_template_name: Optional[str],
) -> None:
    checkpoint_dir, tag, leaf_dir = _resolve_checkpoint_leaf(checkpoint_dir, tag)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = _parse_dtype(dtype_str)

    state_dict = _load_state_dict(leaf_dir, tag)

    config = AutoConfig.from_pretrained(model_config_source, trust_remote_code=trust_remote_code)
    if dtype is not None:
        config.torch_dtype = dtype
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[warning] Missing keys while loading state dict ({len(missing_keys)}): {missing_keys[:8]}")
    if unexpected_keys:
        print(f"[warning] Unexpected keys in state dict ({len(unexpected_keys)}): {unexpected_keys[:8]}")
    model.tie_weights()

    if tokenizer_source is None:
        tokenizer_source = model_config_source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)

    _maybe_set_chat_template(tokenizer, chat_template_name)
    _maybe_add_generation_config(model, tokenizer, chat_template_name)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "source_checkpoint": str(checkpoint_dir),
        "tag": tag,
        "model_config_source": model_config_source,
        "tokenizer_source": tokenizer_source,
        "dtype": dtype_str,
    }
    with (output_dir / "conversion_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Converted checkpoint from {leaf_dir} into {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Directory that holds DeepSpeed checkpoint state.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the Hugging Face model (created if it does not exist).",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path or identifier for AutoConfig.from_pretrained to define the model architecture.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path or identifier for the tokenizer. Defaults to the same value as --model-config.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional DeepSpeed tag (e.g., global_step1234). If omitted, uses the 'latest' file when present.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Optional torch dtype to record in the config (e.g., bf16, fp16).",
    )
    parser.add_argument(
        "--chat-template-name",
        default=None,
        help="Tokenizer chat template key (from CHAT_TEMPLATES) or path to a template file to attach.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True when loading config/tokenizer/model.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    convert_checkpoint(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        model_config_source=args.model_config,
        tokenizer_source=args.tokenizer,
        tag=args.tag,
        dtype_str=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        chat_template_name=args.chat_template_name,
    )


if __name__ == "__main__":
    main()

