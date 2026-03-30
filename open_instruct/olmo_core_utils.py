"""
OLMo-core utility functions and configuration mappings.

This module provides common utilities for working with OLMo-core models,
including model configuration mappings and helper functions.
"""

import os

import torch
import torch.distributed as dist
import transformers
from olmo_core.distributed.utils import is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.backend import has_flash_attn_3
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer import TransformerConfig

from open_instruct import logger_utils, utils
from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu
from open_instruct.dpo_utils import DatasetConfig

logger = logger_utils.setup_logger(__name__)

OLMO_MODEL_CONFIG_MAP: dict[str, str] = {
    "allenai/OLMo-2-0425-1B": "olmo2_1B_v2",
    "allenai/OLMo-2-1124-7B": "olmo2_7B",
    "allenai/OLMo-2-1124-13B": "olmo2_13B",
    "allenai/OLMo-2-0325-32B": "olmo2_32B",
    "allenai/Olmo-3-1025-7B": "olmo3_7B",
    "allenai/OLMoE-1B-7B-0924": "olmoe_1B_7B",
    "Qwen/Qwen3-0.6B": "qwen3_0_6B",
    "Qwen/Qwen3-0.6B-Base": "qwen3_0_6B",
    "Qwen/Qwen3-1.7B": "qwen3_1_7B",
    "Qwen/Qwen3-4B": "qwen3_4B",
    "Qwen/Qwen3-8B": "qwen3_8B",
    "Qwen/Qwen3-14B": "qwen3_14B",
    "Qwen/Qwen3-32B": "qwen3_32B",
}


def get_transformer_config(
    model_name_or_config: str, vocab_size: int, attn_backend: str | None = None
) -> TransformerConfig:
    """Get the appropriate TransformerConfig for a given model name or config name.

    Args:
        model_name_or_config: HuggingFace model name, path, or direct config name (e.g., 'olmo3_7B').
        vocab_size: Vocabulary size for the model.
        attn_backend: Attention backend name (e.g., 'flash_2', 'flash_3'). If None, uses default.

    Returns:
        TransformerConfig for the specified model.

    Raises:
        ValueError: If model/config not found.
    """
    config_name = OLMO_MODEL_CONFIG_MAP.get(model_name_or_config)
    if config_name is None:
        config_name = model_name_or_config

    if not hasattr(TransformerConfig, config_name):
        available_models = ", ".join(OLMO_MODEL_CONFIG_MAP.keys())
        available_configs = [
            name for name in dir(TransformerConfig) if name.startswith(("olmo", "qwen")) and not name.startswith("_")
        ]
        raise ValueError(
            f"Model/config '{model_name_or_config}' not found. "
            f"Available models: {available_models}. "
            f"Available config names: {', '.join(available_configs)}"
        )
    kwargs: dict = {"vocab_size": vocab_size}
    if attn_backend is not None:
        kwargs["attn_backend"] = AttentionBackendName(attn_backend)
    return getattr(TransformerConfig, config_name)(**kwargs)


def resolve_attn_backend(attn_backend: str) -> str:
    if attn_backend == "auto":
        device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
        is_h100 = "h100" in device_name or "h800" in device_name
        attn_backend = "flash_3" if (is_h100 and has_flash_attn_3()) else "flash_2"
        logger.info(f"Auto-detected attn_backend={attn_backend} for device: {device_name}")
    return attn_backend


def setup_model(
    model_name_or_path: str, config_name: str | None, attn_backend: str
) -> tuple[torch.nn.Module, TransformerConfig]:
    hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    vocab_size = hf_config.vocab_size
    logger.info(f"Building OLMo-core model with vocab_size={vocab_size}")
    resolved_backend = resolve_attn_backend(attn_backend)
    model_config = get_transformer_config(config_name or model_name_or_path, vocab_size, attn_backend=resolved_backend)
    model = model_config.build(init_device="cpu")
    return model, model_config


def load_dataset_distributed(
    args: DatasetConfig, tc: TokenizerConfig, transform_fn_args: list[dict], is_main_process: bool
):
    def _load():
        return get_cached_dataset_tulu(
            dataset_mixer_list=args.mixer_list,
            dataset_mixer_list_splits=args.mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.target_columns,
            dataset_cache_mode=args.cache_mode,
            dataset_config_hash=args.config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.local_cache_dir,
            dataset_skip_cache=args.skip_cache,
        )

    if is_main_process:
        dataset = _load()
    if is_distributed():
        dist.barrier()
    if not is_main_process:
        dataset = _load()
    return dataset  # noqa: F821 -- always bound: either is_main_process or not


def setup_tokenizer_and_cache(args, tc: TokenizerConfig):
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer
    args.local_cache_dir = os.path.abspath(args.local_cache_dir)
    if utils.is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.local_cache_dir = beaker_cache_dir
    return tokenizer


def save_state_dict_as_hf(model_config, state_dict, save_dir, original_model_name_or_path, tokenizer):
    try:
        unwrapped_model = model_config.build(init_device="cpu")
        unwrapped_model.load_state_dict(state_dict)
        save_hf_model(save_dir=save_dir, model_state_dict=state_dict, model=unwrapped_model, save_overwrite=True)
    except NotImplementedError as exc:
        logger.warning(
            "Falling back to raw state_dict save because HF export is unsupported for this OLMo-core model: %s", exc
        )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(save_dir, "model_state_dict.pt"))
    tokenizer.save_pretrained(save_dir)
    original_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path)
    original_config.save_pretrained(save_dir)
