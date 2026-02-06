"""
OLMo-core utility functions and configuration mappings.

This module provides common utilities for working with OLMo-core models,
including model configuration mappings and helper functions.
"""

import transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.fla.layer import FLAConfig
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

OLMO_MODEL_CONFIG_MAP: dict[str, str] = {
    "allenai/OLMo-2-0425-1B": "olmo2_1B_v2",
    "allenai/OLMo-2-1124-7B": "olmo2_7B",
    "allenai/OLMo-2-1124-13B": "olmo2_13B",
    "allenai/OLMo-2-0325-32B": "olmo2_32B",
    "allenai/Olmo-3-1025-7B": "olmo3_7B",
    "allenai/OLMoE-1B-7B-0924": "olmoe_1B_7B",
    "Qwen/Qwen3-0.6B": "qwen3_0_6B",
    "Qwen/Qwen3-1.7B": "qwen3_1_7B",
    "Qwen/Qwen3-4B": "qwen3_4B",
    "Qwen/Qwen3-8B": "qwen3_8B",
    "Qwen/Qwen3-14B": "qwen3_14B",
    "Qwen/Qwen3-32B": "qwen3_32B",
    "/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/"
    "OLMo3.1-7B-6T-30h-long-context-drope/step23842-hf": "olmo3_7B_hybrid",
}


def get_olmo3_7b_hybrid_config(vocab_size: int, attn_backend: str | None = None) -> TransformerConfig:
    """Build the OLMo 3.1 7B hybrid config with GatedDeltaNet FLA layers.

    Based on src/scripts/train/linear-rnns/OLMo3.1-7B-hybrid.py from olmo-core.
    """
    kwargs: dict = {"vocab_size": vocab_size}
    if attn_backend is not None:
        kwargs["attn_backend"] = AttentionBackendName(attn_backend)

    config = TransformerConfig.olmo3_7B(**kwargs)

    # Remove 2 heads and scale down d_model to compensate for extra FLA params
    remove_heads = 2
    config.d_model -= remove_heads * 128
    config.block.attention.n_heads -= remove_heads

    # Configure hybrid FLA blocks (attention every 4th layer: indices 3, 7, 11, ...)
    config.block.name = TransformerBlockType.fla_hybrid
    config.block.fla_hybrid_attention_indices = [i for i in range(config.n_layers) if i % 4 == 3]

    # Configure GatedDeltaNet FLA layers
    config.block.fla = FLAConfig(
        name="GatedDeltaNet",
        dtype=config.dtype,
        fla_layer_kwargs={
            "head_dim": int(0.75 * config.d_model / config.block.attention.n_heads),
            "use_gate": True,
            "allow_neg_eigval": True,
        },
    )

    return config


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

    # Handle hybrid config specially (no predefined config in TransformerConfig)
    if config_name == "olmo3_7B_hybrid":
        return get_olmo3_7b_hybrid_config(vocab_size, attn_backend)

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


def save_state_dict_as_hf(model_config, state_dict, save_dir, original_model_name_or_path, tokenizer):
    unwrapped_model = model_config.build(init_device="cpu")
    unwrapped_model.load_state_dict(state_dict)
    save_hf_model(save_dir=save_dir, model_state_dict=state_dict, model=unwrapped_model, save_overwrite=True)
    tokenizer.save_pretrained(save_dir)
    original_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path)
    original_config.save_pretrained(save_dir)
