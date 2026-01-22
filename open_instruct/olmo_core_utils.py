"""
OLMo-core utility functions and configuration mappings.

This module provides common utilities for working with OLMo-core models,
including model configuration mappings and helper functions.
"""

from olmo_core.nn.transformer import TransformerConfig

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
}


def get_transformer_config(model_name_or_path: str, vocab_size: int) -> TransformerConfig:
    """Get the appropriate TransformerConfig for a given model name.

    Args:
        model_name_or_path: HuggingFace model name or path.
        vocab_size: Vocabulary size for the model.

    Returns:
        TransformerConfig for the specified model.

    Raises:
        ValueError: If model not in OLMO_MODEL_CONFIG_MAP.
    """
    config_name = OLMO_MODEL_CONFIG_MAP.get(model_name_or_path)
    if config_name is None:
        available_models = ", ".join(OLMO_MODEL_CONFIG_MAP.keys())
        available_configs = [
            name for name in dir(TransformerConfig) if name.startswith("olmo") and not name.startswith("_")
        ]
        raise ValueError(
            f"Model '{model_name_or_path}' not found in OLMO_MODEL_CONFIG_MAP. "
            f"Available models: {available_models}. "
            f"Available config names: {', '.join(available_configs)}"
        )
    return getattr(TransformerConfig, config_name)(vocab_size=vocab_size)
