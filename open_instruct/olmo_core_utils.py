"""
OLMo-core utility functions and configuration mappings.

This module provides common utilities for working with OLMo-core models,
including model configuration mappings and helper functions.
"""

import json
import pathlib

import transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.conversion.state_converter import StateConverter
from olmo_core.nn.conversion.state_mapping import StateMappingTemplate, StateType, TemplatePlaceholder
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.convert import (
    OLMO_CORE_TO_HF_MODULE_MAPPINGS,
    OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS,
    OLMO_CORE_TO_HF_WEIGHT_MAPPINGS,
)
from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig
from olmo_core.nn.transformer.block import FLABlock, FLAConfig, ReorderedNormTransformerBlock
from safetensors.torch import save_file

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

LAYER = TemplatePlaceholder.LAYER

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

FLA_OLMO_CORE_TO_HF_WEIGHT_MAPPINGS: dict[str, str] = {
    f"blocks.{LAYER}.fla.inner.q_proj.weight": f"model.layers.{LAYER}.linear_attn.q_proj.weight",
    f"blocks.{LAYER}.fla.inner.k_proj.weight": f"model.layers.{LAYER}.linear_attn.k_proj.weight",
    f"blocks.{LAYER}.fla.inner.v_proj.weight": f"model.layers.{LAYER}.linear_attn.v_proj.weight",
    f"blocks.{LAYER}.fla.inner.g_proj.weight": f"model.layers.{LAYER}.linear_attn.g_proj.weight",
    f"blocks.{LAYER}.fla.inner.a_proj.weight": f"model.layers.{LAYER}.linear_attn.a_proj.weight",
    f"blocks.{LAYER}.fla.inner.b_proj.weight": f"model.layers.{LAYER}.linear_attn.b_proj.weight",
    f"blocks.{LAYER}.fla.inner.o_proj.weight": f"model.layers.{LAYER}.linear_attn.o_proj.weight",
    f"blocks.{LAYER}.fla.inner.q_conv1d.weight": f"model.layers.{LAYER}.linear_attn.q_conv1d.weight",
    f"blocks.{LAYER}.fla.inner.k_conv1d.weight": f"model.layers.{LAYER}.linear_attn.k_conv1d.weight",
    f"blocks.{LAYER}.fla.inner.v_conv1d.weight": f"model.layers.{LAYER}.linear_attn.v_conv1d.weight",
    f"blocks.{LAYER}.fla.inner.o_norm.weight": f"model.layers.{LAYER}.linear_attn.o_norm.weight",
    f"blocks.{LAYER}.fla.inner.A_log": f"model.layers.{LAYER}.linear_attn.A_log",
    f"blocks.{LAYER}.fla.inner.dt_bias": f"model.layers.{LAYER}.linear_attn.dt_bias",
    f"blocks.{LAYER}.fla_norm.weight": f"model.layers.{LAYER}.attention_layer_norm.weight",
}

FLA_OLMO_CORE_TO_HF_MODULE_MAPPINGS: dict[str, str] = {
    f"blocks.{LAYER}.fla": f"model.layers.{LAYER}.linear_attn",
    f"blocks.{LAYER}.fla.inner": f"model.layers.{LAYER}.linear_attn",
    f"blocks.{LAYER}.fla_norm": f"model.layers.{LAYER}.post_attention_layernorm",
}


def _build_olmo3_7B_hybrid(vocab_size: int, **kwargs) -> TransformerConfig:
    remove_heads = 2
    config = TransformerConfig.olmo3_7B(vocab_size=vocab_size, **kwargs)
    config.d_model -= remove_heads * 128
    config.block.attention.n_heads -= remove_heads
    config.block.name = TransformerBlockType.fla_hybrid
    config.block.fla = FLAConfig(
        name="GatedDeltaNet",
        dtype=config.dtype,
        fla_layer_kwargs={
            "head_dim": int(0.75 * config.d_model / config.block.attention.n_heads),
            "use_gate": True,
            "allow_neg_eigval": True,
        },
    )
    config.block.fla_hybrid_attention_indices = [i for i in range(config.n_layers) if i % 4 == 3]
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

    kwargs: dict = {"vocab_size": vocab_size}
    if attn_backend is not None:
        kwargs["attn_backend"] = AttentionBackendName(attn_backend)

    if config_name == "olmo3_7B_hybrid":
        return _build_olmo3_7B_hybrid(**kwargs)

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
    return getattr(TransformerConfig, config_name)(**kwargs)


def _is_hybrid_model(model) -> bool:
    blocks = list(model.blocks.values())
    return any(isinstance(block, FLABlock) for block in blocks)


def _get_rope_scaling_for_hybrid(blocks) -> dict | None:
    for block in blocks:
        if isinstance(block, ReorderedNormTransformerBlock):
            rope = block.attention.rope
            if rope is None:
                return None
            if hasattr(rope, "scaling_config") and rope.scaling_config is not None:
                return rope.scaling_config
            return None
    return None


def _get_hybrid_hf_config_dict(model, transformer_config: TransformerConfig) -> dict:
    blocks = list(model.blocks.values())
    n_layers = len(blocks)

    layer_types = []
    attention_block = None
    fla_block = None

    for idx, block in enumerate(blocks):
        if isinstance(block, FLABlock):
            layer_types.append("linear_attention")
            if fla_block is None:
                fla_block = block
        elif isinstance(block, ReorderedNormTransformerBlock):
            layer_types.append("full_attention")
            if attention_block is None:
                attention_block = block
        else:
            raise ValueError(f"Unknown block type at layer {idx}: {type(block)}")

    if attention_block is None:
        raise ValueError("Hybrid model must have at least one attention layer")
    if fla_block is None:
        raise ValueError("Hybrid model must have at least one FLA layer")

    attn = attention_block.attention

    rope_theta = None
    rope_scaling = None
    use_rope = attn.rope is not None
    if use_rope:
        rope_theta = float(attn.rope.theta)
        rope_scaling = _get_rope_scaling_for_hybrid(blocks)

    block_config = transformer_config.block
    fla_config = block_config.fla
    fla_kwargs = fla_config.fla_layer_kwargs if fla_config else {}

    fla_inner = fla_block.fla.inner
    linear_num_heads = getattr(fla_inner, "num_heads", attn.n_heads)
    linear_key_head_dim = fla_kwargs.get("head_dim", getattr(fla_inner, "head_k_dim", 96))
    linear_value_head_dim = getattr(fla_inner, "head_v_dim", linear_key_head_dim * 2)

    config_dict = {
        "model_type": "olmo3_2_hybrid",
        "architectures": ["Olmo3_2HybridForCausalLM"],
        "vocab_size": model.vocab_size,
        "hidden_size": model.d_model,
        "intermediate_size": attention_block.feed_forward.hidden_size,
        "num_hidden_layers": n_layers,
        "num_attention_heads": attn.n_heads,
        "num_key_value_heads": attn.n_kv_heads or attn.n_heads,
        "hidden_act": "silu",
        "max_position_embeddings": 65536,
        "initializer_range": 0.02,
        "use_cache": True,
        "attention_bias": attn.w_out.bias is not None,
        "attention_dropout": 0.0,
        "rms_norm_eps": attention_block.feed_forward_norm.eps,
        "tie_word_embeddings": False,
        "layer_types": layer_types,
        "linear_num_key_heads": linear_num_heads,
        "linear_num_value_heads": linear_num_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_conv_kernel_dim": 4,
        "linear_use_gate": fla_kwargs.get("use_gate", True),
        "linear_allow_neg_eigval": fla_kwargs.get("allow_neg_eigval", True),
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "transformers_version": "4.52.0",
    }

    if use_rope:
        config_dict["rope_theta"] = rope_theta
        if rope_scaling:
            config_dict["rope_scaling"] = rope_scaling

    return config_dict


def _get_hybrid_converter_to_hf() -> StateConverter:
    mapping_templates = {
        olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.module)
        for olmo_core_key, hf_key in OLMO_CORE_TO_HF_MODULE_MAPPINGS.items()
    }
    mapping_templates.update(
        {
            olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.weight)
            for olmo_core_key, hf_key in OLMO_CORE_TO_HF_WEIGHT_MAPPINGS.items()
        }
    )
    mapping_templates.update(OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS)

    mapping_templates.update(
        {
            olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.weight)
            for olmo_core_key, hf_key in FLA_OLMO_CORE_TO_HF_WEIGHT_MAPPINGS.items()
        }
    )
    mapping_templates.update(
        {
            olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.module)
            for olmo_core_key, hf_key in FLA_OLMO_CORE_TO_HF_MODULE_MAPPINGS.items()
        }
    )

    return StateConverter(list(mapping_templates.values()))


def _save_hybrid_hf_model(save_dir, model_state_dict, model, transformer_config):
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    hf_config_dict = _get_hybrid_hf_config_dict(model, transformer_config)

    n_layers = len(list(model.blocks.values()))
    converter = _get_hybrid_converter_to_hf()
    placeholder_bounds = {TemplatePlaceholder.LAYER: n_layers}
    hf_state_dict = converter.convert(model_state_dict, placeholder_bounds)

    blocks = list(model.blocks.values())
    for i, block in enumerate(blocks):
        if isinstance(block, FLABlock):
            old_key = f"model.layers.{i}.post_feedforward_layernorm.weight"
            new_key = f"model.layers.{i}.feedforward_layer_norm.weight"
            if old_key in hf_state_dict:
                hf_state_dict[new_key] = hf_state_dict.pop(old_key)

    config_path = save_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(hf_config_dict, f, indent=2)
    logger.info(f"Saved hybrid HF config to {config_path}")

    save_file(hf_state_dict, save_path / "model.safetensors")
    logger.info(f"Saved hybrid HF weights to {save_path / 'model.safetensors'}")


def save_state_dict_as_hf(model_config, state_dict, save_dir, original_model_name_or_path, tokenizer):
    unwrapped_model = model_config.build(init_device="cpu")
    unwrapped_model.load_state_dict(state_dict)
    if _is_hybrid_model(unwrapped_model):
        _save_hybrid_hf_model(save_dir, state_dict, unwrapped_model, model_config)
    else:
        save_hf_model(save_dir=save_dir, model_state_dict=state_dict, model=unwrapped_model, save_overwrite=True)
    tokenizer.save_pretrained(save_dir)
    if original_model_name_or_path and not _is_native_olmo_checkpoint(original_model_name_or_path):
        original_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path)
        original_config.save_pretrained(save_dir)


def _is_native_olmo_checkpoint(path: str) -> bool:
    return path.startswith("/weka/") and not path.endswith("-hf")
