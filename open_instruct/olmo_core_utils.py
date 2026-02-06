"""
OLMo-core utility functions and configuration mappings.

This module provides common utilities for working with OLMo-core models,
including model configuration mappings and helper functions.
"""

import pathlib
from typing import Any

import safetensors.torch
import torch.distributed as dist
import transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.fla.layer import FLAConfig
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from torch.distributed.tensor import DTensor, distribute_tensor

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


def _convert_hf_key_to_olmo_core(hf_key: str) -> str:
    """Convert a single HF state dict key to the OLMo-core equivalent.

    Handles both standard attention layers and FLA (linear attention) layers in
    hybrid models. Key mappings derived from olmo-core's
    convert_checkpoint_to_hf_hybrid.py (reversed).
    """
    # Global keys
    if hf_key == "model.embed_tokens.weight":
        return "embeddings.weight"
    if hf_key == "model.norm.weight":
        return "lm_head.norm.weight"
    if hf_key == "lm_head.weight":
        return "lm_head.w_out.weight"

    # FLA layer keys: model.layers.{i}.linear_attn.* -> blocks.{i}.fla.inner.*
    if ".linear_attn." in hf_key:
        return hf_key.replace("model.layers.", "blocks.").replace(".linear_attn.", ".fla.inner.")

    # FLA norm: model.layers.{i}.attention_layer_norm.* -> blocks.{i}.fla_norm.*
    if ".attention_layer_norm." in hf_key:
        return hf_key.replace("model.layers.", "blocks.").replace(".attention_layer_norm.", ".fla_norm.")

    # FLA feedforward norm: model.layers.{i}.feedforward_layer_norm.* -> blocks.{i}.feed_forward_norm.*
    if ".feedforward_layer_norm." in hf_key:
        return hf_key.replace("model.layers.", "blocks.").replace(".feedforward_layer_norm.", ".feed_forward_norm.")

    # Attention keys: model.layers.{i}.self_attn.{q,k,v,o}_proj -> blocks.{i}.attention.w_{q,k,v,out}
    attn_proj_map = {
        ".self_attn.q_proj.": ".attention.w_q.",
        ".self_attn.k_proj.": ".attention.w_k.",
        ".self_attn.v_proj.": ".attention.w_v.",
        ".self_attn.o_proj.": ".attention.w_out.",
    }
    for hf_pat, olmo_pat in attn_proj_map.items():
        if hf_pat in hf_key:
            return hf_key.replace("model.layers.", "blocks.").replace(hf_pat, olmo_pat)

    # Attention norms: self_attn.q_norm, self_attn.k_norm
    if ".self_attn.q_norm." in hf_key:
        return hf_key.replace("model.layers.", "blocks.").replace(".self_attn.q_norm.", ".attention.q_norm.")
    if ".self_attn.k_norm." in hf_key:
        return hf_key.replace("model.layers.", "blocks.").replace(".self_attn.k_norm.", ".attention.k_norm.")

    # Layer norms for attention blocks
    if ".post_attention_layernorm." in hf_key:
        return hf_key.replace("model.layers.", "blocks.").replace(".post_attention_layernorm.", ".attention_norm.")
    if ".post_feedforward_layernorm." in hf_key:
        return hf_key.replace("model.layers.", "blocks.").replace(
            ".post_feedforward_layernorm.", ".feed_forward_norm."
        )

    # MLP keys
    mlp_map = {
        ".mlp.gate_proj.": ".feed_forward.w1.",
        ".mlp.down_proj.": ".feed_forward.w2.",
        ".mlp.up_proj.": ".feed_forward.w3.",
    }
    for hf_pat, olmo_pat in mlp_map.items():
        if hf_pat in hf_key:
            return hf_key.replace("model.layers.", "blocks.").replace(hf_pat, olmo_pat)

    raise ValueError(f"Unknown HF key: {hf_key}")


def load_hf_model_weights(
    model_name_or_path: str, model_state_dict: dict[str, Any], process_group: dist.ProcessGroup | None = None
):
    """Load HF weights into an OLMo-core model state dict.

    Loads directly from safetensors and remaps keys using mappings derived from
    olmo-core's convert_checkpoint_to_hf_hybrid.py. This supports both standard
    and hybrid (FLA) checkpoints, unlike olmo-core's load_hf_model which has
    neither FLA key mappings nor support for single-file safetensors checkpoints.
    """
    model_path = pathlib.Path(model_name_or_path)
    safetensors_file = model_path / "model.safetensors"
    if not safetensors_file.exists():
        raise FileNotFoundError(f"Expected safetensors file at {safetensors_file}")

    logger.info(f"Loading safetensors from {safetensors_file}")
    hf_state_dict = safetensors.torch.load_file(str(safetensors_file))

    for hf_key, tensor in hf_state_dict.items():
        olmo_key = _convert_hf_key_to_olmo_core(hf_key)
        olmo_core_state = model_state_dict[olmo_key]
        if isinstance(olmo_core_state, DTensor):
            olmo_core_state = distribute_tensor(tensor, olmo_core_state.device_mesh, olmo_core_state.placements)
        else:
            olmo_core_state = tensor
        model_state_dict[olmo_key] = olmo_core_state


def save_state_dict_as_hf(model_config, state_dict, save_dir, original_model_name_or_path, tokenizer):
    unwrapped_model = model_config.build(init_device="cpu")
    unwrapped_model.load_state_dict(state_dict)
    save_hf_model(save_dir=save_dir, model_state_dict=state_dict, model=unwrapped_model, save_overwrite=True)
    tokenizer.save_pretrained(save_dir)
    original_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path)
    original_config.save_pretrained(save_dir)
