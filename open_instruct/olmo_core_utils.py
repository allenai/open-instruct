"""
OLMo-core utility functions and configuration mappings.

This module provides common utilities for working with OLMo-core models,
including model configuration mappings and helper functions.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.checkpoint import load_hf_model, save_hf_model
from olmo_core.nn.transformer import TransformerConfig
from safetensors.torch import save_file
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
}


def _override_attn_backend(config_dict: Any, attn_backend: str) -> None:
    """Recursively override any `AttentionConfig.backend` field in-place.

    Maintainer-pipeline checkpoints (e.g. OLMo-hybrid small suite) bake in the backend
    (e.g. `flash_3`, H100-only) used at training time. Force it to match the requested
    `attn_backend` so the config also builds on other hardware (e.g. `flash_2`).
    """
    if isinstance(config_dict, dict):
        if config_dict.get("_CLASS_", "").endswith("AttentionConfig") and "backend" in config_dict:
            config_dict["backend"] = attn_backend
        for value in config_dict.values():
            _override_attn_backend(value, attn_backend)
    elif isinstance(config_dict, list):
        for item in config_dict:
            _override_attn_backend(item, attn_backend)


def _load_native_transformer_config(model_name_or_path: str, attn_backend: str | None) -> TransformerConfig | None:
    """Load a `TransformerConfig` from a maintainer-pipeline checkpoint's own config.

    Maintainer pipelines (e.g. OLMo-hybrid small suite) write a full HF-format checkpoint
    (suffixed `-hf`) next to the olmo-core-native checkpoint it was converted from. The
    native checkpoint's `config.json` embeds the exact `TransformerConfig` (as a `_CLASS_`-
    tagged dict) used to train it, so we can deserialize it directly instead of requiring a
    hardcoded named preset (there is no `olmo_hybrid_small` preset in `TransformerConfig`).

    Returns None if `model_name_or_path` doesn't look like such a checkpoint.
    """
    path = Path(model_name_or_path)
    if not path.is_dir() or not path.name.endswith("-hf"):
        return None
    native_dir = path.with_name(path.name[: -len("-hf")])
    config_path = native_dir / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path) as f:
        data = json.load(f)
    model_dict = data.get("model")
    if model_dict is None:
        return None
    if attn_backend is not None:
        _override_attn_backend(model_dict, attn_backend)
    logger.info(f"Loading native TransformerConfig from '{config_path}'")
    return TransformerConfig.from_dict(model_dict)


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
        native_config = _load_native_transformer_config(model_name_or_config, attn_backend)
        if native_config is not None:
            native_config.vocab_size = vocab_size
            return native_config

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


def ensure_hf_checkpoint_sentinel_file(model_name_or_path: str, work_dir: str) -> str:
    """Ensure an HF checkpoint directory has one of the sentinel files `load_hf_model` requires.

    `olmo_core.nn.hf.checkpoint.load_hf_model()` asserts that the checkpoint directory contains
    `generation_config.json`, `model.safetensors.index.json`, or `pytorch_model.bin` as a sanity
    check. Some maintainer-pipeline checkpoints (e.g. OLMo-hybrid small suite) only write a
    single-file `model.safetensors` with none of these markers. Rather than mutating the
    (possibly shared, read-only) checkpoint directory, build a local mirror of symlinks plus a
    synthesized `generation_config.json` and return its path.
    """
    path = Path(model_name_or_path)
    sentinels = ("generation_config.json", "model.safetensors.index.json", "pytorch_model.bin")
    if not path.is_dir() or any((path / name).is_file() for name in sentinels):
        return model_name_or_path

    mirror_dir = Path(work_dir) / "hf_checkpoint_mirror" / path.name
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        link = mirror_dir / item.name
        if not link.exists():
            link.symlink_to(item.resolve())
    hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    transformers.GenerationConfig.from_model_config(hf_config).save_pretrained(mirror_dir)
    logger.info(f"Checkpoint '{model_name_or_path}' missing HF sentinel file; using local mirror '{mirror_dir}'")
    return str(mirror_dir)


def _get_hybrid_small_layer_types(hf_state_dict: dict[str, torch.Tensor], n_layers: int) -> list[str]:
    """Determine per-layer types by checking which layers have GDN-specific keys (`A_log`)."""
    return [
        "linear_attention" if f"model.layers.{i}.linear_attn.A_log" in hf_state_dict else "full_attention"
        for i in range(n_layers)
    ]


def _convert_hybrid_small_layer_from_hf(
    hf_state_dict: dict[str, torch.Tensor], layer_i: int, layer_type: str
) -> dict[str, torch.Tensor]:
    """Convert one transformer block's weights from HF `olmo_hybrid_small` naming to olmo-core.

    Inverse of `convert_gdn_layer_weights`/`convert_attention_layer_weights` in the transformers
    fork's `convert_olmo_hybrid_small_weights_to_hf.py` (the maintainer's own olmo-core -> HF
    mapping for this architecture; there is no HF -> olmo-core direction in the olmo-core fork's
    `load_hf_model`, which only knows the generic non-hybrid key mappings).
    """
    hf_prefix = f"model.layers.{layer_i}"
    prefix = f"blocks.{layer_i}"
    state: dict[str, torch.Tensor] = {
        f"{prefix}.feed_forward.w1.weight": hf_state_dict[f"{hf_prefix}.mlp.gate_proj.weight"],
        f"{prefix}.feed_forward.w2.weight": hf_state_dict[f"{hf_prefix}.mlp.down_proj.weight"],
        f"{prefix}.feed_forward.w3.weight": hf_state_dict[f"{hf_prefix}.mlp.up_proj.weight"],
        f"{prefix}.attention_norm.weight": hf_state_dict[f"{hf_prefix}.input_layernorm.weight"],
        f"{prefix}.post_attention_norm.weight": hf_state_dict[f"{hf_prefix}.post_attention_layernorm.weight"],
        f"{prefix}.feed_forward_norm.weight": hf_state_dict[f"{hf_prefix}.ffn_layernorm.weight"],
        f"{prefix}.post_feed_forward_norm.weight": hf_state_dict[f"{hf_prefix}.post_feedforward_layernorm.weight"],
    }
    if layer_type == "linear_attention":
        state.update(
            {
                f"{prefix}.attention.w_q.weight": hf_state_dict[f"{hf_prefix}.linear_attn.q_proj.weight"],
                f"{prefix}.attention.w_k.weight": hf_state_dict[f"{hf_prefix}.linear_attn.k_proj.weight"],
                f"{prefix}.attention.w_v.weight": hf_state_dict[f"{hf_prefix}.linear_attn.v_proj.weight"],
                f"{prefix}.attention.w_out.weight": hf_state_dict[f"{hf_prefix}.linear_attn.o_proj.weight"],
                f"{prefix}.attention.w_g.weight": hf_state_dict[f"{hf_prefix}.linear_attn.g_proj.weight"],
                f"{prefix}.attention.w_a.weight": hf_state_dict[f"{hf_prefix}.linear_attn.a_proj.weight"],
                f"{prefix}.attention.w_b.weight": hf_state_dict[f"{hf_prefix}.linear_attn.b_proj.weight"],
                f"{prefix}.attention.o_norm.weight": hf_state_dict[f"{hf_prefix}.linear_attn.o_norm.weight"],
                f"{prefix}.attention.q_conv1d.weight": hf_state_dict[f"{hf_prefix}.linear_attn.q_conv1d.weight"],
                f"{prefix}.attention.k_conv1d.weight": hf_state_dict[f"{hf_prefix}.linear_attn.k_conv1d.weight"],
                f"{prefix}.attention.v_conv1d.weight": hf_state_dict[f"{hf_prefix}.linear_attn.v_conv1d.weight"],
                f"{prefix}.attention.A_log": hf_state_dict[f"{hf_prefix}.linear_attn.A_log"],
                f"{prefix}.attention.dt_bias": hf_state_dict[f"{hf_prefix}.linear_attn.dt_bias"],
            }
        )
    else:
        state.update(
            {
                f"{prefix}.attention.w_q.weight": hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"],
                f"{prefix}.attention.w_k.weight": hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"],
                f"{prefix}.attention.w_v.weight": hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"],
                f"{prefix}.attention.w_out.weight": hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"],
                f"{prefix}.attention.w_g.weight": hf_state_dict[f"{hf_prefix}.self_attn.attn_gate.weight"],
                f"{prefix}.attention.q_norm.weight": hf_state_dict[f"{hf_prefix}.self_attn.q_norm.weight"],
                f"{prefix}.attention.k_norm.weight": hf_state_dict[f"{hf_prefix}.self_attn.k_norm.weight"],
            }
        )
    return state


def _load_hf_hybrid_small_model(
    model_name_or_path: str, model_state_dict: dict[str, Any], hf_config: transformers.PretrainedConfig
) -> None:
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    hf_state_dict = hf_model.state_dict()
    n_layers = hf_config.num_hidden_layers
    layer_types = _get_hybrid_small_layer_types(hf_state_dict, n_layers)
    logger.info(f"OLMo-hybrid small: detected layer types {layer_types}")

    converted_state_dict: dict[str, torch.Tensor] = {
        "embeddings.weight": hf_state_dict["model.embed_tokens.weight"],
        "embedding_norm.weight": hf_state_dict["model.embed_norm.weight"],
        "lm_head.norm.weight": hf_state_dict["model.norm.weight"],
        "lm_head.w_out.weight": hf_state_dict["lm_head.weight"],
    }
    for layer_i, layer_type in enumerate(layer_types):
        converted_state_dict.update(_convert_hybrid_small_layer_from_hf(hf_state_dict, layer_i, layer_type))

    for key in sorted(converted_state_dict.keys()):
        state = converted_state_dict[key]
        olmo_core_state = model_state_dict[key]
        if isinstance(olmo_core_state, DTensor):
            olmo_core_state = distribute_tensor(state, olmo_core_state.device_mesh, olmo_core_state.placements)
        else:
            olmo_core_state = state
        model_state_dict[key] = olmo_core_state


def load_hf_model_with_hybrid_support(
    model_name_or_path: str, model_state_dict: dict[str, Any], work_dir: str
) -> None:
    """Load HF checkpoint weights into an olmo-core state dict, with OLMo-hybrid small support.

    The olmo-core fork's `load_hf_model` only implements the generic (non-hybrid) HF -> olmo-core
    weight conversion, which doesn't know about `olmo_hybrid_small`'s GatedDeltaNet linear-attention
    layers. Dispatch to our own conversion for that architecture; delegate to olmo-core otherwise.
    """
    hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    if getattr(hf_config, "model_type", None) == "olmo_hybrid_small":
        _load_hf_hybrid_small_model(model_name_or_path, model_state_dict, hf_config)
    else:
        load_hf_model(model_name_or_path, model_state_dict, work_dir=work_dir)


def _convert_hybrid_small_layer_to_hf(
    state_dict: dict[str, torch.Tensor], layer_i: int, layer_type: str
) -> dict[str, torch.Tensor]:
    """Convert one transformer block's weights from olmo-core to HF `olmo_hybrid_small` naming.

    Inverse of `_convert_hybrid_small_layer_from_hf`; matches the maintainer's own
    `convert_gdn_layer_weights`/`convert_attention_layer_weights` in the transformers fork's
    `convert_olmo_hybrid_small_weights_to_hf.py`.
    """
    prefix = f"blocks.{layer_i}"
    hf_prefix = f"model.layers.{layer_i}"
    hf_state: dict[str, torch.Tensor] = {
        f"{hf_prefix}.mlp.gate_proj.weight": state_dict[f"{prefix}.feed_forward.w1.weight"],
        f"{hf_prefix}.mlp.down_proj.weight": state_dict[f"{prefix}.feed_forward.w2.weight"],
        f"{hf_prefix}.mlp.up_proj.weight": state_dict[f"{prefix}.feed_forward.w3.weight"],
        f"{hf_prefix}.input_layernorm.weight": state_dict[f"{prefix}.attention_norm.weight"],
        f"{hf_prefix}.post_attention_layernorm.weight": state_dict[f"{prefix}.post_attention_norm.weight"],
        f"{hf_prefix}.ffn_layernorm.weight": state_dict[f"{prefix}.feed_forward_norm.weight"],
        f"{hf_prefix}.post_feedforward_layernorm.weight": state_dict[f"{prefix}.post_feed_forward_norm.weight"],
    }
    if layer_type == "linear_attention":
        hf_state.update(
            {
                f"{hf_prefix}.linear_attn.q_proj.weight": state_dict[f"{prefix}.attention.w_q.weight"],
                f"{hf_prefix}.linear_attn.k_proj.weight": state_dict[f"{prefix}.attention.w_k.weight"],
                f"{hf_prefix}.linear_attn.v_proj.weight": state_dict[f"{prefix}.attention.w_v.weight"],
                f"{hf_prefix}.linear_attn.o_proj.weight": state_dict[f"{prefix}.attention.w_out.weight"],
                f"{hf_prefix}.linear_attn.g_proj.weight": state_dict[f"{prefix}.attention.w_g.weight"],
                f"{hf_prefix}.linear_attn.a_proj.weight": state_dict[f"{prefix}.attention.w_a.weight"],
                f"{hf_prefix}.linear_attn.b_proj.weight": state_dict[f"{prefix}.attention.w_b.weight"],
                f"{hf_prefix}.linear_attn.o_norm.weight": state_dict[f"{prefix}.attention.o_norm.weight"],
                f"{hf_prefix}.linear_attn.q_conv1d.weight": state_dict[f"{prefix}.attention.q_conv1d.weight"],
                f"{hf_prefix}.linear_attn.k_conv1d.weight": state_dict[f"{prefix}.attention.k_conv1d.weight"],
                f"{hf_prefix}.linear_attn.v_conv1d.weight": state_dict[f"{prefix}.attention.v_conv1d.weight"],
                f"{hf_prefix}.linear_attn.A_log": state_dict[f"{prefix}.attention.A_log"],
                f"{hf_prefix}.linear_attn.dt_bias": state_dict[f"{prefix}.attention.dt_bias"],
            }
        )
    else:
        hf_state.update(
            {
                f"{hf_prefix}.self_attn.q_proj.weight": state_dict[f"{prefix}.attention.w_q.weight"],
                f"{hf_prefix}.self_attn.k_proj.weight": state_dict[f"{prefix}.attention.w_k.weight"],
                f"{hf_prefix}.self_attn.v_proj.weight": state_dict[f"{prefix}.attention.w_v.weight"],
                f"{hf_prefix}.self_attn.o_proj.weight": state_dict[f"{prefix}.attention.w_out.weight"],
                f"{hf_prefix}.self_attn.attn_gate.weight": state_dict[f"{prefix}.attention.w_g.weight"],
                f"{hf_prefix}.self_attn.q_norm.weight": state_dict[f"{prefix}.attention.q_norm.weight"],
                f"{hf_prefix}.self_attn.k_norm.weight": state_dict[f"{prefix}.attention.k_norm.weight"],
            }
        )
    return hf_state


def _save_hf_hybrid_small_model(
    state_dict: dict[str, torch.Tensor], save_dir: str, hf_config: transformers.PretrainedConfig, tokenizer
) -> None:
    """Save an olmo-core `olmo_hybrid_small` state dict in HF format.

    `olmo_core.nn.hf.checkpoint.save_hf_model()` always calls the generic (non-hybrid)
    `get_hf_config`, which raises `NotImplementedError` for peri-norm blocks. And olmo-core's
    hybrid-aware config builder (`get_hybrid_hf_config`) targets a different HF architecture
    (`olmo3_5_hybrid`) than ours (`olmo_hybrid_small`). Reuse the original HF config verbatim
    (DPO doesn't change the architecture) and convert weights ourselves.
    """
    n_layers = hf_config.num_hidden_layers
    layer_types = [
        "linear_attention" if f"blocks.{i}.attention.A_log" in state_dict else "full_attention"
        for i in range(n_layers)
    ]
    hf_state_dict: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state_dict["embeddings.weight"],
        "model.embed_norm.weight": state_dict["embedding_norm.weight"],
        "model.norm.weight": state_dict["lm_head.norm.weight"],
        "lm_head.weight": state_dict["lm_head.w_out.weight"],
    }
    for layer_i, layer_type in enumerate(layer_types):
        hf_state_dict.update(_convert_hybrid_small_layer_to_hf(state_dict, layer_i, layer_type))
    hf_state_dict = {key: state.contiguous() for key, state in hf_state_dict.items()}

    os.makedirs(save_dir, exist_ok=True)
    save_file(hf_state_dict, os.path.join(save_dir, "model.safetensors"))
    hf_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved OLMo-hybrid small HF checkpoint to '{save_dir}'")


def save_state_dict_as_hf(model_config, state_dict, save_dir, original_model_name_or_path, tokenizer):
    original_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path)
    if getattr(original_config, "model_type", None) == "olmo_hybrid_small":
        _save_hf_hybrid_small_model(state_dict, save_dir, original_config, tokenizer)
        return
    unwrapped_model = model_config.build(init_device="cpu")
    unwrapped_model.load_state_dict(state_dict)
    save_hf_model(save_dir=save_dir, model_state_dict=state_dict, model=unwrapped_model, save_overwrite=True)
    tokenizer.save_pretrained(save_dir)
    original_config.save_pretrained(save_dir)
