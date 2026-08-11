"""Parameter-name conversion utilities for OLMo-core GRPO weight synchronization."""

import re

_BLOCK_PATTERN = re.compile(r"blocks\.(\d+)\.(.*)")
_OLMO_CORE_TO_HF_LAYER_MAPPINGS = {
    "attention.w_q.weight": "self_attn.q_proj.weight",
    "attention.w_k.weight": "self_attn.k_proj.weight",
    "attention.w_v.weight": "self_attn.v_proj.weight",
    "attention.w_out.weight": "self_attn.o_proj.weight",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward_norm.weight": "post_attention_layernorm.weight",
}


def olmo_core_to_hf_name(name: str) -> str:
    """Convert OLMo-core parameter name to HuggingFace format for Qwen3/LLaMA models."""
    # PyTorch's checkpoint wrapper stores the wrapped module below this name. Remove it so
    # activation checkpointing is transparent to the vLLM weight-sync name mapping.
    name = ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")

    if name == "embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "lm_head.norm.weight":
        return "model.norm.weight"
    if name == "lm_head.w_out.weight":
        return "lm_head.weight"

    layer_match = _BLOCK_PATTERN.match(name)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)
        if rest in _OLMO_CORE_TO_HF_LAYER_MAPPINGS:
            return f"model.layers.{layer_idx}.{_OLMO_CORE_TO_HF_LAYER_MAPPINGS[rest]}"

    return name
