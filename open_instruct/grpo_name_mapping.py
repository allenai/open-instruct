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
_OLMO_CORE_GDN_TO_HF_LAYER_MAPPINGS = {
    "attention.w_q.weight": "linear_attn.q_proj.weight",
    "attention.w_k.weight": "linear_attn.k_proj.weight",
    "attention.w_v.weight": "linear_attn.v_proj.weight",
    "attention.w_a.weight": "linear_attn.a_proj.weight",
    "attention.w_b.weight": "linear_attn.b_proj.weight",
    "attention.w_g.weight": "linear_attn.g_proj.weight",
    "attention.w_out.weight": "linear_attn.o_proj.weight",
    "attention.q_conv1d.weight": "linear_attn.q_conv1d.weight",
    "attention.k_conv1d.weight": "linear_attn.k_conv1d.weight",
    "attention.v_conv1d.weight": "linear_attn.v_conv1d.weight",
    "attention.o_norm.weight": "linear_attn.o_norm.weight",
    "attention.A_log": "linear_attn.A_log",
    "attention.dt_bias": "linear_attn.dt_bias",
    # vLLM's Olmo Hybrid implementation uses fused GDN parameters internally. Support
    # these names as well for OLMo-core configurations that expose the fused layout.
    "attention.in_proj_qkvg.weight": "linear_attn.in_proj_qkvg.weight",
    "attention.a_proj.weight": "linear_attn.a_proj.weight",
    "attention.b_proj.weight": "linear_attn.b_proj.weight",
    "attention.o_proj.weight": "linear_attn.o_proj.weight",
    "attention.conv1d.weight": "linear_attn.conv1d.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward_norm.weight": "post_attention_layernorm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
}


def olmo_core_to_hf_name(name: str, gdn_layer_indices: frozenset[int] = frozenset()) -> str:
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
        layer_mappings = (
            _OLMO_CORE_GDN_TO_HF_LAYER_MAPPINGS
            if int(layer_idx) in gdn_layer_indices
            else _OLMO_CORE_TO_HF_LAYER_MAPPINGS
        )
        if rest in layer_mappings:
            return f"model.layers.{layer_idx}.{layer_mappings[rest]}"

    return name
