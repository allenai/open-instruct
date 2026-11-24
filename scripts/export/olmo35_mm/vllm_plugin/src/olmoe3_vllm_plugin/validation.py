# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/validation.py
"""Checkpoint-config checks for the vLLM executor, importable without vLLM."""


def validate_hf_config(config) -> None:
    """Reject HF configs the OLMoE3 vLLM executor would serve incorrectly."""
    if config.latent_moe_dim is not None and config.latent_moe_bias:
        raise NotImplementedError("The OLMoE3 vLLM plugin supports bias-free latent MoE")
    # Attention projections are built bias-free and load_weights skips unmatched biases,
    # so a biased checkpoint would otherwise load with its biases silently dropped.
    if config.attention_bias:
        raise NotImplementedError("The OLMoE3 vLLM plugin supports bias-free attention only")
    # Non-KDA layers are built as full attention, so sliding layers would ignore their window.
    if "sliding_attention" in config.layer_types:
        raise NotImplementedError("The OLMoE3 vLLM plugin does not support sliding attention")
    # Global load balancing is also used by non-EMO training. A missing EMO
    # pool means unrestricted routing, not a restricted evaluation pool.
    if config.emo_eval_document_expert_pool not in (None, config.n_routed_experts):
        raise ValueError("vLLM export requires the EMo evaluation pool to contain all experts")
    if config.linear_num_key_heads != config.linear_num_value_heads:
        raise ValueError("The pinned KDA kernel requires equal key and value head counts")
    for name in (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
    ):
        value = getattr(config, name, None)
        if value is None or value <= 0:
            raise ValueError(f"OLMoE3 KDA requires a positive {name}, got {value!r}")
    if config.use_rope:
        raise NotImplementedError("The OLMoE3 vLLM plugin currently supports NoPE only")
    if config.attention_gate_type not in (None, "elementwise"):
        raise NotImplementedError("Only elementwise full-attention gates are supported")
    if not config.embed_norm or not config.use_peri_ln:
        raise ValueError("The OLMoE3 vLLM executor requires the ladder's embed/peri norms")
