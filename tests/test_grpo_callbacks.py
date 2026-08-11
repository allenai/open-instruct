from open_instruct.grpo_name_mapping import olmo_core_to_hf_name


def test_olmo_core_to_hf_name_unwraps_activation_checkpointed_modules() -> None:
    assert olmo_core_to_hf_name("blocks.7.attention._checkpoint_wrapped_module.w_q.weight") == (
        "model.layers.7.self_attn.q_proj.weight"
    )


def test_olmo_core_to_hf_name_unwraps_checkpointed_norm_modules() -> None:
    assert olmo_core_to_hf_name("blocks.7.attention_norm._checkpoint_wrapped_module.weight") == (
        "model.layers.7.input_layernorm.weight"
    )
