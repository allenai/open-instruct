from open_instruct.grpo_name_mapping import olmo_core_to_hf_name


def test_olmo_core_to_hf_name_unwraps_activation_checkpointed_modules() -> None:
    assert olmo_core_to_hf_name("blocks.7.attention._checkpoint_wrapped_module.w_q.weight") == (
        "model.layers.7.self_attn.q_proj.weight"
    )


def test_olmo_core_to_hf_name_unwraps_checkpointed_norm_modules() -> None:
    assert olmo_core_to_hf_name("blocks.7.attention_norm._checkpoint_wrapped_module.weight") == (
        "model.layers.7.input_layernorm.weight"
    )


def test_olmo_core_to_hf_name_maps_checkpointed_gdn_parameters() -> None:
    gdn_layers = frozenset({7})
    assert olmo_core_to_hf_name(
        "blocks.7.attention._checkpoint_wrapped_module.w_q.weight", gdn_layer_indices=gdn_layers
    ) == "model.layers.7.linear_attn.q_proj.weight"
    assert olmo_core_to_hf_name(
        "blocks.7.attention._checkpoint_wrapped_module.A_log", gdn_layer_indices=gdn_layers
    ) == "model.layers.7.linear_attn.A_log"
    assert olmo_core_to_hf_name(
        "blocks.7.attention._checkpoint_wrapped_module.in_proj_qkvg.weight", gdn_layer_indices=gdn_layers
    ) == "model.layers.7.linear_attn.in_proj_qkvg.weight"
