from open_instruct.olmo_core_utils import make_hybrid_small_name_mapper


def test_make_hybrid_small_name_mapper_detects_gdn_layers() -> None:
    state_dict_keys = ["blocks.0.attention.A_log", "blocks.1.attention.w_q.weight"]
    mapper = make_hybrid_small_name_mapper(state_dict_keys)
    assert mapper("blocks.0.attention.w_q.weight") == "model.layers.0.linear_attn.q_proj.weight"
    assert mapper("blocks.1.attention.w_q.weight") == "model.layers.1.self_attn.q_proj.weight"


def test_make_hybrid_small_name_mapper_unwraps_checkpointed_gdn_layers() -> None:
    state_dict_keys = ["blocks.0.attention._checkpoint_wrapped_module.A_log"]
    mapper = make_hybrid_small_name_mapper(state_dict_keys)
    assert (
        mapper("blocks.0.attention._checkpoint_wrapped_module.w_q.weight")
        == "model.layers.0.linear_attn.q_proj.weight"
    )


def test_make_hybrid_small_name_mapper_unwraps_checkpointed_full_attention_layers() -> None:
    state_dict_keys: list[str] = []
    mapper = make_hybrid_small_name_mapper(state_dict_keys)
    assert (
        mapper("blocks.0.attention._checkpoint_wrapped_module.w_q.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
