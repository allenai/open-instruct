"""Exercise the real checkpoint reader and conversion gate on tiny hero shapes."""

import json

import pytest
import torch
from olmo_core.config import DType
from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM
from safetensors.torch import save_file
from scripts.miles.validate_hero_conversion import check_architecture, validate


@pytest.mark.parametrize("width,experts", [(32, 4), (48, 8)])
def test_flat_master_checkpoint_matches_hf_and_adapter_roundtrip(tmp_path, width, experts):
    hf = Olmo3MoeConfig(
        vocab_size=32,
        hidden_size=width,
        attention_hidden_size=width,
        head_dim=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        dense_mlp_intermediate_size=24,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        n_routed_experts=experts,
        num_experts_per_tok=2,
        latent_moe_dim=16,
        use_head_qk_norm=True,
        qk_norm_per_head_gains=True,
        scalable_softmax=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_allow_neg_eigval=True,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        embed_norm=True,
        use_peri_ln=True,
    )
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, dtype=DType.float32, attention_backend=AttentionBackendName.torch
    )
    native = config.build(init_device="cpu")
    reference = Olmo3MoeForCausalLM(hf)
    with torch.no_grad():
        for name, value in reference.named_parameters():
            if name.endswith((".q_norm.weight", ".k_norm.weight", ".ssmax_scale")):
                value.copy_(torch.linspace(0.6, 1.4, value.numel()).reshape(value.shape))
    olmo3.load_olmo3_moe_hf_state(native, hf, reference.state_dict())
    # Use the model-derived config exactly as the canonical exporter does.
    exported_config = get_hf_config(native)
    raw, exported = tmp_path / "native", tmp_path / "hf"
    raw.mkdir()
    exported.mkdir()
    (raw / "config.json").write_text(
        json.dumps(
            {"model": config.as_dict(include_class_name=True), "dataset": {"tokenizer": {"vocab_size": hf.vocab_size}}}
        )
    )
    (exported / "config.json").write_text(exported_config.to_json_string())
    state = {f"{name}.main": value.detach().flatten().clone() for name, value in native.named_parameters()}
    state["unused_optimizer_moment"] = torch.full((11,), float("nan"))
    save_state_dict(raw / "model_and_optim", state)
    save_file(
        {name: value.bfloat16().contiguous() for name, value in reference.state_dict().items()},
        exported / "model.safetensors",
    )
    report = validate(raw, exported)
    assert report["valid"]
    assert report["native_to_hf"]["exact_match_after_export_cast"]
    assert report["hf_core_hf"]["exact_match_after_export_cast"]
    assert report["hf_core_hf"]["attention_parameter_shapes"]["model.layers.1.self_attn.ssmax_scale"] == [4]


def test_rejects_same_shape_different_attention_semantics():
    original = Olmo3MoeConfig(scalable_softmax=True, use_head_qk_norm=True, qk_norm_per_head_gains=True)
    wrong = Olmo3MoeConfig(scalable_softmax=False, use_head_qk_norm=True, qk_norm_per_head_gains=True)
    with pytest.raises(ValueError, match="scalable_softmax"):
        check_architecture(original, wrong)
