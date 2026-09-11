"""Norm direction, count normalization, and refusal of ambiguous state layouts."""

import math

import pytest
import torch
from olmo_core.nn import attention
from olmo_core.nn.hf import convert
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM
from scripts.miles.checkpoint_drift import DriftAccumulator, compare, parameter_group
from scripts.miles.compare_native_small_drift import read_core_subset, selected
from scripts.miles.core_checkpoint_stream import CoreCheckpointState
from torch.distributed import checkpoint as dcp


def test_both_relative_denominators_and_chunking():
    initial = torch.tensor([3.0, 4.0])
    left, right = initial + torch.tensor([1.0, 0.0]), initial + torch.tensor([0.0, 2.0])
    a = DriftAccumulator()
    a.add(initial, left, right, chunk_elements=1)
    r = a.report()
    assert r["reference_l2"] == 5
    assert r["core"]["relative_to_reference_l2"] == 0.2
    assert r["core"]["change_rms"] == pytest.approx(1 / math.sqrt(2))
    assert r["between_backends"]["relative_to_core_change_l2"] == pytest.approx(math.sqrt(5))
    assert r["between_backends"]["relative_to_megatron_change_l2"] == pytest.approx(math.sqrt(5) / 2)
    assert r["between_backends"]["change_direction_cosine"] == 0


def test_zero_reference_or_change_is_explicitly_undefined():
    a = DriftAccumulator()
    a.add(torch.zeros(2), torch.zeros(2), torch.ones(2))
    r = a.report()
    assert r["core"]["relative_to_reference_l2"] is None
    assert r["between_backends"]["change_direction_cosine"] is None
    assert r["between_backends"]["relative_to_core_change_l2"] is None


def test_master_changes_not_erased_by_storage_rounding():
    initial = torch.ones(4)
    trained = initial + 1e-4
    a, b = DriftAccumulator(), DriftAccumulator()
    a.add(initial, trained, trained)
    b.add(initial.bfloat16(), trained.bfloat16(), trained.bfloat16())
    assert a.report()["core"]["changed_fraction"] == 1
    assert b.report()["core"]["changed_fraction"] == 0


def test_global_l2_aggregates_squared_norms_not_per_tensor_ratios():
    initial = {"model.norm.weight": torch.ones(1), "lm_head.weight": torch.ones(3)}
    trained = {key: value + 1 for key, value in initial.items()}
    r = compare(initial, trained, trained)["aggregates"]["all"]
    assert r["parameters"] == 4
    assert r["core"]["change_l2"] == 2
    assert r["core"]["change_rms"] == 1
    assert r["between_backends"]["change_direction_cosine"] == 1


def test_rejects_shape_inventory_and_nonfinite_corruption():
    a = DriftAccumulator()
    with pytest.raises(ValueError, match="shapes"):
        a.add(torch.ones(2), torch.ones(1, 2), torch.ones(2))
    with pytest.raises(ValueError, match="Nonfinite"):
        a.add(torch.ones(2), torch.tensor([1.0, float("nan")]), torch.ones(2))
    with pytest.raises(ValueError, match="inventories"):
        compare({"a": torch.ones(1)}, {}, {})
    with pytest.raises(ValueError, match="Unclassified"):
        parameter_group("unknown.weight")


def test_actual_flat_dcp_stream_roundtrip_preserves_experts_and_master_precision(tmp_path):
    hf = Olmo3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        attention_hidden_size=16,
        head_dim=8,
        dense_mlp_intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_head_qk_norm=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        latent_moe_dim=8,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        dense_layers_use_shared_expert=True,
        use_peri_ln=True,
        max_position_embeddings=128,
    )
    model = Olmo3MoeForCausalLM(hf)
    reference = {name: value.detach().float().clone() for name, value in model.state_dict().items()}
    native = convert.convert_olmo3moe_state_from_hf(hf, reference)
    meta = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, attention_backend=attention.AttentionBackendName.torch
    ).build(init_device="meta")
    for name, parameter in meta.named_parameters():
        if name.endswith(".attention.w_qkv.weight"):
            native[name] = torch.cat([native.pop(name.replace("w_qkv", suffix)) for suffix in ("w_q", "w_k", "w_v")])
        assert native[name].numel() == parameter.numel()
    dcp.save({f"module.{name}.main": value.flatten() for name, value in native.items()}, checkpoint_id=tmp_path)
    state = CoreCheckpointState(tmp_path, hf, category="fp32_masters")
    recovered = dict(state.stream())
    assert set(recovered) == set(reference)
    assert all(
        value.dtype == torch.float32 and torch.equal(value, reference[name]) for name, value in recovered.items()
    )
    storage = dict(CoreCheckpointState(tmp_path, hf, category="reconstructed_model_storage").stream())
    assert any(value.dtype == torch.bfloat16 for value in storage.values())
    assert all(torch.equal(value, reference[name].to(value.dtype)) for name, value in storage.items())

    subset, inventory, reads = read_core_subset(
        CoreCheckpointState(tmp_path, hf, category="reconstructed_model_storage")
    )
    assert inventory == set(reference)
    assert set(subset) == {name for name in reference if selected(name)}
    assert all(torch.equal(value, storage[name]) for name, value in subset.items())
    assert all(selected(name) for name in reads["native_keys"])
    assert reads["master_payload_bytes"] < sum(value.numel() * 4 for value in native.values())
