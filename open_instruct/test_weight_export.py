import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from open_instruct.weight_export import (
    PassthroughWeightExportAdapter,
    Qwen3MoeWeightExportAdapter,
    map_weight_name,
    resolve_weight_export_adapter,
)


def test_dense_parameter_passthrough_preserves_tensor() -> None:
    adapter = PassthroughWeightExportAdapter()
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    specs = list(adapter.export_specs("model.layers.0.self_attn.q_proj.weight", tuple(tensor.shape), tensor.dtype))
    exported = list(adapter.export_tensors("model.layers.0.self_attn.q_proj.weight", tensor))

    assert [(spec.name, spec.shape, spec.dtype) for spec in specs] == [
        ("model.layers.0.self_attn.q_proj.weight", (2, 3), torch.float32)
    ]
    assert exported == [("model.layers.0.self_attn.q_proj.weight", tensor)]
    assert exported[0][1] is tensor


def test_qwen3_moe_gate_up_expansion_matches_metadata_and_values() -> None:
    adapter = Qwen3MoeWeightExportAdapter()
    name = "model.layers.0.mlp.experts.gate_up_proj"
    tensor = torch.arange(2 * 6 * 2, dtype=torch.bfloat16).reshape(2, 6, 2)
    original = tensor.clone()

    specs = list(adapter.export_specs(name, tuple(tensor.shape), tensor.dtype))
    exported = list(adapter.export_tensors(name, tensor))

    expected_names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    ]
    assert [spec.name for spec in specs] == expected_names
    assert [exported_name for exported_name, _ in exported] == expected_names
    assert [spec.shape for spec in specs] == [(3, 2)] * 4
    assert [tuple(value.shape) for _, value in exported] == [(3, 2)] * 4
    assert all(spec.dtype == torch.bfloat16 for spec in specs)
    assert all(value.dtype == tensor.dtype and value.device == tensor.device for _, value in exported)
    torch.testing.assert_close(exported[0][1], tensor[0, :3])
    torch.testing.assert_close(exported[1][1], tensor[0, 3:])
    torch.testing.assert_close(exported[2][1], tensor[1, :3])
    torch.testing.assert_close(exported[3][1], tensor[1, 3:])
    torch.testing.assert_close(tensor, original)


def test_qwen3_moe_down_expansion_matches_metadata_and_values() -> None:
    adapter = Qwen3MoeWeightExportAdapter()
    name = "model.layers.1.mlp.experts.down_proj"
    tensor = torch.arange(2 * 2 * 3, dtype=torch.float16).reshape(2, 2, 3)

    specs = list(adapter.export_specs(name, tuple(tensor.shape), tensor.dtype))
    exported = list(adapter.export_tensors(name, tensor))

    expected_names = ["model.layers.1.mlp.experts.0.down_proj.weight", "model.layers.1.mlp.experts.1.down_proj.weight"]
    assert [spec.name for spec in specs] == expected_names
    assert [name for name, _ in exported] == expected_names
    assert [spec.shape for spec in specs] == [(2, 3), (2, 3)]
    torch.testing.assert_close(exported[0][1], tensor[0])
    torch.testing.assert_close(exported[1][1], tensor[1])


def test_qwen3_moe_rejects_malformed_fused_parameters() -> None:
    adapter = Qwen3MoeWeightExportAdapter()
    name = "model.layers.0.mlp.experts.gate_up_proj"

    with pytest.raises(ValueError, match=r"gate_up_proj.*odd projection dimension.*\(2, 5, 3\)"):
        list(adapter.export_specs(name, (2, 5, 3), torch.float32))
    with pytest.raises(ValueError, match=r"gate_up_proj.*rank 3.*\(2, 3\)"):
        list(adapter.export_tensors(name, torch.zeros(2, 3)))


def test_shared_expert_is_not_expanded() -> None:
    adapter = Qwen3MoeWeightExportAdapter()
    name = "model.layers.0.mlp.shared_expert.gate_up_proj"
    tensor = torch.zeros(8, 4)

    assert list(adapter.export_tensors(name, tensor)) == [(name, tensor)]


def test_name_mapping_precedes_qwen_expansion() -> None:
    adapter = Qwen3MoeWeightExportAdapter()
    mapped_name = map_weight_name(
        "module.model.layers.0.mlp.experts.down_proj", lambda name: name.removeprefix("module.")
    )

    specs = list(adapter.export_specs(mapped_name, (2, 4, 3), torch.float32))

    assert [spec.name for spec in specs] == [
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ]


def test_adapter_resolution_is_explicitly_model_type_based() -> None:
    qwen_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))
    dense_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))

    assert isinstance(resolve_weight_export_adapter(qwen_model), Qwen3MoeWeightExportAdapter)
    assert isinstance(resolve_weight_export_adapter(dense_model), PassthroughWeightExportAdapter)


def test_transformers_553_qwen3_moe_uses_expected_fused_parameter_layout() -> None:
    assert transformers.__version__ == "5.5.3"
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        head_dim=4,
    )
    model = Qwen3MoeForCausalLM(config)
    parameters = dict(model.named_parameters())

    assert config.model_type == "qwen3_moe"
    assert tuple(parameters["model.layers.0.mlp.experts.gate_up_proj"].shape) == (2, 8, 8)
    assert tuple(parameters["model.layers.0.mlp.experts.down_proj"].shape) == (2, 8, 4)
    assert tuple(parameters["model.layers.0.mlp.gate.weight"].shape) == (2, 8)


def test_export_iterator_does_not_retain_previously_yielded_expert_copy() -> None:
    adapter = Qwen3MoeWeightExportAdapter()
    tensor = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)[:, :, ::2]
    iterator = adapter.export_tensors("model.layers.0.mlp.experts.gate_up_proj", tensor)

    first = next(iterator)[1]
    first_ref = weakref.ref(first)
    del first
    gc.collect()

    assert first_ref() is None
    assert next(iterator)[0] == "model.layers.0.mlp.experts.0.up_proj.weight"
