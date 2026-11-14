"""Prevent a nominal expert-only benchmark from silently quantizing dense layers."""

import pytest
from scripts.miles.qualify_math_quantization import verify_inventory


def inventory(method):
    return [{"name": f"model.layers.{index}.mlp.experts", "quant_method": method} for index in range(1, 16)]


@pytest.mark.parametrize(
    "precision,method",
    [("fp8", "Fp8MoEMethod"), ("nvfp4", "ModelOptNvFp4OnlineFusedMoEMethod"), ("bf16", "UnquantizedFusedMoEMethod")],
)
def test_expected_expert_quantization(precision, method):
    assert verify_inventory(inventory(method), precision)["routed_expert_modules"] == 15


def test_dense_quantization_and_missing_experts_fail():
    modules = inventory("Fp8MoEMethod")
    modules.append({"name": "model.layers.1.self_attn.qkv_proj", "quant_method": "Fp8LinearMethod"})
    with pytest.raises(ValueError, match="non-expert"):
        verify_inventory(modules, "fp8")
    with pytest.raises(ValueError, match="15 routed"):
        verify_inventory(inventory("Fp8MoEMethod")[:-1], "fp8")
    with pytest.raises(ValueError, match="not quantized"):
        verify_inventory(inventory("UnquantizedFusedMoEMethod"), "nvfp4")
