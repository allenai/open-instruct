"""Precision audits distinguish KV scale loaders from quantized weights."""

import copy

import pytest
from scripts.miles import qualify_math_quantization as qualifier


def inventory():
    return [
        {"name": f"model.layers.{i}.mlp.experts", "class": "FusedMoE", "quant_method": "Fp8MoEMethod"}
        for i in range(1, 16)
    ] + [
        {
            "name": "model.layers.7.self_attn.attn",
            "class": "RadixAttention",
            "quant_method": "Fp8KVCacheMethod",
            "parameters": {key: {"dtype": "torch.float32", "shape": []} for key in ("k_scale", "v_scale")},
        }
    ]


def test_bf16_kv_scale_loader_with_fp8_experts():
    assert qualifier.verify_inventory(inventory(), "fp8", kv_cache_dtype="bf16")["routed_expert_modules"] == 15


@pytest.mark.parametrize("dtype", [None, "auto", "fp8_e4m3"])
def test_cache_dtype_must_be_explicit_bf16(dtype):
    with pytest.raises(ValueError, match="Unverified BF16 KV"):
        qualifier.verify_inventory(inventory(), "fp8", kv_cache_dtype=dtype)


@pytest.mark.parametrize("change", ["weight", "dtype", "class"])
def test_cache_exception_does_not_hide_quantized_nonexperts(change):
    modules = copy.deepcopy(inventory())
    if change == "weight":
        modules[-1]["parameters"]["weight"] = {"dtype": "torch.float8_e4m3fn", "shape": [8, 8]}
    elif change == "dtype":
        modules[-1]["parameters"]["k_scale"]["dtype"] = "torch.float8_e4m3fn"
    else:
        modules[-1]["class"] = "Linear"
    with pytest.raises(ValueError):
        qualifier.verify_inventory(modules, "fp8", kv_cache_dtype="bf16")


def test_other_quantized_nonexpert_still_rejected():
    modules = inventory()
    modules.append({"name": "lm_head", "class": "Linear", "quant_method": "Fp8LinearMethod"})
    with pytest.raises(ValueError, match="Unexpected quantized non-expert"):
        qualifier.verify_inventory(modules, "fp8", kv_cache_dtype="bf16")
