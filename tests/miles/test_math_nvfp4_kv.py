"""Offline fail-closed checks for the inference-only FP4 KV gate."""

import copy

import pytest
from scripts.miles import qualify_math_nvfp4_kv as gate


def audit():
    return {
        "pool": "HybridLinearKVPool",
        "full_layer_count": 2,
        "quant_name": "nvfp4",
        "kv_dtype": "torch.float4_e2m1fn_x2",
        "buffers": {
            key: [{"dtype": "torch.uint8", "shape": [2, 16], "bytes": 32}]
            for key in ("k_buffer", "v_buffer", "native_k_scale_buffer", "native_v_scale_buffer")
        },
        "recurrent_temporal": {"dtype": "torch.float32"},
        "backends": {"full_attn_backend": {"uses_trtllm_gen_native_fp4": True}},
        "modules": [
            {"name": f"model.layers.{index}.mlp.experts", "class": "FusedMoE", "quant_method": "NvFp4MoEMethod"}
            for index in range(1, 16)
        ]
        + [
            {
                "name": "lm_head",
                "class": "ParallelLMHead",
                "quant_method": "UnquantizedLinearMethod",
                "parameters": {"weight": {"dtype": "torch.bfloat16"}},
            }
        ],
    }


def test_valid_native_pool_scope():
    gate.verify_audit(audit())


@pytest.mark.parametrize(
    "key,value",
    [
        ("quant_name", "fp4_mx_block16"),
        ("full_layer_count", 16),
        ("recurrent_temporal", {"dtype": "torch.bfloat16"}),
        ("backends", {"uses_trtllm_gen_native_fp4": False}),
    ],
)
def test_reject_recipe_layer_state_or_backend_substitution(key, value):
    changed = audit()
    changed[key] = value
    with pytest.raises(ValueError):
        gate.verify_audit(changed)


def test_reject_missing_scales_and_quantized_dense_weights():
    changed = audit()
    changed["buffers"]["native_v_scale_buffer"] = None
    with pytest.raises(ValueError):
        gate.verify_audit(changed)
    changed = audit()
    changed["modules"].append(
        {"name": "model.layers.0.self_attn.q_proj", "class": "Linear", "quant_method": "Fp8Method"}
    )
    with pytest.raises(ValueError):
        gate.verify_audit(changed)


def test_finite_candidates_and_complete_short_generation():
    output = {
        "meta_info": {
            "output_token_logprobs": [(-1.0, 42)],
            "output_top_logprobs": [[(-1.0, index) for index in range(20)]],
        }
    }
    gate.verify_outputs([output], 1)
    for bad in (float("nan"), float("inf")):
        changed = copy.deepcopy(output)
        changed["meta_info"]["output_top_logprobs"][0][0] = (bad, 0)
        with pytest.raises(ValueError):
            gate.verify_outputs([changed], 1)
    with pytest.raises(ValueError):
        gate.verify_outputs([output], 16)
