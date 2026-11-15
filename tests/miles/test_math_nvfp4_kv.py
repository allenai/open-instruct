"""Offline fail-closed checks for the inference-only FP4 KV gate."""

import copy
from types import SimpleNamespace

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


def test_audit_waits_for_scheduler_pool_and_backend_initialization(monkeypatch):
    runner = SimpleNamespace()
    observed = []

    class Scheduler:
        def __init__(self, marker):
            # ModelRunner construction alone does not allocate a pool on these pins.
            self.tp_worker = SimpleNamespace(model_runner=runner)
            assert not hasattr(runner, "token_to_kv_pool")
            runner.token_to_kv_pool = marker
            runner.attn_backend = "initialized"

    def inspect(completed):
        observed.append((completed.token_to_kv_pool, completed.attn_backend))

    monkeypatch.setattr(gate, "write_runner_audit", inspect)
    gate.install_scheduler_audit(Scheduler)
    Scheduler("pool")
    assert observed == [("pool", "initialized")]


def test_failed_native_initialization_does_not_emit_audit(monkeypatch):
    observed = []

    class Scheduler:
        def __init__(self):
            raise RuntimeError("native pool allocation failed")

    monkeypatch.setattr(gate, "write_runner_audit", observed.append)
    gate.install_scheduler_audit(Scheduler)
    with pytest.raises(RuntimeError, match="native pool allocation failed"):
        Scheduler()
    assert observed == []
