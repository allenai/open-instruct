"""Check the experimental fused rounding against a separate one-step expression."""

from types import SimpleNamespace

import pytest
import torch
from scripts.miles import selective_kda_precision as selective
from scripts.miles import selective_precision_runtime as runtime


def test_forcing_selects_each_requests_causal_position():
    params = [{"prefix_length": 3, "forced_ids": [11, 12]}, {"prefix_length": 8, "forced_ids": [21, 22]}]
    assert runtime.forced_next_tokens(params, [2, 7]) == [11, 21]
    assert runtime.forced_next_tokens(params, [3, 8]) == [12, 22]
    with pytest.raises(ValueError, match="outside continuation"):
        runtime.forced_next_tokens(params, [4, 8])


def test_latent_patch_skips_dense_first_block_and_restores_forward():
    model = torch.nn.Module()
    model.dense = torch.nn.Module()
    model.dense.feed_forward_norm = torch.nn.Identity()
    model.sparse = torch.nn.Module()
    model.sparse.feed_forward_norm = torch.nn.Identity()
    model.sparse.latent_up_proj = torch.nn.Linear(2, 2, bias=False)
    original = model.sparse.latent_up_proj.forward
    assert runtime.set_linear_variant(model, "latent_up", serving=False) == {"latent_up": 1}
    assert not hasattr(model.dense.feed_forward_norm, "_diagnostic_original_forward")
    runtime.set_linear_variant(model, "none", serving=False)
    assert model.sparse.latent_up_proj.forward == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diagnostic")
def test_fp32_latent_norm_returns_bf16_at_residual_boundary():
    torch.manual_seed(73)
    value = torch.randn(5, 1024, device="cuda")
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    module = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).bfloat16()
    actual = runtime.latent_norm(module, True, value)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diagnostic")
def test_qk_round_matches_separate_bf16_normalization(tmp_path):
    torch.manual_seed(41)
    q, k, v = [torch.randn(1, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    raw_gate = torch.full_like(q, -3.0)
    raw_beta = torch.zeros(1, 2, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(1, 2, 64, 64, device="cuda") * 0.1
    norm_q = (q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16().float()
    norm_k = (k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16().float()
    expected = initial * torch.exp(-torch.nn.functional.softplus(raw_gate.float())).unsqueeze(-2)
    delta = (v.float() - (expected * norm_k.unsqueeze(-2)).sum(-1)) * 0.5
    expected = expected + delta.unsqueeze(-1) * norm_k.unsqueeze(-2)
    expected_output = (expected * (norm_q / 8).unsqueeze(-2)).sum(-1).bfloat16()
    packed = torch.cat([x.flatten(1) for x in (q, k, v)], -1)
    results = {}
    for name in ("baseline", "qk_round"):
        function, _ = selective.load_variant(name, tmp_path)
        state = initial.clone()
        output = function(
            packed,
            raw_gate,
            raw_beta,
            a_log=torch.zeros(2, device="cuda"),
            dt_bias=torch.zeros(128, device="cuda"),
            scale=1 / 8,
            state=state,
            state_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
            num_value_heads=2,
            value_dim=64,
            allow_neg_eigval=False,
        )
        results[name] = (output, state)
    torch.testing.assert_close(results["qk_round"][1], expected, rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(results["qk_round"][0][0], expected_output, rtol=0, atol=0)
    assert (results["baseline"][1] - expected).abs().max() > 1e-4
