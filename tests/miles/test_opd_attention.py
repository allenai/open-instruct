"""Exercise the pilot's padded Megatron GQA/FlashAttention path before model startup."""

import math

import torch
from transformer_engine import pytorch as te


def test_padded_qwen_attention_forward_backward(monkeypatch):
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    torch.manual_seed(17)
    length, padded, heads, kv_heads, dimension = 319, 320, 8, 2, 256
    q = torch.randn(padded, 1, heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(padded, 1, kv_heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(padded, 1, kv_heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attention = te.DotProductAttention(
        heads, dimension, num_gqa_groups=kv_heads, attention_dropout=0.0, qkv_format="sbhd", attn_mask_type="causal"
    )
    output = attention(q, k, v)
    output[:length].float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(output[:length]).all()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().max() > 0
    with torch.no_grad():
        q_ref = q[:length, 0].float().transpose(0, 1)
        k_ref = k[:length, 0].float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        v_ref = v[:length, 0].float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        scores = q_ref @ k_ref.transpose(-1, -2) / math.sqrt(dimension)
        causal = torch.ones(length, length, device="cuda", dtype=torch.bool).tril()
        expected = (scores.masked_fill(~causal, -torch.inf).softmax(-1) @ v_ref).transpose(0, 1)
        torch.testing.assert_close(
            output[:length].reshape(length, heads, dimension).float(), expected, atol=0.02, rtol=0.02
        )
