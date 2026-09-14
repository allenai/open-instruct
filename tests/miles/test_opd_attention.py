"""Exercise Qwen's packed GQA shape before paying for model startup."""

import math

import torch
from transformer_engine import pytorch as te


def test_packed_qwen_attention_forward_backward():
    torch.manual_seed(17)
    length, padded, heads, kv_heads, dimension = 319, 320, 8, 2, 256
    q = torch.randn(padded, heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(padded, kv_heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(padded, kv_heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    cu_padded = torch.tensor([0, padded], device="cuda", dtype=torch.int32)
    attention = te.DotProductAttention(
        heads,
        dimension,
        num_gqa_groups=kv_heads,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding_causal",
    )
    output = attention(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=padded,
        max_seqlen_kv=padded,
    )
    output[:length].float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(output[:length]).all()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().max() > 0
    with torch.no_grad():
        q_ref = q[:length].float().transpose(0, 1)
        k_ref = k[:length].float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        v_ref = v[:length].float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        scores = q_ref @ k_ref.transpose(-1, -2) / math.sqrt(dimension)
        causal = torch.ones(length, length, device="cuda", dtype=torch.bool).tril()
        expected = (scores.masked_fill(~causal, -torch.inf).softmax(-1) @ v_ref).transpose(0, 1)
        torch.testing.assert_close(
            output[:length].reshape(length, heads, dimension).float(), expected, atol=0.02, rtol=0.02
        )
