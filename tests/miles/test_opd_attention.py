"""Exercise native Miles packed batching and the pilot's Qwen FlashAttention path."""

import math
from types import SimpleNamespace

import torch
from miles.backends.megatron_utils import parallel
from miles.backends.training_utils import cp_utils, data
from transformer_engine import pytorch as te


def test_native_packed_qwen_attention_forward_backward(monkeypatch):
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=2))
    monkeypatch.setattr(data, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    torch.manual_seed(17)
    length, heads, kv_heads, dimension = 319, 8, 2, 256
    rollout = {
        "tokens": [torch.arange(length, device="cuda")],
        "total_lengths": [length],
        "response_lengths": [7],
        "loss_masks": [torch.ones(7, device="cuda", dtype=torch.int32)],
    }
    batch = data.get_batch(data.DataIterator(rollout, micro_batch_size=1), list(rollout), qkv_format="thd")
    args = SimpleNamespace(qkv_format="thd", use_opd=True, context_parallel_size=1)
    packed = parallel.get_packed_seq_params(batch, args)
    padded = batch["tokens"].numel()
    assert packed.cu_seqlens_q.tolist() == [0, length, padded]
    assert packed.pad_between_seqs is False
    assert batch["full_loss_masks"][0, length:].count_nonzero() == 0
    q = torch.randn(padded, heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(padded, kv_heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(padded, kv_heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
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
        cu_seqlens_q=packed.cu_seqlens_q,
        cu_seqlens_kv=packed.cu_seqlens_kv,
        max_seqlen_q=packed.max_seqlen_q,
        max_seqlen_kv=packed.max_seqlen_kv,
        pad_between_seqs=packed.pad_between_seqs,
    )
    output[:length].float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(output[:length]).all()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().max() > 0
        assert tensor.grad[length:].count_nonzero() == 0
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
