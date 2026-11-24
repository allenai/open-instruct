# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/expert_reference.py
"""Opt-in native-vLLM expert arithmetic matching the HF BF16/FP32 rounding points."""

import torch
from torch.nn import functional as F


def grouped_experts(layer, x, topk_weights, topk_ids, shared_experts_input=None, **kwargs):
    """Keep GEMMs/SiLU in BF16, then combine weighted expert outputs in FP32."""
    del shared_experts_input, kwargs
    w13, w2 = layer.w13_weight, layer.w2_weight
    if x.dtype != torch.bfloat16 or w13.dtype != x.dtype or w2.dtype != x.dtype:
        raise RuntimeError("Grouped reference supports only unquantized BF16 experts")
    experts, twice_hidden, width = w13.shape
    hidden = twice_hidden // 2
    if w2.shape != (experts, width, hidden) or x.shape[-1] != width:
        raise RuntimeError("Unexpected native fused-expert weight layout")
    n, topk = topk_ids.shape
    ids = topk_ids.reshape(-1).long()
    tokens = torch.arange(n, device=x.device).repeat_interleave(topk)
    order = ids.argsort(stable=True)
    sorted_ids = ids[order]
    sorted_tokens = tokens[order]
    offsets = torch.bincount(sorted_ids, minlength=experts).cumsum(0).to(torch.int32)
    up_gate = F.grouped_mm(x[sorted_tokens], w13.transpose(1, 2), offs=offsets)
    gate, up = up_gate.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    outputs = F.grouped_mm(activated, w2.transpose(1, 2), offs=offsets)
    weighted = outputs.float() * topk_weights.reshape(-1)[order].float()[:, None]
    # Match the independent HF loop's expert-ascending FP32 accumulation order.
    token_expert_order = (sorted_tokens * experts + sorted_ids).argsort()
    ordered = weighted[token_expert_order].reshape(n, topk, width)
    output = torch.zeros((n, width), dtype=torch.float32, device=x.device)
    for slot in range(topk):
        output = output + ordered[:, slot]
    return output.to(x.dtype)
