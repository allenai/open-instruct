"""One-time normalization of an untied input embedding at SFT initialization."""

import hashlib

import torch


def tensor_sha256(weight: torch.Tensor) -> str:
    return hashlib.sha256(weight.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


@torch.no_grad()
def normalize_input_rows(weight: torch.Tensor, vocab_limit: int) -> list[int]:
    """Normalize nonzero rows in fp32, preserving zero rows outside the tokenizer."""
    rows = weight.float()
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("Input embeddings contain non-finite values")
    scale = rows.abs().amax(dim=1, keepdim=True)
    zero = scale.eq(0)
    if bool(zero[:vocab_limit].any()):
        raise ValueError("A tokenizer-reachable input embedding has zero norm")
    # Scale first to keep the L2 reduction from underflowing on very small rows.
    scaled = rows / scale.clamp_min(torch.finfo(rows.dtype).tiny)
    norm = scaled.norm(dim=1, keepdim=True)
    normalized = (scaled / norm.masked_fill(zero, 1)).to(weight.dtype)
    error = (normalized.float().norm(dim=1, keepdim=True) - 1).abs()
    if not bool(torch.isfinite(error).all()) or bool((error.masked_fill(zero, 0) > 0.0042).any()):
        raise ValueError("Input embedding norms exceed the rounding tolerance")
    weight.copy_(normalized)
    return zero.squeeze(1).nonzero().flatten().tolist()
