# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/norm.py
"""Portable per-head query/key normalization, shared by the vLLM executor and tests."""

import torch
from torch import nn


class CoreRMSNorm(nn.Module):
    """Match OLMo-core's FP32 normalization and gain product, then cast once."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class PerHeadRMSNorm(nn.Module):
    """Normalize over D only, preserving OLMo-core's independent [H,D] gains."""

    def __init__(self, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)
