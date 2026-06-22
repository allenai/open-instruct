# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

"""Sparse distribution signals used by distillation losses."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseTeacherSignal:
    """Top-k teacher distribution values aligned to sequence positions.

    `token_ids` and `logprobs` are expected to have identical shape
    `[..., k]`. Non-trainable or missing positions should use token id 0 and
    `-inf` logprob values.
    """

    token_ids: torch.Tensor
    logprobs: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_ids.shape != self.logprobs.shape:
            raise ValueError(f"token_ids shape {self.token_ids.shape} != logprobs shape {self.logprobs.shape}")
        if self.token_ids.ndim < 1:
            raise ValueError("SparseTeacherSignal tensors must have at least one dimension")
        if not torch.is_floating_point(self.logprobs):
            raise TypeError("SparseTeacherSignal.logprobs must be a floating point tensor")
        if self.token_ids.dtype != torch.long:
            raise TypeError("SparseTeacherSignal.token_ids must have dtype torch.long")

    @property
    def topk(self) -> int:
        return self.token_ids.shape[-1]

    def to(self, device: torch.device, non_blocking: bool = True) -> "SparseTeacherSignal":
        return SparseTeacherSignal(
            token_ids=self.token_ids.to(device, non_blocking=non_blocking),
            logprobs=self.logprobs.to(device, non_blocking=non_blocking),
        )
