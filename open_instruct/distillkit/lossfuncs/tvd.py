# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch

from open_instruct.distillkit.lossfuncs.common import (
    MissingProbabilityHandling,
    accumulate_over_chunks,
    get_logprobs,
)
from open_instruct.distillkit.signals import SparseSignal


def sparse_tvd_inner(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    log_target: bool = True,
    temperature: float = 1.0,
    target_generation_temperature: float = 1.0,
    student_generation_temperature: float = 1.0,
) -> torch.Tensor:
    sparse_student_logprobs, sparse_target_logprobs = get_logprobs(
        logits,
        target_ids,
        target_values,
        eps=eps,
        missing=missing,
        log_target=log_target,
        distillation_temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )
    sparse_student_probs = torch.exp(sparse_student_logprobs)
    sparse_teacher_probs = torch.exp(sparse_target_logprobs)
    tvd_sparse_terms_sum = torch.sum(
        torch.abs(sparse_teacher_probs - sparse_student_probs), dim=-1
    )

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        teacher_prob_sum_sparse = sparse_teacher_probs.sum(dim=-1)
        teacher_missing_prob_mass = (1.0 - teacher_prob_sum_sparse).clamp(
            min=0.0, max=1.0
        )
        student_prob_sum_sparse = sparse_student_probs.sum(dim=-1)
        student_missing_prob_mass = (1.0 - student_prob_sum_sparse).clamp(
            min=0.0, max=1.0
        )
        tvd_missing_contrib = torch.abs(
            teacher_missing_prob_mass - student_missing_prob_mass
        )
    else:
        student_prob_sum_sparse = sparse_student_probs.sum(dim=-1)
        tvd_missing_contrib = (1.0 - student_prob_sum_sparse).clamp(min=0.0)

    tvd_token_level = 0.5 * (tvd_sparse_terms_sum + tvd_missing_contrib)
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        tvd_token_level = tvd_token_level * mask
    return torch.sum(tvd_token_level)


def sparse_tvd(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    log_target: bool = True,
    temperature: float = 1.0,
    target_generation_temperature: float = 1.0,
    student_generation_temperature: float = 1.0,
    chunk_length: int | None = None,
) -> torch.Tensor:
    return accumulate_over_chunks(
        logits,
        target_ids,
        target_values,
        mask,
        chunk_length,
        sparse_tvd_inner,
        eps=eps,
        missing=missing,
        log_target=log_target,
        temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )


class TVDLoss:
    def __init__(
        self,
        temperature: float = 1.0,
        missing_probability_handling: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
        sparse_chunk_length: int | None = None,
    ):
        self.temperature = temperature
        self.missing = missing_probability_handling
        self.chunk_length = sparse_chunk_length

    def __call__(
        self,
        student_logits: torch.Tensor,
        signal: SparseSignal,
        mask: torch.Tensor | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if num_items_in_batch is None:
            num_items_in_batch = (
                mask.float().sum()
                if mask is not None
                else student_logits.shape[0] * student_logits.shape[1]
            )
        res = sparse_tvd(
            logits=student_logits,
            target_ids=signal.sparse_ids,
            target_values=signal.sparse_values,
            mask=mask,
            missing=self.missing,
            log_target=signal.log_values,
            temperature=self.temperature,
            target_generation_temperature=signal.generation_temperature,
            chunk_length=self.chunk_length,
        )
        return res * (self.temperature**2) / num_items_in_batch
