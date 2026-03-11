# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch

from open_instruct.distillkit.lossfuncs.common import (
    MissingProbabilityHandling,
    accumulate_over_chunks,
    get_logprobs,
)
from open_instruct.distillkit.signals import SparseSignal


def sparse_kl_div_inner(
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
    out_dtype = logits.dtype
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

    teacher_sparse_probs = torch.exp(sparse_target_logprobs)
    teacher_prob_sum = teacher_sparse_probs.to(torch.float32).sum(dim=-1)
    inner_sum = torch.sum(
        teacher_sparse_probs * (sparse_target_logprobs - sparse_student_logprobs),
        dim=-1,
    )

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        log_teacher_missing = torch.log1p(-teacher_prob_sum.clamp(min=eps, max=1 - eps))
        student_probs = sparse_student_logprobs.to(torch.float32).exp_()
        student_prob_sum = student_probs.sum(dim=-1)
        log_student_missing = torch.log1p(-student_prob_sum.clamp(min=eps, max=1 - eps))
        missing_kl = torch.exp(log_teacher_missing) * (
            log_teacher_missing - log_student_missing
        )
    else:
        missing_kl = None

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        inner_sum *= mask
        if missing_kl is not None:
            missing_kl *= mask

    if missing_kl is not None:
        inner_sum += missing_kl
    return torch.sum(inner_sum).to(out_dtype)


def sparse_kl_div(
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
        sparse_kl_div_inner,
        eps=eps,
        missing=missing,
        log_target=log_target,
        temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )


class KLDLoss:
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
        res = sparse_kl_div(
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
