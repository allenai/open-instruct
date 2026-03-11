# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import math

import torch

from open_instruct.distillkit.lossfuncs.common import (
    MissingProbabilityHandling,
    accumulate_over_chunks,
    get_logprobs,
)
from open_instruct.distillkit.signals import SparseSignal


def sparse_jsd_inner(
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
    sparse_student_probs = torch.exp(sparse_student_logprobs)
    sparse_teacher_probs = torch.exp(sparse_target_logprobs)

    m_sparse_probs = 0.5 * (sparse_teacher_probs + sparse_student_probs)
    log_m_sparse = torch.log(m_sparse_probs.clamp(min=eps))

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        teacher_prob_sum_sparse = sparse_teacher_probs.to(torch.float32).sum(
            dim=-1, keepdim=True
        )
        log_teacher_missing_prob = torch.log1p(
            -teacher_prob_sum_sparse.clamp(min=eps, max=1.0 - eps)
        )
        teacher_missing_prob = torch.exp(log_teacher_missing_prob)

        student_prob_sum_sparse = sparse_student_probs.to(torch.float32).sum(
            dim=-1, keepdim=True
        )
        log_student_missing_prob = torch.log1p(
            -student_prob_sum_sparse.clamp(min=eps, max=1.0 - eps)
        )
        student_missing_prob = torch.exp(log_student_missing_prob)

        m_missing_prob = 0.5 * (teacher_missing_prob + student_missing_prob)
        log_m_missing = torch.log(m_missing_prob.clamp(min=eps))

    kl_p_m_sparse_terms = sparse_teacher_probs * (sparse_target_logprobs - log_m_sparse)
    kl_p_m_sparse_sum = torch.sum(
        torch.where(
            sparse_teacher_probs > eps,
            kl_p_m_sparse_terms,
            torch.zeros_like(kl_p_m_sparse_terms),
        ),
        dim=-1,
    )

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        kl_p_m_missing_term = teacher_missing_prob * (
            log_teacher_missing_prob - log_m_missing
        )
        kl_p_m = kl_p_m_sparse_sum + torch.where(
            teacher_missing_prob.squeeze(-1) > eps,
            kl_p_m_missing_term.squeeze(-1),
            torch.zeros_like(kl_p_m_missing_term.squeeze(-1)),
        )
    else:
        kl_p_m = kl_p_m_sparse_sum

    kl_q_m_sparse_terms = sparse_student_probs * (
        sparse_student_logprobs - log_m_sparse
    )
    kl_q_m_sparse_sum = torch.sum(
        torch.where(
            sparse_student_probs > eps,
            kl_q_m_sparse_terms,
            torch.zeros_like(kl_q_m_sparse_terms),
        ),
        dim=-1,
    )

    if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
        kl_q_m_missing_term = student_missing_prob * (
            log_student_missing_prob - log_m_missing
        )
        kl_q_m = kl_q_m_sparse_sum + torch.where(
            student_missing_prob.squeeze(-1) > eps,
            kl_q_m_missing_term.squeeze(-1),
            torch.zeros_like(kl_q_m_missing_term.squeeze(-1)),
        )
    else:
        student_prob_sum_sparse = sparse_student_probs.sum(dim=-1)
        student_total_missing_prob_mass = (1.0 - student_prob_sum_sparse).clamp(min=0.0)
        kl_q_m = kl_q_m_sparse_sum + student_total_missing_prob_mass * math.log(2.0)

    jsd_terms = 0.5 * (kl_p_m + kl_q_m)
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        jsd_terms = jsd_terms * mask
    return torch.sum(jsd_terms).to(out_dtype)


def sparse_js_div(
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
        sparse_jsd_inner,
        eps=eps,
        missing=missing,
        log_target=log_target,
        temperature=temperature,
        target_generation_temperature=target_generation_temperature,
        student_generation_temperature=student_generation_temperature,
    )


class JSDLoss:
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
        res = sparse_js_div(
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
