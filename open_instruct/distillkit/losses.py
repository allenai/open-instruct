# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

"""Sparse distillation losses."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ForwardKLTopKOutput:
    """Per-token sparse forward-KL output and diagnostics."""

    loss: torch.Tensor
    teacher_topk_mass: torch.Tensor


def forward_kl_topk_from_logprobs(
    student_topk_logprobs: torch.Tensor, teacher_topk_logprobs: torch.Tensor
) -> ForwardKLTopKOutput:
    """Compute teacher-top-k forward KL without requiring full teacher logits.

    Uses the unnormalized teacher top-k probabilities:

    `sum_k exp(teacher_logprob_k) * (teacher_logprob_k - student_logprob_k)`

    This matches the OPD approximation where the teacher tail mass is omitted.
    `student_topk_logprobs` and `teacher_topk_logprobs` must have identical shape
    `[..., k]`; missing teacher entries use `-inf` logprobs and contribute 0.
    """
    if student_topk_logprobs.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            f"student_topk_logprobs shape {student_topk_logprobs.shape} != "
            f"teacher logprobs shape {teacher_topk_logprobs.shape}"
        )

    finite_teacher = torch.isfinite(teacher_topk_logprobs)
    safe_teacher_logprobs = torch.where(
        finite_teacher, teacher_topk_logprobs.float(), torch.zeros_like(teacher_topk_logprobs.float())
    )
    teacher_probs = torch.where(
        finite_teacher, torch.exp(teacher_topk_logprobs.float()), torch.zeros_like(student_topk_logprobs.float())
    )
    teacher_topk_mass = teacher_probs.sum(dim=-1)

    # Zero the student term where the teacher entry is missing (-inf). Otherwise a
    # non-finite student logprob there would give 0 * inf = NaN, since teacher_probs
    # is already 0 at those positions.
    safe_student_logprobs = torch.where(
        finite_teacher, student_topk_logprobs.float(), torch.zeros_like(student_topk_logprobs.float())
    )
    loss = (teacher_probs * (safe_teacher_logprobs - safe_student_logprobs)).sum(dim=-1)
    return ForwardKLTopKOutput(loss=loss, teacher_topk_mass=teacher_topk_mass)
