# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

"""Sparse distillation losses."""

from dataclasses import dataclass

import torch

from open_instruct.distillkit.signals import SparseTeacherSignal


@dataclass(frozen=True)
class ForwardKLTopKOutput:
    """Per-token sparse forward-KL output and diagnostics."""

    loss: torch.Tensor
    teacher_topk_mass: torch.Tensor


def gather_student_logprobs_at_teacher_topk(
    student_logits: torch.Tensor, teacher_token_ids: torch.Tensor
) -> torch.Tensor:
    """Gather student logprobs at teacher-selected top-k token ids.

    Args:
        student_logits: Tensor of shape `[..., vocab_size]`.
        teacher_token_ids: Tensor of shape `[..., k]`.

    Returns:
        Tensor of shape `[..., k]` containing log-softmax student values.
    """
    if student_logits.shape[:-1] != teacher_token_ids.shape[:-1]:
        raise ValueError(
            f"student_logits prefix shape {student_logits.shape[:-1]} does not match "
            f"teacher_token_ids prefix shape {teacher_token_ids.shape[:-1]}"
        )
    safe_token_ids = teacher_token_ids.to(student_logits.device).clamp(min=0, max=student_logits.shape[-1] - 1)
    student_logprobs = torch.log_softmax(student_logits.float(), dim=-1)
    return torch.gather(student_logprobs, dim=-1, index=safe_token_ids)


def forward_kl_topk_from_logprobs(
    student_topk_logprobs: torch.Tensor, teacher_signal: SparseTeacherSignal, *, normalize_topk: bool = False
) -> ForwardKLTopKOutput:
    """Compute teacher-top-k forward KL without requiring full teacher logits.

    By default this uses the unnormalized teacher top-k probabilities:

    `sum_k exp(teacher_logprob_k) * (teacher_logprob_k - student_logprob_k)`

    This matches the OPD approximation where the teacher tail mass is omitted.
    Set `normalize_topk=True` only when intentionally training against the
    teacher distribution renormalized over the observed top-k support.
    """
    if student_topk_logprobs.shape != teacher_signal.logprobs.shape:
        raise ValueError(
            f"student_topk_logprobs shape {student_topk_logprobs.shape} != "
            f"teacher logprobs shape {teacher_signal.logprobs.shape}"
        )

    finite_teacher = torch.isfinite(teacher_signal.logprobs)
    safe_teacher_logprobs = torch.where(
        finite_teacher, teacher_signal.logprobs.float(), torch.zeros_like(teacher_signal.logprobs.float())
    )
    teacher_probs = torch.where(
        finite_teacher, torch.exp(teacher_signal.logprobs.float()), torch.zeros_like(student_topk_logprobs.float())
    )
    teacher_topk_mass = teacher_probs.sum(dim=-1)

    if normalize_topk:
        denom = teacher_topk_mass.clamp_min(torch.finfo(teacher_probs.dtype).tiny).unsqueeze(-1)
        teacher_probs = teacher_probs / denom
        safe_teacher_logprobs = torch.where(
            finite_teacher, safe_teacher_logprobs - torch.log(denom), torch.zeros_like(safe_teacher_logprobs)
        )

    # Zero the student term where the teacher entry is missing (-inf). Otherwise a
    # non-finite student logprob there would give 0 * inf = NaN, since teacher_probs
    # is already 0 at those positions.
    safe_student_logprobs = torch.where(
        finite_teacher, student_topk_logprobs.float(), torch.zeros_like(student_topk_logprobs.float())
    )
    loss = (teacher_probs * (safe_teacher_logprobs - safe_student_logprobs)).sum(dim=-1)
    return ForwardKLTopKOutput(loss=loss, teacher_topk_mass=teacher_topk_mass)
