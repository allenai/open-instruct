# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch

from open_instruct.distillkit.signals import SparseSignal


def sparse_hinge_loss(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
    log_target: bool = True,
    margin: float | None = None,
) -> torch.Tensor:
    assert logits.size()[:2] == target_ids.size()[:2] == target_values.size()[:2]
    bsz, seq_len, k = target_ids.shape
    assert target_values.shape == (bsz, seq_len, k)

    student_probs = torch.softmax(logits, dim=-1)
    student_target_probs = student_probs.gather(-1, target_ids)
    teacher_probs = torch.exp(target_values) if log_target else target_values

    prob_diff = student_target_probs.unsqueeze(-1) - student_target_probs.unsqueeze(-2)
    margin_values = (
        teacher_probs.unsqueeze(-1) - teacher_probs.unsqueeze(-2)
        if margin is None
        else margin * torch.ones_like(prob_diff)
    )
    max_terms = torch.relu(margin_values - prob_diff)

    actually_supported_mask = teacher_probs > eps
    supported_k = actually_supported_mask.unsqueeze(-1)
    supported_l = actually_supported_mask.unsqueeze(-2)
    pair_is_genuinely_supported_mask = supported_k & supported_l
    preference_mask = teacher_probs.unsqueeze(-1) > (teacher_probs.unsqueeze(-2) + eps)
    valid_mask = preference_mask & pair_is_genuinely_supported_mask

    active_terms = max_terms * valid_mask.float()
    num_contributing_pairs = valid_mask.float()

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).float()
        active_terms = active_terms * mask_expanded
        num_contributing_pairs = num_contributing_pairs * mask_expanded

    return active_terms.sum() / (num_contributing_pairs.sum() + eps)


class HingeLoss:
    def __init__(self, margin: float = 0.0):
        self.margin = margin

    def __call__(
        self,
        student_logits: torch.Tensor,
        signal: SparseSignal,
        mask: torch.Tensor | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        del num_items_in_batch
        return sparse_hinge_loss(
            logits=student_logits,
            target_ids=signal.sparse_ids,
            target_values=signal.sparse_values,
            mask=mask,
            log_target=signal.log_values,
            margin=self.margin,
        )
