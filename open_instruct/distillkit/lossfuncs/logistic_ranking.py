# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch
import torch.nn.functional as F

from open_instruct.distillkit.signals import SparseSignal


def sparse_logistic_ranking_loss(
    student_logits: torch.Tensor,
    teacher_target_ids: torch.LongTensor,
    teacher_target_values: torch.Tensor,
    log_target: bool = True,
    sequence_mask: torch.Tensor | None = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    teacher_probs = (
        torch.exp(teacher_target_values) if log_target else teacher_target_values
    )
    student_logits_at_targets = torch.gather(
        student_logits, dim=2, index=teacher_target_ids
    )

    actually_supported_mask = teacher_probs > eps
    teacher_prefers_k_over_l_mask = teacher_probs.unsqueeze(-1) > (
        teacher_probs.unsqueeze(-2) + eps
    )
    student_logit_diff_k_minus_l = student_logits_at_targets.unsqueeze(
        -1
    ) - student_logits_at_targets.unsqueeze(-2)
    supported_k = actually_supported_mask.unsqueeze(-1)
    supported_l = actually_supported_mask.unsqueeze(-2)
    pair_is_supported_mask = supported_k & supported_l
    valid_preference_pair_mask = teacher_prefers_k_over_l_mask & pair_is_supported_mask

    pair_loss = F.softplus(-student_logit_diff_k_minus_l)
    masked_pair_loss = pair_loss * valid_preference_pair_mask.float()
    sum_pair_loss_per_pos = masked_pair_loss.sum(dim=(2, 3))

    if sequence_mask is None:
        active_sequence_mask = torch.ones_like(
            sum_pair_loss_per_pos, dtype=torch.bool, device=student_logits.device
        )
    else:
        if sequence_mask.dim() == 3:
            sequence_mask = sequence_mask.squeeze(-1)
        active_sequence_mask = sequence_mask.bool()

    final_summed_loss = (sum_pair_loss_per_pos * active_sequence_mask.float()).sum()
    num_contributing_pairs = (
        valid_preference_pair_mask.float()
        * active_sequence_mask.float().unsqueeze(-1).unsqueeze(-1)
    ).sum()
    return final_summed_loss / (num_contributing_pairs + eps)


class LogisticRankingLoss:
    def __call__(
        self,
        student_logits: torch.Tensor,
        signal: SparseSignal,
        mask: torch.Tensor | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        del num_items_in_batch
        return sparse_logistic_ranking_loss(
            student_logits=student_logits,
            teacher_target_ids=signal.sparse_ids,
            teacher_target_values=signal.sparse_values,
            sequence_mask=mask,
            log_target=signal.log_values,
        )
