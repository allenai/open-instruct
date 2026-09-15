"""Validate and mask teacher supervision before Miles computes advantages."""

import torch
from torch import distributed as dist

from open_instruct.miles import contract


def prepare(rollout):
    required = ("log_probs", "teacher_log_probs", "metadata", "loss_masks", "response_lengths")
    if any(key not in rollout for key in required):
        raise ValueError("Core OPD requires pre-update scores, teacher scores, and alignment metadata")
    count = len(rollout["response_lengths"])
    if any(len(rollout[key]) != count for key in required):
        raise ValueError("OPD sample count mismatch")
    reverse = []
    for student, teacher, metadata, active, length in zip(*(rollout[key] for key in required), strict=True):
        mask = torch.as_tensor(metadata["opd_alignment_mask"], device=student.device)
        if any(value.shape != (length,) for value in (student, teacher, mask, active)):
            raise ValueError("OPD response score/mask length mismatch")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("OPD alignment mask must be binary")
        if not bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()):
            raise ValueError("OPD scores must be finite before advantage calculation")
        reverse.append(torch.where(mask.bool() & active.bool(), student.detach() - teacher.detach(), 0.0))
    rollout["opd_reverse_kl"] = reverse


def record(args, rollout, rollout_id):
    device = rollout["log_probs"][0].device
    matched = sum(
        int((torch.as_tensor(meta["opd_alignment_mask"], device=device).bool() & active.bool()).sum())
        for meta, active in zip(rollout["metadata"], rollout["loss_masks"], strict=True)
    )
    values = torch.tensor(
        [
            matched,
            sum(int(mask.sum()) for mask in rollout["loss_masks"]),
            sum(float(value.sum()) for value in rollout["opd_reverse_kl"]),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values)
    matched, active, reverse_sum = values.tolist()
    if matched == 0:
        raise ValueError("OPD batch has no aligned active tokens")
    contract.record(
        args,
        {
            "event": "opd_alignment",
            "rollout_id": rollout_id,
            "matched_tokens": matched,
            "active_tokens": active,
            "coverage": matched / active,
            "sampled_log_ratio_mean": reverse_sum / matched,
        },
    )


def audit_advantages(args, rollout, rollout_id):
    """Pure-OPD policy signal must survive the shared loss path unchanged."""
    if args.kl_coef or args.normalize_advantages:
        return
    errors = []
    for advantage, reverse in zip(rollout["advantages"], rollout["opd_reverse_kl"], strict=True):
        if not bool(torch.isfinite(advantage).all()):
            raise ValueError("Nonfinite OPD advantage")
        errors.append(float((advantage + args.opd_kl_coef * reverse).abs().max()))
    error = max(errors)
    if error > 1e-6:
        raise ValueError(f"OPD advantage differs from the masked teacher signal: {error}")
    contract.record(args, {"event": "opd_advantages", "rollout_id": rollout_id, "max_abs_error": error})
