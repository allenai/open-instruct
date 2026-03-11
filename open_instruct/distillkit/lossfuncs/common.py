# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import math
from enum import Enum
from typing import Callable

import torch


class MissingProbabilityHandling(str, Enum):
    ZERO = "zero"
    SYMMETRIC_UNIFORM = "symmetric_uniform"


def get_target_logprobs(
    values_in: torch.Tensor,
    log_target: bool,
    distillation_temperature: float,
    target_generation_temperature: float,
    missing: MissingProbabilityHandling,
) -> torch.Tensor:
    temperature_change = not math.isclose(
        distillation_temperature, target_generation_temperature
    )
    if log_target:
        if (
            not temperature_change
            and missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM
        ):
            return values_in
        alpha = target_generation_temperature / distillation_temperature
        max_in, _ = values_in.max(dim=-1, keepdim=True)
        lse_in = torch.logsumexp(values_in - max_in, dim=-1, keepdim=True) + max_in
        target_sum = lse_in.exp()
        leftover = (1.0 - target_sum).clamp(0, 1)

        alpha_values_in = alpha * values_in
        max_alpha, _ = alpha_values_in.max(dim=-1, keepdim=True)
        alpha_lse = (
            torch.logsumexp(alpha_values_in - max_alpha, dim=-1, keepdim=True)
            + max_alpha
        )
        alpha_sum = alpha_lse.exp()

        if missing == MissingProbabilityHandling.SYMMETRIC_UNIFORM:
            leftover_alpha = leftover.pow(alpha)
            final_lse = (alpha_sum + leftover_alpha).log()
        else:
            final_lse = alpha_lse
        return alpha_values_in - final_lse

    logits = (
        values_in * (target_generation_temperature / distillation_temperature)
        if temperature_change
        else values_in
    )
    sparse_max = torch.max(logits, dim=-1, keepdim=True).values
    sparse_lse = (
        torch.logsumexp((logits - sparse_max).to(torch.float32), dim=-1, keepdim=True)
        + sparse_max
    ).to(values_in.dtype)
    return logits - sparse_lse


def get_logprobs(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    eps: float = 1e-8,
    missing: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
    log_target: bool = True,
    distillation_temperature: float = 1.0,
    target_generation_temperature: float = 1.0,
    student_generation_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _vocab_size = logits.shape
    assert target_ids.shape[:-1] == (batch_size, seq_len)
    assert target_values.shape == target_ids.shape
    assert distillation_temperature > eps
    out_dtype = logits.dtype

    if (not log_target) and (missing != MissingProbabilityHandling.ZERO):
        raise ValueError(
            "MissingProbabilityHandling.SYMMETRIC_UNIFORM is only valid for log_target=True"
        )

    if not math.isclose(distillation_temperature, student_generation_temperature):
        logits = logits * (student_generation_temperature / distillation_temperature)
    student_lse = torch.logsumexp(logits.to(torch.float32), dim=-1, keepdim=True).to(
        out_dtype
    )
    sparse_student_logprobs = logits.gather(-1, target_ids) - student_lse
    with torch.no_grad():
        sparse_target_logprobs = get_target_logprobs(
            target_values.to(torch.float32),
            log_target=log_target,
            distillation_temperature=distillation_temperature,
            target_generation_temperature=target_generation_temperature,
            missing=missing,
        ).to(out_dtype)
    return sparse_student_logprobs, sparse_target_logprobs


def accumulate_over_chunks(
    logits: torch.Tensor,
    target_ids: torch.LongTensor,
    target_values: torch.Tensor,
    mask: torch.Tensor | None,
    chunk_length: int | None,
    fn: Callable,
    *args,
    **kwargs,
) -> torch.Tensor:
    seq_len = logits.shape[1]
    if chunk_length is None:
        chunk_length = seq_len

    total = 0.0
    for start_idx in range(0, seq_len, chunk_length):
        cur_mask = (
            mask[:, start_idx : start_idx + chunk_length] if mask is not None else None
        )
        end_idx = min(start_idx + chunk_length, seq_len)
        total += fn(
            logits[:, start_idx:end_idx],
            target_ids[:, start_idx:end_idx],
            target_values[:, start_idx:end_idx],
            cur_mask,
            *args,
            **kwargs,
        )
    return total
