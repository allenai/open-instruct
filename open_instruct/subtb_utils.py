from __future__ import annotations

import math
import random
from collections.abc import Iterable

import torch
import torch.nn.functional as F


def compute_subtb_g_values(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    q: float,
    alpha: float,
    omega: float,
) -> torch.Tensor:
    """Compute GM-filtered per-token g-values from policy logits."""
    tau_policy = q * alpha + omega
    tau_sharp = alpha

    lp_policy = F.log_softmax(tau_policy * logits, dim=-1)
    lp_sharp = F.log_softmax(tau_sharp * logits, dim=-1)

    lp1 = lp_policy.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    lp2 = lp_sharp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (lp1 - q * lp2) / omega


def split_response_positions(response_mask: torch.Tensor, dones: torch.Tensor) -> list[torch.Tensor]:
    """Split packed response positions into per-rollout subsequences."""
    response_positions = torch.nonzero(response_mask, as_tuple=False).flatten()
    if response_positions.numel() == 0:
        return []

    sequences: list[torch.Tensor] = []
    start_idx = 0
    for idx, position in enumerate(response_positions.tolist()):
        if bool(dones[position]):
            sequences.append(response_positions[start_idx : idx + 1])
            start_idx = idx + 1

    if start_idx < response_positions.numel():
        sequences.append(response_positions[start_idx:])
    return [seq for seq in sequences if seq.numel() > 0]


def sample_subtb_windows(
    sequence_length: int,
    num_windows: int,
    min_window_size: int,
    max_window_size: int,
    include_terminal_window: bool = True,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    """Sample `[start, end)` windows with log-uniform lengths."""
    if sequence_length <= 0 or num_windows <= 0:
        return []

    rng = rng or random
    min_len = max(1, min(min_window_size, sequence_length))
    max_len = max(min_len, min(max_window_size, sequence_length))

    def sample_length() -> int:
        if min_len == max_len:
            return min_len
        sampled = math.exp(rng.uniform(math.log(min_len), math.log(max_len)))
        return max(min_len, min(max_len, int(round(sampled))))

    windows: list[tuple[int, int]] = []
    if include_terminal_window:
        terminal_length = sample_length()
        windows.append((sequence_length - terminal_length, sequence_length))

    while len(windows) < num_windows:
        length = sample_length()
        max_start = sequence_length - length
        start = 0 if max_start <= 0 else rng.randint(0, max_start)
        windows.append((start, start + length))
    return windows


def compute_subtb_loss(
    flow_values: torch.Tensor,
    g_values: torch.Tensor,
    response_mask: torch.Tensor,
    dones: torch.Tensor,
    rewards: torch.Tensor,
    reward_scale: float,
    num_windows: int,
    min_window_size: int,
    max_window_size: int,
    lambda_decay: float,
    rng: random.Random | None = None,
    g_sparsity_threshold: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute SubTB loss and diagnostics for one packed sample."""
    sequences = split_response_positions(response_mask, dones)
    if not sequences:
        zero = flow_values.sum() * 0.0
        return zero, {
            "num_windows": 0.0,
            "mean_abs_residual": 0.0,
            "flow_variance": 0.0,
            "g_sparsity": 0.0,
            "mean_window_length": 0.0,
        }

    losses: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    flow_variances: list[float] = []
    valid_g_values: list[torch.Tensor] = []
    sampled_window_lengths: list[int] = []

    for positions in sequences:
        seq_flow = flow_values.index_select(0, positions)
        seq_g = g_values.index_select(0, positions)
        valid_g_values.append(seq_g)
        flow_variances.append(float(seq_flow.var(unbiased=False).detach().cpu()) if seq_flow.numel() > 1 else 0.0)

        terminal_reward = rewards[positions[-1]]
        windows = sample_subtb_windows(
            sequence_length=positions.numel(),
            num_windows=num_windows,
            min_window_size=min_window_size,
            max_window_size=max_window_size,
            include_terminal_window=True,
            rng=rng,
        )
        for start, end in windows:
            left = seq_flow[start]
            right = seq_flow[end] if end < seq_flow.numel() else reward_scale * terminal_reward
            g_sum = seq_g[start:end].sum()
            residual = left + g_sum - right
            weight = lambda_decay ** (end - start)
            residuals.append(residual)
            sampled_window_lengths.append(end - start)
            losses.append(weight * residual.square())

    stacked_losses = torch.stack(losses)
    stacked_residuals = torch.stack(residuals)
    all_g = torch.cat(valid_g_values) if valid_g_values else flow_values.new_zeros(0)

    return stacked_losses.mean(), {
        "num_windows": float(len(losses)),
        "mean_abs_residual": float(stacked_residuals.abs().mean().detach().cpu()),
        "flow_variance": sum(flow_variances) / max(len(flow_variances), 1),
        "g_sparsity": (
            float((all_g.abs() > g_sparsity_threshold).float().mean().detach().cpu()) if all_g.numel() > 0 else 0.0
        ),
        "mean_window_length": (
            sum(sampled_window_lengths) / max(len(sampled_window_lengths), 1) if sampled_window_lengths else 0.0
        ),
    }
