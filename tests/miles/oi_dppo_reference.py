"""Verbatim Open Instruct DPPO helpers from the verifier-RL teacher run's overlay (commit c9790acfe).

Copied from ``open_instruct/grpo_utils.py`` of Beaker dataset 01M1TC99P2C6YYT90J3EMPC90V, which
experiment 01M1TMAYJ1RCMSKCJFS5ZBVSX7 copied over its image. Only used as the parity reference for
``open_instruct.miles.dppo_math``; do not edit.
"""

import enum

import torch


class DPPODivergenceType(enum.StrEnum):
    tv = "tv"
    kl = "kl"


def compute_binary_divergence(
    behavior_logprobs: torch.Tensor, policy_logprobs: torch.Tensor, response_mask: torch.Tensor, divergence_type: str
) -> torch.Tensor:
    """Per-token binary (Bernoulli) divergence between behavior and policy.

    Implements the binary approximation from Eqs. 13/14 of the DPPO paper
    (https://arxiv.org/abs/2602.04879): collapse the categorical distribution
    over the vocabulary into a Bernoulli over ``{sampled_token, all_others}``
    using only the per-token logprobs. This is a memory-cheap lower bound on
    the true policy divergence that requires no extra forward passes.

    Args:
        behavior_logprobs: log μ(a_t|s_t), the rollout (vLLM) policy.
        policy_logprobs:   log π(a_t|s_t), the current trainer policy.
        response_mask:     bool mask selecting valid response positions.
        divergence_type:   ``"tv"`` for total variation or ``"kl"`` for KL.

    Returns:
        Float tensor of the same shape as ``policy_logprobs``; entries outside
        ``response_mask`` are zeroed.
    """
    eps = 1e-9
    # Real logprobs are <= 0; non-response sentinel positions can be > 0
    # (see ``mask_logprobs`` / INVALID_LOGPROB). Clamp so exp() stays in [eps, 1].
    mu = torch.exp(behavior_logprobs.clamp(min=-30.0, max=0.0))
    pi = torch.exp(policy_logprobs.clamp(min=-30.0, max=0.0))
    if divergence_type == DPPODivergenceType.tv:
        divergence = (mu - pi).abs()
    elif divergence_type == DPPODivergenceType.kl:
        mu_clip = mu.clamp(eps, 1.0 - eps)
        pi_clip = pi.clamp(eps, 1.0 - eps)
        divergence = mu_clip * (mu_clip.log() - pi_clip.log()) + (1.0 - mu_clip) * (
            (1.0 - mu_clip).log() - (1.0 - pi_clip).log()
        )
    else:
        raise ValueError(
            f"Unknown DPPO divergence type: {divergence_type}. Expected one of {list(DPPODivergenceType)}."
        )
    return torch.where(response_mask, divergence, torch.zeros_like(divergence))


def compute_dppo_mask(
    new_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    response_mask: torch.Tensor,
    divergence_type: str,
    divergence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the DPPO trust-region mask M_t (Eq. 12).

    The mask zeros out updates that would both push the policy further away from
    the rollout (``r_t > 1`` for positive advantage, or ``r_t < 1`` for negative
    advantage) AND have already exceeded the divergence threshold ``δ``.
    Updates that move the ratio back towards 1 are never masked, preserving
    PPO's beneficial asymmetric structure.

    Args:
        new_logprobs:        log π_θ at the sampled tokens.
        behavior_logprobs:   log μ_θ' at the sampled tokens (from the rollout).
        advantages:          per-token advantages, same shape.
        ratio:               π_θ(y_t|s_t) / μ_θ'(y_t|s_t).
        response_mask:       bool mask selecting valid response positions.
        divergence_type:     ``"tv"`` or ``"kl"`` (passed to
            :func:`compute_binary_divergence`).
        divergence_threshold: scalar trust-region radius δ.

    Returns:
        ``(mask, divergence)`` where ``mask`` is a 0/1 float tensor and
        ``divergence`` is the per-token binary divergence (for logging).
    """
    with torch.no_grad():
        divergence = compute_binary_divergence(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=new_logprobs,
            response_mask=response_mask,
            divergence_type=divergence_type,
        )
        outside_region = divergence > divergence_threshold
        bad_high = (advantages > 0) & (ratio > 1.0) & outside_region
        bad_low = (advantages < 0) & (ratio < 1.0) & outside_region
        bad = bad_high | bad_low
        mask = (~bad & response_mask).to(new_logprobs.dtype)
    return mask, divergence
