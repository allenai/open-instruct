"""DPPO trust-region mask (arXiv 2602.04879, Eqs. 11-14), ported from Open Instruct ``grpo_utils``.

Pure tensor helpers for the Miles custom loss in ``dppo_loss``. They reproduce the Open Instruct
verifier-RL teacher run (``--loss_fn dppo --dppo_divergence_type tv --dppo_divergence_threshold
0.1 --use_vllm_logprobs true``, overlay commit ``c9790acfe``): with behavior log-probability
``log mu`` from the rollout engine and trainer log-probability ``log pi``,

* ``r_t = pi / mu``,
* the binary divergence collapses each position to a Bernoulli over {sampled token, the rest}:
  TV is ``|mu - pi|``; KL is the Bernoulli KL(mu || pi),
* ``M_t`` drops a token only when its update moves further from the rollout and it is already
  outside the region: ``(A > 0 and r > 1 or A < 0 and r < 1) and D_t > delta``,
* the per-token loss is ``-M_t * A_t * r_t`` with no symmetric clipping.

The mask is computed without gradient. Settings reach the Ray training actors as ``OI_DPPO_*``
environment, the same way ``eopd_math`` forwards EOPD settings.
"""

import torch

from open_instruct.miles.megatron_grpo import dppo_settings

# Re-exported so the loss and its tests name one module; the run-file contract imports
# dppo_settings directly to stay free of torch.
ENV_PREFIX = dppo_settings.ENV_PREFIX
DIVERGENCE_TYPES = dppo_settings.DIVERGENCE_TYPES
Settings = dppo_settings.Settings
LOG_PROB_FLOOR = -30.0


def binary_divergence(behavior_log_probs, policy_log_probs, response_mask, divergence_type):
    """Per-token Bernoulli divergence between behavior ``mu`` and policy ``pi``; zero off the mask."""
    eps = 1e-9
    # Real log-probabilities are <= 0; clamp so exp() stays in [e^-30, 1] as in Open Instruct.
    mu = torch.exp(behavior_log_probs.clamp(min=LOG_PROB_FLOOR, max=0.0))
    pi = torch.exp(policy_log_probs.clamp(min=LOG_PROB_FLOOR, max=0.0))
    if divergence_type == "tv":
        divergence = (mu - pi).abs()
    elif divergence_type == "kl":
        mu = mu.clamp(eps, 1.0 - eps)
        pi = pi.clamp(eps, 1.0 - eps)
        divergence = mu * (mu.log() - pi.log()) + (1.0 - mu) * ((1.0 - mu).log() - (1.0 - pi).log())
    else:
        raise ValueError(f"DPPO divergence type must be one of {DIVERGENCE_TYPES}, got {divergence_type!r}")
    return torch.where(response_mask, divergence, torch.zeros_like(divergence))


def trust_region_mask(policy_log_probs, behavior_log_probs, advantages, ratio, response_mask, settings):
    """``(mask, divergence)``: the 0/1 DPPO mask ``M_t`` (Eq. 12) and the divergence for logging."""
    with torch.no_grad():
        divergence = binary_divergence(behavior_log_probs, policy_log_probs, response_mask, settings.divergence_type)
        outside = divergence > settings.threshold
        bad = ((advantages > 0) & (ratio > 1.0) & outside) | ((advantages < 0) & (ratio < 1.0) & outside)
        mask = (~bad & response_mask).to(policy_log_probs.dtype)
    return mask, divergence


def per_token_loss(policy_log_probs, behavior_log_probs, advantages, response_mask, settings):
    """``(loss, mask, ratio, divergence)`` with ``loss_t = -M_t A_t r_t``; zero off the response mask."""
    response_mask = response_mask.bool()
    ratio = torch.exp(policy_log_probs - behavior_log_probs)
    mask, divergence = trust_region_mask(
        policy_log_probs, behavior_log_probs, advantages, ratio.detach(), response_mask, settings
    )
    loss = -advantages * ratio * mask
    return torch.where(response_mask, loss, torch.zeros_like(loss)), mask, ratio, divergence
