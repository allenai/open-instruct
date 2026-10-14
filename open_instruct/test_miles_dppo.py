"""Parity of the Megatron DPPO loss with Open Instruct's DPPO objective.

``DPPODivergenceType``, ``compute_binary_divergence`` and ``compute_dppo_mask`` below are verbatim
copies of the Open Instruct helpers from the verifier-RL teacher run's overlay (commit c9790acfe,
``open_instruct/grpo_utils.py`` of Beaker dataset 01M1TC99P2C6YYT90J3EMPC90V). They are the parity
reference for ``open_instruct.miles.megatron_grpo.dppo_math``; do not edit them.
"""

import enum

import pytest
import torch

from open_instruct.miles.megatron_grpo import dppo_math


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


TV = dppo_math.Settings("tv", 0.1)


def _tensors(seed, shape=(6, 40)):
    generator = torch.Generator().manual_seed(seed)
    behavior = -torch.rand(shape, generator=generator) * 3.0
    # Spread the policy around the behavior so all four (sign of A, side of r) cases occur on both
    # sides of the TV threshold.
    policy = (behavior + torch.randn(shape, generator=generator) * 0.6).clamp(max=0.0)
    advantages = torch.randn(shape[0], 1, generator=generator).expand(shape).clone()
    advantages[0] = 0.0
    response_mask = torch.rand(shape, generator=generator) > 0.2
    return policy, behavior, advantages, response_mask


@pytest.mark.parametrize("divergence_type", ["tv", "kl"])
@pytest.mark.parametrize("seed", range(5))
def test_mask_matches_open_instruct(divergence_type, seed):
    policy, behavior, advantages, response_mask = _tensors(seed)
    ratio = torch.exp(policy - behavior)
    settings = dppo_math.Settings(divergence_type, 0.1 if divergence_type == "tv" else 0.05)
    mask, divergence = dppo_math.trust_region_mask(policy, behavior, advantages, ratio, response_mask, settings)
    expected_mask, expected_divergence = compute_dppo_mask(
        new_logprobs=policy,
        behavior_logprobs=behavior,
        advantages=advantages,
        ratio=ratio,
        response_mask=response_mask,
        divergence_type=divergence_type,
        divergence_threshold=settings.threshold,
    )
    assert torch.equal(mask, expected_mask)
    torch.testing.assert_close(divergence, expected_divergence, rtol=0, atol=0)
    # The fixture must exercise both kept and dropped response tokens.
    assert 0 < mask[response_mask].sum() < response_mask.sum()


def _open_instruct_token_mean_loss(policy, behavior, advantages, response_mask):
    """``compute_grpo_loss`` (loss_fn=dppo, beta 0, TIS cap 0) with the DPPO mask as ``tis_weights``,
    then ``masked_mean`` over response tokens: Open Instruct's ``loss_denominator=token``."""
    ratio = torch.exp(policy - behavior)
    mask, _ = compute_dppo_mask(policy, behavior, advantages, ratio, response_mask, "tv", 0.1)
    pg_losses = -advantages * ratio * mask
    pg_loss_max = torch.max(pg_losses, pg_losses)
    return (pg_loss_max * response_mask).sum() / response_mask.sum()


@pytest.mark.parametrize("seed", range(5))
def test_token_mean_loss_and_gradient_match_open_instruct(seed):
    policy, behavior, advantages, response_mask = _tensors(seed)
    ours_policy = policy.clone().requires_grad_(True)
    theirs_policy = policy.clone().requires_grad_(True)

    loss_t, _, _, _ = dppo_math.per_token_loss(ours_policy, behavior, advantages, response_mask, TV)
    ours = loss_t.sum() / response_mask.sum()
    theirs = _open_instruct_token_mean_loss(theirs_policy, behavior, advantages, response_mask)
    ours.backward()
    theirs.backward()

    torch.testing.assert_close(ours, theirs)
    torch.testing.assert_close(ours_policy.grad, theirs_policy.grad)


def test_gradient_is_zero_exactly_where_masked_or_off_response():
    policy, behavior, advantages, response_mask = _tensors(3)
    policy = policy.requires_grad_(True)
    loss_t, mask, ratio, _ = dppo_math.per_token_loss(policy, behavior, advantages, response_mask, TV)
    loss_t.sum().backward()
    dropped = (mask == 0) | ~response_mask
    assert torch.all(policy.grad[dropped] == 0)
    # d(-A r)/d log pi = -A r on kept tokens; DPPO adds no clipping.
    kept = ~dropped
    torch.testing.assert_close(policy.grad[kept], (-advantages * ratio.detach())[kept])


def test_mask_only_drops_moves_away_from_the_rollout():
    behavior = torch.log(torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]))
    policy = torch.log(torch.tensor([[0.7, 0.3, 0.7, 0.3, 0.55, 0.45]]))
    advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0, 1.0, -1.0]])
    ratio = torch.exp(policy - behavior)
    mask, _ = dppo_math.trust_region_mask(policy, behavior, advantages, ratio, torch.ones_like(policy).bool(), TV)
    # TV 0.2 > 0.1: drop A>0 with r>1 and A<0 with r<1; keep moves back towards the rollout.
    # TV 0.05 < 0.1: keep both remaining tokens regardless of direction.
    assert mask.tolist() == [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]]


def test_settings_round_trip_and_reject_missing_or_invalid_values():
    assert dppo_math.Settings.from_environment(TV.environment()) == TV
    with pytest.raises(ValueError, match="OI_DPPO_THRESHOLD"):
        dppo_math.Settings.from_environment({"OI_DPPO_DIVERGENCE_TYPE": "tv"})
    with pytest.raises(ValueError, match="divergence type"):
        dppo_math.Settings("js", 0.1)
    with pytest.raises(ValueError, match="threshold"):
        dppo_math.Settings("tv", 0.0)
