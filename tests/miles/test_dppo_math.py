"""Parity of the Miles DPPO port with the Open Instruct teacher run's DPPO objective."""

import oi_dppo_reference
import pytest
import torch

from open_instruct.miles import dppo_math

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
    expected_mask, expected_divergence = oi_dppo_reference.compute_dppo_mask(
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
    mask, _ = oi_dppo_reference.compute_dppo_mask(policy, behavior, advantages, ratio, response_mask, "tv", 0.1)
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
