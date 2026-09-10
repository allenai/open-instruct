"""CPU gradient parity with SP shards and DeepSpeed's reduction arithmetic.

The sequence-count collectives and final gradient reduction are simulated;
forward/backward use the real dense and tiled GRPO losses.
"""

import copy
from unittest.mock import patch

import pytest
import torch

from open_instruct import grpo_utils, model_utils


@pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
@pytest.mark.parametrize("sp_size", [1, 2, 4])
@pytest.mark.parametrize("mode", ["token", "sequence"])
@pytest.mark.parametrize("uneven", [False, True])
def test_sharded_gradients_match_unsharded(zero_stage, sp_size, mode, uneven):
    torch.manual_seed(12)
    world_size = 2 * sp_size
    # Independent reference for DeepSpeed's configured reduction behavior.
    reduction_divisor = world_size if zero_stage == 3 else 2
    multiplier = grpo_utils.deepspeed_gradient_reduction_divisor(world_size, sp_size, zero_stage)
    backbone = torch.nn.Linear(3, 4)
    head = torch.nn.Linear(4, 7)
    inputs = torch.randn(2, 2, 8, 3)  # two accumulation steps, two DP samples
    labels = torch.randint(0, 7, (2, 2, 8))
    advantages = torch.randn(2, 2, 8)
    masks = torch.ones(2, 2, 8, dtype=torch.bool)
    if uneven:
        masks[0] = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1, 0, 1]])
        masks[1, 0] = False
        masks[1, 1, 4:] = False
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).expand(2, 2, -1)
    old_logprobs = model_utils.log_softmax_and_gather(head(backbone(inputs)), labels).detach()
    weights = masks.float()
    counts = {}
    if mode == "sequence":
        weights = torch.zeros_like(weights)
        denominator = 0.0
        for step in range(2):
            for dp in range(2):
                counts[step, dp], _ = grpo_utils._sequence_id_counts(masks[step, dp : dp + 1], ids[step, dp : dp + 1])
                weights[step, dp], seq_count = grpo_utils._sequence_loss_weights(
                    masks[step, dp : dp + 1], ids[step, dp : dp + 1]
                )
                denominator += seq_count.item()
    else:
        denominator = masks.sum().item()
    config = grpo_utils.GRPOExperimentConfig.__new__(grpo_utils.GRPOExperimentConfig)
    config.loss_fn = grpo_utils.GRPOLossType.dapo
    config.clip_lower = 0.2
    config.clip_higher = 0.2
    config.load_ref_policy = False
    config.beta = 0.0

    def dense_token_loss(logprobs, old, adv):
        return grpo_utils.compute_grpo_loss(
            new_logprobs=logprobs, ratio=torch.exp(logprobs - old), advantages=adv, ref_logprobs=None, config=config
        )[2]

    logprobs = model_utils.log_softmax_and_gather(head(backbone(inputs)), labels)
    expected = (dense_token_loss(logprobs, old_logprobs, advantages) * weights).sum() / denominator
    expected.backward()
    expected_grads = [p.grad.clone() for model in (backbone, head) for p in model.parameters()]

    for tiled in (False, True):
        summed_grads = [torch.zeros_like(g) for g in expected_grads]
        summed_loss = torch.zeros(())
        for dp in range(2):
            for sp in range(sp_size):
                local_backbone, local_head = copy.deepcopy(backbone), copy.deepcopy(head)
                local_backbone.zero_grad()
                local_head.zero_grad()
                chunk = slice(sp * (8 // sp_size), (sp + 1) * (8 // sp_size))
                for step in range(2):
                    mask = masks[step, dp : dp + 1, chunk]
                    local_ids = ids[step, dp : dp + 1, chunk]
                    local_labels = labels[step, dp : dp + 1, chunk]
                    hidden = local_backbone(inputs[step, dp : dp + 1, chunk])
                    old = old_logprobs[step, dp : dp + 1, chunk]
                    adv = advantages[step, dp : dp + 1, chunk]
                    if tiled:
                        # Each SP rank sees the full sequence counts for its DP sample.
                        with patch.object(
                            grpo_utils,
                            "_sequence_id_counts",
                            return_value=(counts.get((step, dp), torch.zeros(0)), mask),
                        ):
                            scale = grpo_utils.tiled_grpo_loss_scale(mask, denominator, multiplier, mode, local_ids)
                            loss = grpo_utils.tiled_grpo_lm_head_loss(
                                lm_head=local_head,
                                hidden_states=hidden,
                                selected_token_ids=local_labels,
                                response_mask=mask,
                                advantages=adv,
                                old_logprobs=old,
                                ref_logprobs=None,
                                temperature=1.0,
                                beta=0.0,
                                clip_lower=0.2,
                                clip_higher=0.2,
                                shards=2,
                                loss_scale=scale,
                                loss_denominator=mode,
                                rollout_sample_ids=local_ids,
                            )[0]
                    else:
                        local_logprobs = model_utils.log_softmax_and_gather(local_head(hidden), local_labels)
                        loss = (
                            (dense_token_loss(local_logprobs, old, adv) * weights[step, dp : dp + 1, chunk]).sum()
                            / denominator
                            * multiplier
                        )
                    loss.backward()
                    summed_loss += loss.detach()
                for total, param in zip(
                    summed_grads, list(local_backbone.parameters()) + list(local_head.parameters()), strict=True
                ):
                    total += param.grad
        torch.testing.assert_close(summed_loss / reduction_divisor, expected.detach())
        for actual, target in zip(summed_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual / reduction_divisor, target, atol=1e-6, rtol=1e-5)
