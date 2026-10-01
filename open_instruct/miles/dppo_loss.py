"""Miles custom loss: Open Instruct's DPPO objective for verifier RL.

Selected with ``--loss-type custom_loss --custom-loss-function-path
open_instruct.miles.dppo_loss.policy_loss`` and ``--use-rollout-logprobs``. The ratio is taken
against the rollout engine's sampled-token log-probabilities, matching Open Instruct's
``--use_vllm_logprobs true`` with the TIS cap at 0; the ``dppo_math`` mask replaces PPO clipping.
Advantages are the batch's ``advantages`` unchanged (centered verifier rewards from the GRPO
estimator). The per-token loss uses the same ``sum_of_sample_mean`` reducer as upstream, so
``--calculate-per-token-loss`` gives Open Instruct's ``loss_denominator=token`` global token mean.
Settings come from the ``OI_DPPO_*`` environment.
"""

import torch
from miles.backends.training_utils.cp_utils import get_local_response_loss_masks
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy

from open_instruct.miles import dppo_math


def policy_loss(args, batch, logits, sum_of_sample_mean):
    if not args.use_rollout_logprobs or batch.get("rollout_log_probs") is None:
        raise ValueError("DPPO needs --use-rollout-logprobs and rollout_log_probs in the train batch")
    if args.use_tis or args.use_opsm or args.use_kl_loss or args.entropy_coef != 0:
        raise ValueError("DPPO loss does not combine with TIS, OPSM, a KL loss or an entropy bonus")
    settings = dppo_math.Settings.from_environment()
    log_probs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens"),
    )["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    behavior = torch.cat([value.detach() for value in batch["rollout_log_probs"]], dim=0).to(log_probs)
    advantages = torch.cat([value.detach() for value in batch["advantages"]], dim=0).to(log_probs)
    response_mask = torch.cat(
        get_local_response_loss_masks(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
            args.qkv_format,
            batch.get("max_seq_lens"),
        ),
        dim=0,
    ).to(device=log_probs.device)
    response_mask = response_mask.bool()
    behavior = torch.nan_to_num(behavior, nan=0.0, posinf=0.0, neginf=dppo_math.LOG_PROB_FLOOR)
    advantages = torch.where(response_mask, torch.nan_to_num(advantages, nan=0.0), advantages.new_zeros(()))
    loss_t, mask, ratio, divergence = dppo_math.per_token_loss(
        log_probs, behavior, advantages, response_mask, settings
    )
    loss = sum_of_sample_mean(loss_t)
    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()
    active = response_mask.to(log_probs.dtype)
    ppo_kl = torch.where(response_mask, behavior - log_probs.detach(), log_probs.new_zeros(()))
    metrics = {
        "loss": loss.clone().detach(),
        "pg_loss": loss.clone().detach(),
        "ppo_kl": sum_of_sample_mean(ppo_kl).detach(),
        "dppo_kept_frac": sum_of_sample_mean(mask * active).detach(),
        "dppo_masked_frac": sum_of_sample_mean((1.0 - mask) * active).detach(),
        "dppo_divergence": sum_of_sample_mean(divergence).detach(),
        "dppo_ratio": sum_of_sample_mean(ratio.detach() * active).detach(),
        "train_rollout_logprob_abs_diff": sum_of_sample_mean(ppo_kl.abs()).detach(),
    }
    return loss, metrics
