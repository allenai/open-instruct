"""Skipping scoring preserves real MILES TIS/KL losses and gradients on fixed logits."""

import pytest
import torch
from miles.backends.training_utils import loss as miles_loss
from miles.backends.training_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo
from test_contract import batch_for, fixture_data, loss_args
from torch import distributed as dist


@pytest.mark.parametrize("rollout_anchor", [False, True])
@pytest.mark.parametrize("token_average", [False, True])
@pytest.mark.parametrize("reference_kl", [False, True])
def test_skip_preserves_tis_and_reference_kl(tmp_path, rollout_anchor, token_average, reference_kl):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
    previous = parallel._parallel_state
    try:
        group = GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=group,
                intra_dp_cp=group,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        initial, samples = fixture_data()
        args = loss_args(token_average)
        args.use_rollout_logprobs = rollout_anchor
        args.use_tis = True
        args.tis_clip_low, args.tis_clip = 0.9, 1.1
        args.custom_tis_function_path = None
        args.use_kl_loss = reference_kl
        args.use_unbiased_kl = False
        args.kl_loss_type, args.kl_loss_coef = "k3", 0.03
        results = []
        for skipped in (False, True):
            batch = batch_for(samples, list(range(4)))
            args.skip_actor_forward_only = skipped
            logits = initial.clone().requires_grad_()
            if skipped:
                del batch["log_probs"]
            else:
                with torch.no_grad():
                    batch["log_probs"] = miles_loss.get_log_probs_and_entropy(
                        logits,
                        args=args,
                        unconcat_tokens=batch["unconcat_tokens"],
                        total_lengths=batch["total_lengths"],
                        response_lengths=batch["response_lengths"],
                        max_seq_lens=batch["max_seq_lens"],
                    )["log_probs"]
            loss, _, metrics = miles_loss.loss_function(args, batch, 1, logits, apply_megatron_loss_scaling=False)
            loss.backward()
            metrics = dict(zip(metrics["keys"], metrics["values"][1:], strict=True))
            assert metrics["tis_clipfrac"] > 0, "fixture must actually clip importance weights"
            results.append((loss.detach(), logits.grad, metrics))
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    finally:
        parallel.set_parallel_state(previous)
        dist.destroy_process_group()
