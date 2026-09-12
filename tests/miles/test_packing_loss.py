"""Real MILES response slicing and gradients across packed document boundaries."""

from types import SimpleNamespace

import pytest
import torch
from miles.backends.training_utils import loss as miles_loss
from miles.backends.training_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo
from torch import distributed as dist

from open_instruct.miles import contract, data, packing


@pytest.mark.parametrize("token_average", [False, True])
@pytest.mark.parametrize("skip_scoring", [False, True])
def test_packing_preserves_rl_objective_and_gradients(tmp_path, token_average, skip_scoring):
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
        args = SimpleNamespace(
            calculate_per_token_loss=token_average,
            global_batch_size=4,
            qkv_format="bshd",
            loss_type="policy_loss",
            advantage_estimator="grpo",
            use_rollout_logprobs=False,
            skip_actor_forward_only=skip_scoring,
            entropy_coef=0.0,
            observe_training_entropy=False,
            use_opsm=False,
            eps_clip=0.2,
            eps_clip_high=0.28,
            eps_clip_c=None,
            use_tis=True,
            tis_clip_low=0.9,
            tis_clip=1.1,
            custom_tis_function_path=None,
            get_mismatch_metrics=False,
            use_kl_loss=True,
            use_unbiased_kl=False,
            kl_loss_type="k3",
            kl_loss_coef=0.03,
            custom_pg_loss_reducer_function_path=None,
            rollout_temperature=1.0,
            true_on_policy_mode=False,
            allgather_cp=False,
            log_probs_chunk_size=32,
            recompute_loss_function=False,
            use_dynamic_global_batch_size=False,
            multi_lora=False,
        )
        lengths, responses = [5, 8, 6, 7], [3, 5, 2, 4]
        raw = {
            "tokens": [torch.arange(n) % 11 for n in lengths],
            "total_lengths": lengths,
            "response_lengths": responses,
            "loss_masks": [torch.ones(n) for n in responses],
        }
        raw["loss_masks"][1][1] = 0
        raw["loss_masks"][2].zero_()  # clamped denominator for a completely masked response
        torch.manual_seed(17)
        initial = [torch.randn(1, n, 11) * 0.3 for n in lengths]
        scores = [
            miles_loss.get_log_probs_and_entropy(
                x, args=args, unconcat_tokens=[t], total_lengths=[n], response_lengths=[resp], max_seq_lens=[n]
            )["log_probs"][0]
            for x, t, n, resp in zip(initial, raw["tokens"], lengths, responses, strict=True)
        ]
        raw.update(
            log_probs=scores,
            rollout_log_probs=[v + torch.linspace(-0.4, 0.4, len(v)) for v in scores],
            ref_log_probs=[v + 0.17 for v in scores],
            advantages=[torch.full((n,), (-1.0 if i % 2 else 1.0)) for i, n in enumerate(responses)],
        )
        results = []
        for packed in [False, True]:
            args.qkv_format = "thd" if packed else "bshd"
            values = [v.clone().requires_grad_() for v in initial]
            samples = data.sample_batches(raw, 16)
            indices = packing.plan(lengths, 16) if packed else [[i] for i in range(4)]
            batches = [packing.combine(samples, ids) for ids in indices] if packed else samples
            normalization = contract.step_normalization(batches, 4)
            objective = 0
            for batch, ids in zip(batches, indices, strict=True):
                if skip_scoring:
                    batch.pop("log_probs")
                logits = torch.cat([values[i] for i in ids], dim=1)
                loss, _, _ = miles_loss.loss_function(
                    args, batch, len(batches), logits, apply_megatron_loss_scaling=False
                )
                if token_average:
                    loss = normalization.scale_token_loss(loss)
                objective = objective + loss
            objective.backward()
            results.append((objective.detach(), [v.grad for v in values]))
        torch.testing.assert_close(results[0], results[1], rtol=1e-6, atol=1e-7)
        for grads in results[1][1]:
            assert grads[0, -1].count_nonzero() == 0, "last token must not predict across a document boundary"
    finally:
        parallel.set_parallel_state(previous)
        dist.destroy_process_group()
