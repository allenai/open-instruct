"""Exercise score-centering through Core batching and native loss dispatch."""

from types import SimpleNamespace

import pytest
import torch
from miles.backends.core_utils import data, packing
from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss import objective
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


@pytest.mark.parametrize("per_token", [False, True])
def test_packed_core_score_centering_matches_unpacked_gradients(monkeypatch, per_token):
    singleton = GroupInfo(rank=0, size=1, group=None)
    monkeypatch.setattr(
        parallel,
        "_parallel_state",
        ParallelState(
            **{name: singleton for name in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
        ),
    )
    args = SimpleNamespace(
        loss_type="score_centering",
        calculate_per_token_loss=per_token,
        global_batch_size=2,
        qkv_format="bshd",
        use_kl_loss=False,
        use_unbiased_kl=False,
        entropy_coef=0.0,
        observe_training_entropy=False,
        use_sampling_support_replay=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=2,
        allgather_cp=False,
        score_centering_is="tis",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        recompute_loss_function=False,
        use_dynamic_global_batch_size=False,
        multi_lora=False,
    )
    lengths, responses = [5, 7], [2, 3]
    raw = {
        "tokens": [torch.arange(n) % 11 for n in lengths],
        "total_lengths": lengths,
        "response_lengths": responses,
        "loss_masks": [torch.tensor([1, 0]), torch.tensor([1, 1, 1])],
        "advantages": [torch.tensor([0.7, float("nan")]), torch.tensor([-0.4, 0.2, -0.7])],
    }
    generator = torch.Generator().manual_seed(17)
    initial = [torch.randn(1, length, 11, generator=generator) for length in lengths]
    candidates, head, behavior = [], [], []
    for tokens, logits, length, response in zip(raw["tokens"], initial, lengths, responses, strict=True):
        logp = (logits[0, length - response - 1 : length - 1] + torch.linspace(-0.3, 0.3, 11)).log_softmax(-1)
        ids = logp.topk(3, dim=-1).indices
        candidates.append(ids)
        head.append(logp.gather(-1, ids))
        behavior.append(logp.gather(-1, tokens[-response:, None]).squeeze(-1))
    raw.update(rollout_topk_token_ids=candidates, rollout_topk_log_probs=head, rollout_log_probs=behavior)
    results = []
    for packed in (False, True):
        args.qkv_format = "thd" if packed else "bshd"
        values = [value.clone().requires_grad_() for value in initial]
        samples = data.sample_batches(raw, 16)
        batches = [packing.combine(samples, [0, 1])] if packed else samples
        logits = [torch.cat(values, dim=1)] if packed else values
        losses = [
            objective.loss_function(args, batch, len(batches), value)[0]
            for batch, value in zip(batches, logits, strict=True)
        ]
        loss = sum(losses)
        results.append((loss.detach(), torch.autograd.grad(loss, values)))
    torch.testing.assert_close(results[0][0], results[1][0])
    for unpacked, packed in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(unpacked, packed)
        assert torch.isfinite(packed).all() and packed.abs().sum() > 0
