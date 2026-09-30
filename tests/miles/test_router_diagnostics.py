"""Analytic and gradient-neutrality checks for live routing diagnostics."""

from types import SimpleNamespace

import pytest
import torch
from miles.backends.core_utils import router_diagnostics
from olmo_core.nn.moe.v2.router import MoERouterConfigV2


@pytest.mark.parametrize("gating", ["softmax", "topk_softmax"])
def test_document_boundaries_mixing_and_fresh_replay_agreement(gating):
    router = MoERouterConfigV2(d_model=4, num_experts=4, top_k=2, gating_function=gating).build()
    logits = torch.tensor([[[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]])
    scores = logits.softmax(-1)
    ids = torch.tensor([[[0, 1], [0, 1], [1, 3], [0, 1]]])
    weights = scores.gather(-1, ids)
    weights /= weights.sum(-1, keepdim=True)
    row = router_diagnostics.observe(router, (weights, ids, None, (scores, logits)), [2, 2])
    assert row["tokens"] == 2 and row["synthetic_tail_tokens"] == 2
    assert row["expert_counts"] == [1, 2, 0, 1]
    assert row["documents"][0]["expert_counts"] == [1, 1, 0, 0]
    assert row["documents"][1]["expert_counts"] == [0, 1, 0, 1]
    assert row["fresh_replay_overlap"] == 0.75
    assert row["fresh_replay_set_agreement"] == 0.5
    assert sum(row["expert_mixing_mass"]) == pytest.approx(2)
    assert row["token_sample"]["positions"] == [0, 2]
    expected_margin = 1.0 if gating == "topk_softmax" else float(scores[0, 0, 1] - scores[0, 0, 2])
    assert row["selection_margin_mean"] == pytest.approx(expected_margin)


def test_observation_preserves_weights_and_gradients():
    torch.manual_seed(72)
    router = MoERouterConfigV2(d_model=4, num_experts=4, top_k=2).build()
    torch.nn.init.normal_(router.weight)
    x = torch.randn(1, 4, 4, requires_grad=True)
    output = router(x, scores_only=False)
    baseline = torch.autograd.grad(output[0].square().sum(), (x, router.weight))
    output = router(x, scores_only=False)
    row = router_diagnostics.observe(router, output, [2, 2])
    observed = torch.autograd.grad(output[0].square().sum(), (x, router.weight))
    for actual, expected in zip(observed, baseline, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert row["fresh_replay_set_agreement"] == 1


def test_rejects_unsupported_selection():
    router = SimpleNamespace(gating_function="sigmoid")
    with pytest.raises(ValueError, match="unbiased softmax"):
        router_diagnostics.observe(router, None, [2])
