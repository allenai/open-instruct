"""Guard scoring alignment and prefill capture in the fidelity diagnostic."""

from types import SimpleNamespace

import torch
from scripts.miles import hero_core_fidelity, hero_serving_arithmetic


def test_selected_scores_predict_next_token():
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 2.0, 1.0], [2.0, 5.0, 1.0]]])
    result = hero_core_fidelity.summarize_logits(logits, [2, 0, 1])
    expected = logits.log_softmax(-1)[0, [0, 1], [0, 1]]
    torch.testing.assert_close(result["selected"], expected)
    torch.testing.assert_close(result["last"], logits[0, -1])


def test_prefill_logits_are_not_overwritten_by_decode(tmp_path):
    (tmp_path / "case.txt").write_text("sample")
    hook = hero_core_fidelity.capture_logits_factory({"root": str(tmp_path)})
    first = torch.tensor([[1.0, 2.0]])
    hook(None, (), SimpleNamespace(next_token_logits=first))
    first.add_(10)
    hook(None, (), SimpleNamespace(next_token_logits=torch.zeros(1, 2)))
    torch.testing.assert_close(torch.load(tmp_path / "sample.pt", weights_only=True), torch.tensor([1.0, 2.0]))


def test_expert_combine_retains_fp32_weights_and_delays_final_cast():
    routes = torch.tensor([[[1.5, -3.25], [1.5, -3.25]]], dtype=torch.bfloat16)
    weights = torch.tensor([[0.1234567, 0.7654321]])
    expected = (routes[:, 0].float() * weights.sum(-1, keepdim=True)).bfloat16()
    torch.testing.assert_close(hero_serving_arithmetic.combine_fp32(routes, weights), expected, rtol=0, atol=0)
