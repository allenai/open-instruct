import math

import pytest
import torch

from open_instruct import teacher_entropy


def _peaked_logits(n: int = 4, vocab: int = 50, peak: float = 20.0) -> torch.Tensor:
    logits = torch.zeros(n, vocab)
    logits[torch.arange(n), torch.arange(n)] = peak
    return logits


class TestTokenStatistics:
    def test_uniform_teacher(self):
        vocab, k = 64, 16
        logits = torch.zeros(3, vocab)
        sampled = torch.tensor([0, 17, 63])
        stats = teacher_entropy.token_statistics(logits, sampled, k)
        assert torch.allclose(stats["entropy"], torch.full((3,), math.log(vocab)), atol=1e-5)
        assert torch.allclose(stats["topk_mass"], torch.full((3,), k / vocab), atol=1e-6)
        assert torch.allclose(stats["proxy_entropy"], torch.full((3,), math.log(k)), atol=1e-5)
        assert torch.allclose(stats["sampled_logprob"], torch.full((3,), -math.log(vocab)), atol=1e-5)
        assert (stats["sampled_rank"] == 0).all()  # ties: nothing strictly greater

    def test_peaked_teacher_and_topk_membership(self):
        logits = _peaked_logits()
        sampled = torch.tensor([0, 1, 30, 31])  # first two inside the peak, last two far outside
        stats = teacher_entropy.token_statistics(logits, sampled, k=2)
        assert (stats["entropy"] < 1e-3).all()
        assert (stats["topk_mass"] > 0.999).all()
        assert (stats["proxy_entropy"] < 1e-3).all()
        assert stats["sampled_in_topk"].tolist() == [True, True, False, False]
        assert stats["sampled_rank"][0] == 0
        assert stats["sampled_rank"][2] >= 1
        assert stats["sampled_logprob"][2] < stats["sampled_logprob"][0]

    def test_proxy_matches_exact_entropy_when_mass_is_in_topk(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 100) * 0.1
        logits[:, :16] += 12.0  # essentially all mass inside the top 16
        stats = teacher_entropy.token_statistics(logits, torch.zeros(8, dtype=torch.long), k=16)
        assert torch.allclose(stats["proxy_entropy"], stats["entropy"], atol=1e-3)

    def test_shape_and_k_validation(self):
        with pytest.raises(ValueError):
            teacher_entropy.token_statistics(torch.zeros(2, 10), torch.zeros(3, dtype=torch.long), 4)
        with pytest.raises(ValueError):
            teacher_entropy.token_statistics(torch.zeros(2, 10), torch.zeros(2, dtype=torch.long), 11)


class TestSummarize:
    def test_summary_fields(self):
        torch.manual_seed(1)
        chunks = []
        for _ in range(3):
            logits = torch.randn(32, 200) * 3
            sampled = torch.randint(0, 200, (32,))
            chunks.append(teacher_entropy.token_statistics(logits, sampled, k=16))
        stats = teacher_entropy.concatenate(chunks)
        summary = teacher_entropy.summarize(stats, tau=0.8, k=16)
        assert summary["tokens"] == 96
        assert sum(b["count"] for b in summary["entropy_histogram"]) == 96
        assert 0.0 <= summary["frac_entropy_gt_tau"] <= 1.0
        assert 0.0 <= summary["frac_sampled_outside_topk"] <= 1.0
        assert 0.0 <= summary["proxy_gate_agreement"] <= 1.0
        assert summary["proxy_abs_error_mean"] >= 0.0
        assert summary["proxy_entropy_mean"] <= summary["entropy_mean"] + 1e-6  # top-k renormalization drops mass

    def test_empty(self):
        summary = teacher_entropy.summarize(teacher_entropy.concatenate([]), tau=0.8, k=16)
        assert summary["tokens"] == 0
