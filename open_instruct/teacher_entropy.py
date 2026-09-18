"""Token-level teacher-entropy statistics for on-policy distillation.

EOPD (arXiv 2603.07079) gates the extra forward-KL term on the teacher's per-token entropy and
approximates that entropy from the renormalized top-k teacher distribution (k = 16). These
helpers compute, for every response token of a student rollout scored by the teacher:

* the exact teacher entropy over the full vocabulary,
* the top-k probability mass and the entropy of the renormalized top-k distribution (the
  paper's proxy),
* the teacher log-probability and rank of the token the student actually sampled, and
  whether it lies inside the teacher's top-k.

``summarize`` turns those per-token tensors into the quantities the validation program
records (paper Fig. 3 / Fig. 9): the entropy histogram, the fraction of tokens above the
gate threshold ``tau``, the mean top-k mass, the fraction of student tokens outside the
teacher's top-k, and how well the proxy reproduces the exact gate decision.
"""

import math

import torch

from open_instruct import model_utils

STAT_KEYS = ("entropy", "topk_mass", "proxy_entropy", "sampled_logprob", "sampled_rank", "sampled_in_topk")
DEFAULT_BINS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, math.inf)


def token_statistics(logits: torch.Tensor, sampled: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Per-token statistics from teacher ``logits`` ``[N, V]`` at the ``N`` positions where the
    student sampled ``sampled`` ``[N]``. Computed in float32; ``sampled_rank`` is 0-based."""
    if logits.ndim != 2 or sampled.shape != (logits.shape[0],):
        raise ValueError(
            f"expected logits [N, V] and sampled [N]; got {tuple(logits.shape)} and {tuple(sampled.shape)}"
        )
    if not 1 <= k <= logits.shape[1]:
        raise ValueError(f"k must be in [1, vocab={logits.shape[1]}], got {k}")
    logits = logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)
    entropy = model_utils.entropy_from_logits(logits)
    top_logprobs, top_indices = logprobs.topk(k, dim=-1)
    topk_mass = top_logprobs.exp().sum(dim=-1)
    renormalized = torch.log_softmax(top_logprobs, dim=-1)
    proxy_entropy = -(renormalized.exp() * renormalized).sum(dim=-1)
    sampled = sampled.to(logits.device).long()
    sampled_logprob = logprobs.gather(-1, sampled[:, None]).squeeze(-1)
    sampled_rank = (logprobs > sampled_logprob[:, None]).sum(dim=-1)
    sampled_in_topk = (top_indices == sampled[:, None]).any(dim=-1)
    return {
        "entropy": entropy,
        "topk_mass": topk_mass,
        "proxy_entropy": proxy_entropy,
        "sampled_logprob": sampled_logprob,
        "sampled_rank": sampled_rank,
        "sampled_in_topk": sampled_in_topk,
    }


def concatenate(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate per-chunk ``token_statistics`` results (moved to CPU)."""
    if not chunks:
        return {key: torch.empty(0) for key in STAT_KEYS}
    return {key: torch.cat([chunk[key].detach().cpu() for chunk in chunks]) for key in STAT_KEYS}


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return math.nan
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a * b).sum() / denominator) if denominator > 0 else math.nan


def summarize(stats: dict[str, torch.Tensor], tau: float, k: int, bins: tuple[float, ...] = DEFAULT_BINS) -> dict:
    """Aggregate per-token statistics. ``bins`` are ascending histogram edges over the exact entropy."""
    entropy = stats["entropy"].float()
    proxy = stats["proxy_entropy"].float()
    n = int(entropy.numel())
    if n == 0:
        return {"tokens": 0, "tau": tau, "k": k}
    high = entropy > tau
    proxy_high = proxy > tau
    outside = ~stats["sampled_in_topk"].bool()
    edges = torch.tensor(bins, dtype=torch.float32)
    counts = torch.bucketize(entropy, edges[1:], right=False)
    histogram = [
        {"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int((counts == i).sum())}
        for i in range(len(edges) - 1)
    ]
    return {
        "tokens": n,
        "tau": tau,
        "k": k,
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(entropy.median()),
        "frac_entropy_gt_tau": float(high.float().mean()),
        "entropy_histogram": histogram,
        "topk_mass_mean": float(stats["topk_mass"].float().mean()),
        "topk_mass_mean_high_entropy": float(stats["topk_mass"].float()[high].mean()) if high.any() else math.nan,
        "frac_sampled_outside_topk": float(outside.float().mean()),
        "frac_sampled_outside_topk_high_entropy": float(outside[high].float().mean()) if high.any() else math.nan,
        "frac_sampled_outside_topk_low_entropy": float(outside[~high].float().mean()) if (~high).any() else math.nan,
        "sampled_logprob_mean": float(stats["sampled_logprob"].float().mean()),
        "sampled_rank_median": float(stats["sampled_rank"].float().median()),
        "proxy_entropy_mean": float(proxy.mean()),
        "proxy_abs_error_mean": float((proxy - entropy).abs().mean()),
        "proxy_pearson": _pearson(proxy, entropy),
        "proxy_gate_agreement": float((proxy_high == high).float().mean()),
        "proxy_gate_false_negative": float((high & ~proxy_high).float().sum() / max(int(high.sum()), 1)),
    }
