"""Entropy-gated forward KL over the teacher's top-k (EOPD, arXiv 2603.07079, Eq. 9-10).

Pure tensor helpers shared by the rollout-side hooks (gate bookkeeping), the Megatron custom
loss and the offline audit. Per response position ``t`` with teacher top-k log-probabilities
``l_t`` over token ids ``S_t``:

* ``q~_t = softmax(l_t)`` is the teacher renormalised over its own top-k,
* the gate is ``1[H(q~_t) > tau]`` (the paper's entropy proxy; an SGLang teacher only returns
  its top-k, see step 5 of docs/algorithms/opd_validation_program.md for how well it tracks the
  exact entropy),
* ``FKL_t = sum_{j in S_t} q~_t(j) (log q~_t(j) - log p_theta,t(j))`` with the student left as its
  full-vocabulary log-probability at those ids.

``student_log_probs_at`` evaluates ``log p_theta,t(j) = logit_j - logsumexp_V(logits_t)`` on a
tensor-parallel vocabulary shard: the row max and the sum of exponentials are reduced across the
shards, and the gathered logits come from the owning shard with zeros elsewhere. Reductions follow
Megatron's vocab-parallel convention (``_VocabParallelCrossEntropy``): the loss is replicated on
every tensor-parallel rank, so the SUM all-reduce is an identity in backward rather than a second
reduction of the (identical) gradients.
"""

import dataclasses
import math
import os

import torch
from torch import distributed

ENV_PREFIX = "OI_OPD_EOPD_"


@dataclasses.dataclass(frozen=True)
class Settings:
    top_k: int = 0
    alpha: float = 1.0
    tau: float = 0.8

    @property
    def enabled(self):
        return self.top_k > 0

    def environment(self):
        return {
            ENV_PREFIX + "TOP_K": str(self.top_k),
            ENV_PREFIX + "ALPHA": repr(float(self.alpha)),
            ENV_PREFIX + "TAU": repr(float(self.tau)),
        }

    @classmethod
    def from_distillation(cls, distillation):
        if not distillation.get("eopd"):
            return cls()
        return cls(
            top_k=int(distillation["eopd_top_k"]),
            alpha=float(distillation["eopd_alpha"]),
            tau=float(distillation["eopd_tau"]),
        )

    @classmethod
    def from_environment(cls, environment=None):
        environment = os.environ if environment is None else environment
        top_k = int(environment.get(ENV_PREFIX + "TOP_K", "0"))
        if top_k <= 0:
            return cls()
        settings = cls(
            top_k=top_k,
            alpha=float(environment.get(ENV_PREFIX + "ALPHA", "1.0")),
            tau=float(environment.get(ENV_PREFIX + "TAU", "0.8")),
        )
        if not (math.isfinite(settings.alpha) and settings.alpha > 0 and math.isfinite(settings.tau)):
            raise ValueError(f"Invalid EOPD settings {settings}")
        return settings


def renormalized_teacher(topk_log_probs):
    """``log q~`` ``[R, k]``: the teacher's top-k log-probabilities renormalised over the k entries."""
    if topk_log_probs.ndim != 2:
        raise ValueError(f"expected teacher top-k log-probs [R, k]; got {tuple(topk_log_probs.shape)}")
    return torch.log_softmax(topk_log_probs.float(), dim=-1)


def proxy_entropy(topk_log_probs):
    """Entropy ``[R]`` of the renormalised top-k teacher distribution (the paper's gate proxy)."""
    log_q = renormalized_teacher(topk_log_probs)
    return -(log_q.exp() * log_q).sum(dim=-1)


def gate(topk_log_probs, tau):
    """Hard gate ``[R]`` (float 0/1): 1 where the proxy entropy exceeds ``tau``."""
    return (proxy_entropy(topk_log_probs) > tau).float()


def topk_mass(topk_log_probs):
    """Total teacher probability ``[R]`` inside the returned top-k."""
    return topk_log_probs.float().exp().sum(dim=-1)


def forward_kl(topk_log_probs, student_log_probs):
    """``FKL_t`` ``[R]``: ``sum_j q~(j) (log q~(j) - log p_theta(j))`` over the teacher's top-k ids."""
    if student_log_probs.shape != topk_log_probs.shape:
        raise ValueError(
            f"student log-probs {tuple(student_log_probs.shape)} must match teacher top-k {tuple(topk_log_probs.shape)}"
        )
    log_q = renormalized_teacher(topk_log_probs)
    return (log_q.exp() * (log_q - student_log_probs.float())).sum(dim=-1)


class _SumAcrossShards(torch.autograd.Function):
    """SUM all-reduce whose backward passes the gradient straight through (Megatron convention)."""

    @staticmethod
    def forward(ctx, value, group):
        value = value.clone()
        distributed.all_reduce(value, op=distributed.ReduceOp.SUM, group=group)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def student_log_probs_at(logits, ids, *, vocab_start=0, group=None, chunk_size=0):
    """Student log-probabilities ``[R, k]`` at token ``ids`` from a vocabulary shard of ``logits``.

    ``logits`` is ``[R, V_local]`` (this rank's contiguous shard starting at ``vocab_start``; the
    whole vocabulary when ``group`` is None or has one rank). Computed in float32 and chunked over
    rows when ``chunk_size > 0``.
    """
    if logits.ndim != 2 or ids.ndim != 2 or ids.shape[0] != logits.shape[0]:
        raise ValueError(f"expected logits [R, V] and ids [R, k]; got {tuple(logits.shape)} and {tuple(ids.shape)}")
    sharded = group is not None and distributed.get_world_size(group) > 1
    if logits.shape[0] == 0:
        return logits.new_zeros(ids.shape, dtype=torch.float32)
    if chunk_size and chunk_size < logits.shape[0]:
        pieces = [
            student_log_probs_at(
                logits[start : start + chunk_size],
                ids[start : start + chunk_size],
                vocab_start=vocab_start,
                group=group,
            )
            for start in range(0, logits.shape[0], chunk_size)
        ]
        return torch.cat(pieces, dim=0)
    logits = logits.float()
    local_size = logits.shape[-1]
    ids = ids.to(device=logits.device, dtype=torch.long)
    if not sharded and ((ids < 0) | (ids >= local_size)).any():
        raise ValueError("teacher top-k ids fall outside the student vocabulary")
    # logsumexp over the full vocabulary; the shift is detached because logsumexp is shift-invariant.
    row_max = logits.detach().max(dim=-1, keepdim=True).values
    if sharded:
        distributed.all_reduce(row_max, op=distributed.ReduceOp.MAX, group=group)
    sum_exp = (logits - row_max).exp().sum(dim=-1, keepdim=True)
    if sharded:
        sum_exp = _SumAcrossShards.apply(sum_exp, group)
    log_normalizer = row_max + sum_exp.log()
    local_ids = ids - vocab_start
    owned = (local_ids >= 0) & (local_ids < local_size)
    gathered = logits.gather(-1, local_ids.clamp(0, local_size - 1)) * owned
    if sharded:
        gathered = _SumAcrossShards.apply(gathered, group)
    return gathered - log_normalizer


def parse_top_entries(entries, top_k):
    """Split SGLang ``input_top_logprobs`` rows (``[logprob, token_id, ...]`` lists) into ids and log-probs.

    Returns ``(ids [R, k] long, log_probs [R, k] float32)`` and rejects rows of the wrong width or
    with nonfinite scores, so a truncated teacher answer fails the rollout instead of the update.
    """
    ids, log_probs = [], []
    for position, row in enumerate(entries):
        if row is None or len(row) != top_k:
            raise ValueError(
                f"Teacher returned {0 if row is None else len(row)} top-k entries at position {position}; expected {top_k}"
            )
        ids.append([int(entry[1]) for entry in row])
        log_probs.append([float(entry[0]) for entry in row])
    ids = torch.tensor(ids, dtype=torch.long).reshape(len(entries), top_k)
    log_probs = torch.tensor(log_probs, dtype=torch.float32).reshape(len(entries), top_k)
    if not torch.isfinite(log_probs).all() or (ids < 0).any():
        raise ValueError("Teacher top-k scores are nonfinite or carry invalid token ids")
    return ids, log_probs


def sample_tensors(metadata, top_k, device=None):
    """Tensorise one sample's stored top-k (``eopd_topk_ids`` / ``eopd_topk_logprobs``) as ``[R, k]``."""
    try:
        ids = torch.tensor(metadata["eopd_topk_ids"], dtype=torch.long, device=device)
        log_probs = torch.tensor(metadata["eopd_topk_logprobs"], dtype=torch.float32, device=device)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Sample metadata lacks the EOPD teacher top-k; enable distillation.eopd in the run"
        ) from error
    if ids.ndim != 2 or ids.shape[-1] != top_k or ids.shape != log_probs.shape:
        raise ValueError(f"EOPD top-k has shape {tuple(ids.shape)} / {tuple(log_probs.shape)}; expected [R, {top_k}]")
    return ids, log_probs
