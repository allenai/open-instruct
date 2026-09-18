"""Miles custom loss: upstream sampled-token OPD plus the entropy-gated forward-KL term of EOPD.

Selected with ``--loss-type custom_loss --custom-loss-function-path
open_instruct.miles.eopd_loss.policy_loss``. The PPO/OPD part is upstream ``policy_loss_function``
unchanged (advantages already carry ``kl_coef * (teacher - student)``); the extra term is
``alpha * mean_t g_t FKL_t`` with the same per-sample reducer as the policy loss. The teacher's
top-k ids and log-probs travel in each sample's ``metadata`` dict (set by ``opd_hooks``); the
runtime patch adds ``metadata`` to the Megatron train-step batch keys. Settings come from the
``OI_OPD_EOPD_*`` environment that ``opd_runtime`` forwards to the Ray actors.
"""

import torch
from miles.backends.training_utils.loss_hub import logit_processors, losses
from miles.backends.training_utils.parallel import get_parallel_state
from torch import distributed

from open_instruct.miles import eopd_math


def forward_kl_terms(args, batch, logits, settings):
    """Per-token ``(fkl, gate, proxy_entropy, topk_mass)`` ``[sum R]`` over the micro-batch's responses."""
    metadata = batch.get("metadata")
    if metadata is None:
        raise ValueError("EOPD needs sample metadata in the train batch; the runtime patch must request `metadata`")
    parallel_state = get_parallel_state()
    if parallel_state.cp.size != 1:
        raise ValueError("EOPD supports context-parallel size 1 only")
    tp_group = parallel_state.tp.group
    tp_rank = distributed.get_rank(tp_group) if parallel_state.tp.size > 1 else 0
    chunks = logit_processors._iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=False,
    )
    fkl, gates, entropies, masses = [], [], [], []
    for sample_metadata, (logits_chunk, tokens_chunk, _) in zip(metadata, chunks, strict=True):
        ids, teacher = eopd_math.sample_tensors(sample_metadata, settings.top_k, device=logits.device)
        if ids.shape[0] != tokens_chunk.shape[0]:
            raise ValueError(f"EOPD top-k covers {ids.shape[0]} positions; the response has {tokens_chunk.shape[0]}")
        student = eopd_math.student_log_probs_at(
            logits_chunk,
            ids,
            vocab_start=tp_rank * logits_chunk.shape[-1],
            group=tp_group if parallel_state.tp.size > 1 else None,
            chunk_size=max(int(getattr(args, "log_probs_chunk_size", 0) or 0), 0),
        )
        fkl.append(eopd_math.forward_kl(teacher, student))
        gates.append(eopd_math.gate(teacher, settings.tau))
        entropies.append(eopd_math.proxy_entropy(teacher))
        masses.append(eopd_math.topk_mass(teacher))
    return tuple(torch.cat(values, dim=0) for values in (fkl, gates, entropies, masses))


def policy_loss(args, batch, logits, sum_of_sample_mean):
    loss, metrics = losses.policy_loss_function(args, batch, logits, sum_of_sample_mean)
    settings = eopd_math.Settings.from_environment()
    if not settings.enabled:
        raise ValueError("EOPD loss selected without OI_OPD_EOPD_TOP_K; enable distillation.eopd")
    fkl, gates, entropies, masses = forward_kl_terms(args, batch, logits, settings)
    fkl = torch.nan_to_num(fkl, nan=0.0, posinf=0.0, neginf=0.0)
    fkl_loss = sum_of_sample_mean(gates * fkl)
    loss = loss + settings.alpha * fkl_loss
    metrics = dict(metrics)
    metrics["loss"] = loss.clone().detach()
    metrics["eopd_fkl_loss"] = fkl_loss.clone().detach()
    metrics["eopd_fkl"] = sum_of_sample_mean(fkl).clone().detach()
    metrics["eopd_gate_frac"] = sum_of_sample_mean(gates).clone().detach()
    metrics["eopd_teacher_proxy_entropy"] = sum_of_sample_mean(entropies).clone().detach()
    metrics["eopd_teacher_topk_mass"] = sum_of_sample_mean(masses).clone().detach()
    return loss, metrics
