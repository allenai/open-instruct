"""Explicit forced-action guidance; sampled-token PPO keeps honest behavior scores."""

import contextlib
from pathlib import Path

import torch
from miles.backends.training_utils.loss_hub import losses


def probe_rows(batch, tensor, context_parallel_size=1):
    """Map parent response cuts to rows, rejecting unsupported batch layouts."""
    if context_parallel_size != 1:
        raise ValueError("Stopping guidance/readiness capture require context_parallel_size=1")
    lengths = batch["total_lengths"]
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != sum(lengths):
        raise ValueError("Stopping guidance/readiness capture require an unpadded [1, tokens, features] layout")
    if len(lengths) > 1:
        docs = batch.get("doc_lens")
        if docs is None or docs.reshape(-1).tolist() != lengths:
            raise ValueError("Packed stopping states require matching document boundaries")
    offset = 0
    for sample_index, (total, response, metadata) in enumerate(
        zip(lengths, batch["response_lengths"], batch["metadata"], strict=True)
    ):
        for info in metadata["stopping_probes"]:
            cut = info["cut"]
            if not 0 < cut <= response or total <= response:
                raise ValueError("Stopping probe cut must lie within the parent response")
            yield sample_index, offset + total - response + cut - 1, info
        offset += total


@contextlib.contextmanager
def capture_readiness(model, batch, records, context_parallel_size=1):
    """Capture natural-parent states before intervention, with forced-answer labels."""
    if records is None:
        yield
        return
    if context_parallel_size != 1:
        raise ValueError("Readiness capture requires context_parallel_size=1")
    heads = [module for name, module in model.named_modules() if name.endswith("lm_head")]
    if len(heads) != 1:
        raise ValueError("Readiness capture requires one identifiable lm_head")

    def capture(module, inputs):
        hidden = inputs[0]
        for _, row, info in probe_rows(batch, hidden, context_parallel_size):
            records.append({**info, "state": hidden[0, row].detach().float().cpu()})

    handle = heads[0].register_forward_pre_hook(capture)
    try:
        yield
    finally:
        handle.remove()


def save_readiness(root, step, rank, records):
    if records:
        folder = Path(root) / "readiness-probes"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"policy_step": step, "layer": "lm_head_input", "records": records}, folder / f"step{step}-rank{rank}.pt"
        )


def closing_objective(current, anchor, advantage, clip_low, clip_high):
    """Clipped sequence-event surrogate, not a behavior-policy importance ratio."""
    ratio = (current.sum() - anchor.detach().sum()).exp()
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    return -torch.minimum(ratio * advantage, clipped * advantage)


def auxiliary_batches(batches, pad_to=None):
    """One teacher-forced prefix per cut, independent of answer trial count."""
    result = []
    for batch in batches:
        for tokens, total, response, metadata in zip(
            batch["unconcat_tokens"], batch["total_lengths"], batch["response_lengths"], batch["metadata"], strict=True
        ):
            probes = metadata["stopping_probes"]
            for info in probes:
                cut = info["cut"]
                if not 0 < cut <= response or not info["close_ids"]:
                    raise ValueError("Invalid full-tag stopping cut")
                prefix = tokens[: total - response + cut]
                close = tokens.new_tensor(info["close_ids"])
                sequence = torch.cat([prefix, close])
                result.append(
                    {
                        "tokens": sequence.unsqueeze(0),
                        "total_lengths": [len(sequence)],
                        "stopping_context": {"info": info, "count": len(probes), "prefix_length": len(prefix)},
                    }
                )
    if pad_to is not None:
        if pad_to < len(result):
            raise ValueError("Auxiliary schedule cannot discard cuts")
        # All EP ranks must issue equally many forwards/backwards. Zero-loss
        # placeholders preserve the schedule without duplicating real guidance.
        for _ in range(pad_to - len(result)):
            tokens = batches[0]["unconcat_tokens"][0][:2]
            result.append(
                {"tokens": tokens.unsqueeze(0), "total_lengths": [len(tokens)], "stopping_context": {"dummy": True}}
            )
    return result


def closing_scores(batch, logits):
    context = batch["stopping_context"]
    if logits.ndim != 3 or logits.shape[:2] != batch["tokens"].shape:
        raise ValueError("Full-tag guidance requires an unpadded [1, tokens, vocab] layout")
    if context.get("dummy"):
        return logits[0, :1, 0].float() * 0
    start = context["prefix_length"] - 1
    ids = logits.new_tensor(context["info"]["close_ids"], dtype=torch.long)
    selected = logits[0, start : start + len(ids)].float()
    scores = selected.gather(1, ids[:, None]).squeeze(1) - selected.logsumexp(dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite full-tag stopping scores")
    return scores


def anchor_closing_scores(batch, logits):
    scores = closing_scores(batch, logits).detach()
    batch["stopping_anchor"] = scores
    if not batch["stopping_context"].get("dummy"):
        batch["stopping_context"]["info"]["anchor_logprob"] = float(scores.sum())


METRICS = (
    "stopping_guidance",
    "stopping_positive_probability",
    "stopping_negative_probability",
    "stopping_positive_count",
    "stopping_negative_count",
    "stopping_anchor_abs_diff",
)


def auxiliary_loss(args, batch, logits, template, world_size):
    context = batch["stopping_context"]
    scores = closing_scores(batch, logits)
    metrics = {key: value.new_zeros(()) for key, value in template.items()}
    if context.get("dummy"):
        loss = scores.sum() * 0
    else:
        info = context["info"]
        weight = info.get("parent_weight", 1.0) / context["count"]
        guidance = closing_objective(
            scores, batch["stopping_anchor"], info["advantage"], args.eps_clip, args.eps_clip_high
        )
        loss = args.olmo_core.forced_exit_coefficient * guidance * weight * world_size / args.global_batch_size
        metrics["stopping_guidance"] = guidance.detach() * weight
        sign = "positive" if info["advantage"] > 0 else "negative" if info["advantage"] < 0 else None
        if sign:
            metrics[f"stopping_{sign}_probability"] = scores.detach().sum().exp() * weight
            metrics[f"stopping_{sign}_count"] = scores.new_tensor(weight)
        metrics["stopping_anchor_abs_diff"] = (scores.detach() - batch["stopping_anchor"]).abs().mean() * weight
    metrics["normalized_policy_objective"] = loss.detach()
    # Zero native sample count: guidance never changes the GRPO denominator.
    return loss, metrics


def policy_loss(args, batch, logits, sum_of_sample_mean):
    """Natural samples use unmodified native GRPO; auxiliary contexts train separately."""
    return losses.policy_loss_function(args, batch, logits, sum_of_sample_mean)
