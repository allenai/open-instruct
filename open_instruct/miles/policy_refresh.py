"""Token provenance for responses continued across weight publications.

Behavior probabilities stay in rollout_log_probs. These spans describe draws,
while replay_version identifies the forward that rebuilt the final route table.
"""

import math

import torch


def version_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Policy refresh requires nonnegative integer policy versions")
    if isinstance(value, str) and (not value.isascii() or not value.isdigit()):
        raise ValueError("Policy refresh received an invalid policy version")
    result = int(value)
    if result < 0:
        raise ValueError("Policy refresh received a negative policy version")
    return result


def validate_spans(spans, length, *, replay_version):
    """Validate exact half-open response-token coverage; omit empty spans."""
    if type(length) is not int or length <= 0 or not isinstance(spans, list) or not spans:
        raise ValueError("Policy refresh requires version spans for every response token")
    replay_version = version_number(replay_version)
    result, cursor, previous = [], 0, -1
    for span in spans:
        if not isinstance(span, dict) or set(span) != {"version", "start", "end"}:
            raise ValueError("Policy spans require version, start, and end")
        start, end = span["start"], span["end"]
        version = version_number(span["version"])
        if type(start) is not int or type(end) is not int or start != cursor or not start <= end <= length:
            raise ValueError("Policy spans must cover response tokens without gaps or overlaps")
        if version < previous or version > replay_version:
            raise ValueError("Policy span versions must increase and cannot exceed the replay version")
        previous, cursor = version, end
        if start != end:
            if result and result[-1]["version"] == version:
                result[-1]["end"] = end
            else:
                result.append(dict(version=version, start=start, end=end))
    if cursor != length or not result or result[-1]["version"] != replay_version:
        raise ValueError("Policy spans must end at the completed response and its final replay version")
    return result


def record_response(sample, meta):
    """Keep original probabilities and latest routes, adding transportable provenance."""
    replay_version = version_number(meta.get("weight_version"))
    spans = validate_spans(meta.get("weight_versions"), sample.response_length, replay_version=replay_version)
    scores = sample.rollout_log_probs
    if scores is None or len(scores) != sample.response_length or not all(math.isfinite(x) for x in scores):
        raise ValueError("Policy refresh requires one finite original rollout logprob per response token")
    records = meta.get("output_token_logprobs")
    if records is None or len(records) != sample.response_length:
        raise ValueError("Policy refresh response is missing token/logprob records")
    if [x[1] for x in records] != sample.tokens[-sample.response_length :]:
        raise ValueError("Policy refresh behavior logprobs are not aligned with response tokens")
    provenance = {"spans": spans, "replay_version": replay_version}
    sample.weight_versions = [str(span["version"]) for span in spans]
    sample.train_metadata = {**(sample.train_metadata or {}), "policy_refresh": provenance}
    sample.metadata = {**(sample.metadata or {}), "policy_refresh": provenance}
    return provenance


def validate_batch(batch):
    """Verify provenance survived sample conversion, partitioning and packing."""
    metadata = batch.get("metadata")
    lengths = batch["response_lengths"]
    versions = batch.get("weight_versions")
    if metadata is None or versions is None or len(metadata) != len(lengths) or len(versions) != len(lengths):
        raise ValueError("Policy refresh training batch lost token-version metadata")
    result = []
    for item, length, sample_versions in zip(metadata, lengths, versions, strict=True):
        record = item.get("policy_refresh") if isinstance(item, dict) else None
        if not isinstance(record, dict):
            raise ValueError("Policy refresh training sample has no provenance")
        spans = validate_spans(record.get("spans"), length, replay_version=record.get("replay_version"))
        if [version_number(v) for v in sample_versions] != [s["version"] for s in spans]:
            raise ValueError("Policy refresh span versions disagree with the staleness ledger")
        result.append(spans)
    return result


def score_metrics(rollout, *, current_version, clip_low, clip_high):
    """Distinguish historical-prefix drift from scores of newly sampled suffixes.

    These compare trainer scores to original draws, not to a fresh inference
    score under identical weights. They therefore measure policy and numerical
    mismatch together. Values are local-rank diagnostics, not loss modifiers.
    """
    spans_by_sample = validate_batch(rollout)
    values = {"historical_prefix": [], "latest_forward": []}
    mixed, tokens, current_tokens = 0, 0, 0
    for spans, scores, behavior, mask, metadata in zip(
        spans_by_sample,
        rollout["log_probs"],
        rollout["rollout_log_probs"],
        rollout["loss_masks"],
        rollout["metadata"],
        strict=True,
    ):
        delta = scores.detach().float() - behavior.detach().float()
        replay_version = metadata["policy_refresh"]["replay_version"]
        mixed += int(len(spans) > 1)
        for span in spans:
            start, end = span["start"], span["end"]
            selected = delta[start:end][mask[start:end].bool()]
            kind = "historical_prefix" if span["version"] < replay_version else "latest_forward"
            values[kind].append(selected)
            tokens += selected.numel()
            if span["version"] == current_version:
                current_tokens += selected.numel()
    metrics = dict(
        mixed_responses=mixed,
        response_count=len(spans_by_sample),
        active_tokens=tokens,
        current_version_token_fraction=current_tokens / max(tokens, 1),
    )
    for kind, chunks in values.items():
        if not chunks:
            continue
        deltas = torch.cat(chunks)
        if not deltas.numel():
            continue
        ratios = deltas.exp()
        metrics[kind] = dict(
            tokens=deltas.numel(),
            mean_abs_logratio=deltas.abs().mean().item(),
            max_abs_logratio=deltas.abs().max().item(),
            tis_clip_fraction=((ratios < clip_low) | (ratios > clip_high)).float().mean().item(),
        )
    return metrics
