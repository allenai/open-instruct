"""Domain-balanced token-mean loss (``training.loss_aggregation = "domain"``).

The token mean over a mixed batch weights a domain by its share of response tokens, so a domain with
short responses (IF, ~300 tokens) barely trains next to one with long ones (math, ~8k). Domain balance
keeps the token mean within each domain and averages the domains equally: a training token of domain
``d`` gets weight ``T / (D * T_d)``, where ``T`` is the rollout's training tokens, ``T_d`` domain
``d``'s, and ``D`` the number of domains with training tokens. The weights sum to ``T``, so the
global token-mean normalizer (``--calculate-per-token-loss``) is unchanged.

The reward post-process hooks see the whole rollout, compute the weights, and store them in each
sample's ``train_metadata``; the DPPO loss scales each sample's advantages by its weight. DPPO's
trust-region mask depends only on the advantage's sign, so a positive scale multiplies that sample's
loss exactly. One optimizer step per rollout is required: the weights are normalized over the rollout.
"""

import os

ENV = "OI_DPPO_DOMAIN_BALANCE"
WEIGHT_KEY = "oi_domain_weight"
DOMAIN_KEY = "domain"


def environment(loss_aggregation):
    """The ``OI_DPPO_DOMAIN_BALANCE`` setting for a run file's ``training.loss_aggregation``."""
    return {ENV: "1"} if loss_aggregation == "domain" else {}


def enabled(environment=None):
    return (os.environ if environment is None else environment).get(ENV, "0") == "1"


def _training_tokens(sample):
    if getattr(sample, "remove_sample", False):
        return 0
    mask = getattr(sample, "loss_mask", None)
    return int(sum(mask)) if mask is not None else int(sample.response_length)


def weights(samples):
    """Per-sample domain weights over one rollout; samples without training tokens get 1.0."""
    domains = []
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        if DOMAIN_KEY not in metadata:
            raise ValueError(f"Domain-balanced loss needs metadata[{DOMAIN_KEY!r}] on every training sample")
        domains.append(str(metadata[DOMAIN_KEY]))
    tokens = [_training_tokens(sample) for sample in samples]
    per_domain = {}
    for domain, count in zip(domains, tokens, strict=True):
        per_domain[domain] = per_domain.get(domain, 0) + count
    active = {domain: count for domain, count in per_domain.items() if count > 0}
    total = sum(active.values())
    return [total / (len(active) * active[domain]) if domain in active else 1.0 for domain in domains]


def attach(samples):
    """Store each sample's domain weight in its ``train_metadata``; return the weight per domain."""
    by_domain = {}
    for sample, weight in zip(samples, weights(samples), strict=True):
        sample.train_metadata = {**(sample.train_metadata or {}), WEIGHT_KEY: weight}
        by_domain.setdefault(sample.metadata[DOMAIN_KEY], weight)
    return by_domain


def batch_weights(batch):
    """The per-sample weights a train micro-batch carries, in the order of its per-sample tensors."""
    metadata = batch.get("metadata")
    if metadata is None or any(not isinstance(item, dict) or WEIGHT_KEY not in item for item in metadata):
        raise ValueError(f"Domain-balanced loss needs {WEIGHT_KEY!r} in every sample's train metadata")
    return [float(item[WEIGHT_KEY]) for item in metadata]
