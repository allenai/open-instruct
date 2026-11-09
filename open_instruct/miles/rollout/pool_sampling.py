"""Selection policy independent of pool membership, replay and exposure accounting."""

import math
from collections import Counter

from open_instruct.miles.errors import InputError


def validate(sampling, pools):
    if not isinstance(sampling, dict) or set(sampling) - {"policy", "weights"}:
        raise InputError("data.sampling accepts policy and optional weights")
    policy = sampling.get("policy", "shuffle")
    if policy not in ("shuffle", "uniform", "weighted", "exposure"):
        raise InputError("data.sampling.policy must be shuffle, uniform, weighted, or exposure")
    weights = sampling.get("weights")
    if policy == "weighted":
        if not isinstance(weights, dict) or set(weights) != set(pools):
            raise InputError("Weighted sampling requires weights for exactly the configured pool names")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise InputError("Pool weights must be finite positive numbers")
    elif "weights" in sampling:
        raise InputError("Pool weights require data.sampling.policy=weighted")
    return {"policy": policy, **({"weights": dict(weights)} if policy == "weighted" else {})}


def draw_weights(sampling, active, *, trained, targets, terminal, pending):
    """Return relative draw weights; only exposure policy reads training outcomes."""
    if sampling["policy"] == "weighted":
        maximum = max(sampling["weights"][p] for p in active)
        return [sampling["weights"][p] / maximum for p in active]
    if sampling["policy"] != "exposure":
        return [1.0] * len(active)
    outstanding = Counter(entry["pool"] for entry in pending.values())
    weights = []
    for name in active:
        delivery = max(0.05, (trained[name] + 1) / (terminal[name] + 2))
        deficit = targets[name] - trained[name] - outstanding[name] * delivery
        # Bound proposal ratios (20:1) and use a 64-group correction horizon.
        weight = math.exp(max(-3, min(3, deficit / 64))) / delivery
        weights.append(max(0.25, min(5.0, weight)))
    return weights
