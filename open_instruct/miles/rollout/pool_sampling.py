"""Selection policy independent of pool membership, replay and exposure accounting."""

import math
from collections import Counter

from open_instruct.miles.errors import InputError


def validate(sampling, pools):
    if not isinstance(sampling, dict) or set(sampling) - {"policy", "weights", "priority"}:
        raise InputError("data.sampling accepts policy, optional weights and optional priority")
    policy = sampling.get("policy", "shuffle")
    if policy not in ("shuffle", "uniform", "weighted", "exposure"):
        raise InputError("data.sampling.policy must be shuffle, uniform, weighted, or exposure")
    weights = sampling.get("weights")
    weighted = policy == "weighted" or (policy == "exposure" and "weights" in sampling)
    if weighted:
        if not isinstance(weights, dict) or set(weights) != set(pools):
            raise InputError("Weighted sampling requires weights for exactly the configured pool names")
        try:
            valid = all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in weights.values())
        except OverflowError:
            valid = False
        if not valid:
            raise InputError("Pool weights must be finite positive numbers")
        if policy == "exposure" and pools:
            shares = target_shares(weights, list(pools))
            # Leave enough floating-point range for the full bounded correction
            # and delivery-estimate range, rather than fail after GPU allocation.
            spread = math.log(min(shares.values())) - math.log(max(shares.values()))
            if math.exp(spread - 6 + math.log(0.05)) == 0:
                raise InputError("Exposure pool weight ratios are too extreme for finite proposal corrections")
    elif "weights" in sampling:
        raise InputError("Pool weights require data.sampling.policy=weighted or exposure")
    result = {"policy": policy, **({"weights": dict(weights)} if weighted else {})}
    if "priority" in sampling:
        priority = sampling["priority"]
        if policy not in ("uniform", "exposure"):
            raise InputError("Priority ordering requires data.sampling.policy=uniform or exposure")
        if (
            not isinstance(priority, dict)
            or set(priority) != {"field"}
            or not isinstance(priority["field"], str)
            or not priority["field"].strip()
        ):
            raise InputError("data.sampling.priority requires a nonempty metadata field")
        result["priority"] = {"field": priority["field"]}
    return result


def priority_value(metadata, field):
    """Use the same four historical strata as standalone priority sampling."""
    value = (metadata or {}).get(field)
    if type(value) is not int or not 1 <= value <= 4:
        raise InputError(f"Priority metadata.{field} must be an integer in 1..4; got {value!r}")
    return value


def target_shares(weights, active):
    """Normalize explicit target weights over a selection's available pools."""
    if not active:
        return {}
    maximum = max(weights[name] for name in active)
    scaled = {name: weights[name] / maximum for name in active}
    total = math.fsum(scaled.values())
    shares = {name: value / total for name, value in scaled.items()}
    if any(share <= 0 for share in shares.values()):
        raise InputError("Exposure pool weights must have representable positive normalized shares")
    return shares


def _weighted_exposure(sampling, active, *, trained, targets, terminal, pending):
    shares = target_shares(sampling["weights"], active)
    outstanding = Counter(entry["pool"] for entry in pending.values())
    delivery = {name: max(0.05, (trained[name] + 1) / (terminal[name] + 2)) for name in active}
    projected = {name: targets[name] - trained[name] - outstanding[name] * delivery[name] for name in active}
    total = math.fsum(projected.values())
    scores = []
    for name in active:
        # Allocate projected pending exposure to the desired shares. Removing
        # the common balance also handles historical debt in exhausted pools.
        deficit = projected[name] - shares[name] * total
        scores.append(math.log(shares[name]) - math.log(delivery[name]) + max(-3, min(3, deficit / 64)))
    maximum = max(scores)
    # Bound feedback relative to each pool's baseline, never absolute weights:
    # a 0.3% target must remain possible alongside a 50% backbone.
    return [math.exp(score - maximum) for score in scores]


def draw_weights(sampling, active, *, trained, targets, terminal, pending):
    """Return relative draw weights; only exposure policy reads training outcomes."""
    if sampling["policy"] == "weighted":
        maximum = max(sampling["weights"][p] for p in active)
        return [sampling["weights"][p] / maximum for p in active]
    if sampling["policy"] != "exposure":
        return [1.0] * len(active)
    if "weights" in sampling:
        return _weighted_exposure(
            sampling, active, trained=trained, targets=targets, terminal=terminal, pending=pending
        )
    outstanding = Counter(entry["pool"] for entry in pending.values())
    weights = []
    for name in active:
        delivery = max(0.05, (trained[name] + 1) / (terminal[name] + 2))
        deficit = targets[name] - trained[name] - outstanding[name] * delivery
        # Bound proposal ratios (20:1) and use a 64-group correction horizon.
        weight = math.exp(max(-3, min(3, deficit / 64))) / delivery
        weights.append(max(0.25, min(5.0, weight)))
    return weights
