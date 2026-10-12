"""Construct bounded token pressure without reversing cancelled encouragement."""

import torch


def token_pressure(student, full, answer, *, clip=None):
    values = [torch.as_tensor(value, dtype=torch.float32).detach() for value in (student, full, answer)]
    if not values[0].numel() or any(value.ndim != 1 or value.shape != values[0].shape for value in values):
        raise ValueError("Cancellation requires aligned, nonempty response score vectors")
    if any(not bool(torch.isfinite(value).all()) or bool((value > 1e-5).any()) for value in values):
        raise ValueError("Cancellation requires finite nonpositive log probabilities")
    student, full, answer = values
    boost = full - student
    overlap = torch.minimum(boost.clamp_min(0), (answer - student).clamp_min(0))
    pressure = boost - overlap
    unclipped = pressure.clone()
    if clip is not None:
        if not isinstance(clip, (float, int)) or not 0 < clip < float("inf"):
            raise ValueError("pressure_clip must be a positive finite number")
        pressure = pressure.clamp(-clip, clip)
    metrics = {
        "tokens": pressure.numel(),
        "full_positive_sum": boost.clamp_min(0).sum().item(),
        "full_negative_sum": (-boost).clamp_min(0).sum().item(),
        "cancelled_positive_sum": overlap.sum().item(),
        "retained_positive_sum": pressure.clamp_min(0).sum().item(),
        "retained_negative_sum": (-pressure).clamp_min(0).sum().item(),
        "pressure_sum": pressure.sum().item(),
        "pressure_squared_sum": pressure.square().sum().item(),
        "pressure_abs_max": pressure.abs().max().item(),
        "zero_tokens": (pressure == 0).sum().item(),
        "positive_tokens": (pressure > 0).sum().item(),
        "negative_tokens": (pressure < 0).sum().item(),
        "clipped_tokens": (pressure != unclipped).sum().item(),
    }
    return pressure, metrics
