# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch


def value_clipped_mse_loss(
    new_values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None,
    mask: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO-style clipped value loss. Returns per-token loss and clip fraction."""
    mask_f = mask.float()
    losses = (new_values - returns).pow(2)
    if clip_range > 0 and old_values is not None:
        clipped_values = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
        clipped_losses = (clipped_values - returns).pow(2)
        per_token = torch.maximum(losses, clipped_losses)
        clipfrac = ((clipped_losses > losses).float() * mask_f).sum() / mask_f.sum().clamp(min=1)
    else:
        per_token = losses
        clipfrac = torch.zeros((), dtype=torch.float32, device=new_values.device)
    return per_token * mask_f, clipfrac
