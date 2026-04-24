# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch


class CrossEntropyLoss:
    """Wrap the model's pre-computed CE loss."""

    def __call__(self, model_loss: torch.Tensor) -> torch.Tensor:
        return model_loss
