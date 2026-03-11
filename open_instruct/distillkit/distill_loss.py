# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

from typing import Any

import torch

from open_instruct.distillkit.compression import (
    DistributionQuantizationConfig,
    LogprobCompressor,
)
from open_instruct.distillkit.lossfuncs import (
    CrossEntropyLoss,
    HingeLoss,
    JSDLoss,
    KLDLoss,
    LogisticRankingLoss,
    MissingProbabilityHandling,
    TVDLoss,
)
from open_instruct.distillkit.signals import OfflineSignalSource, SparseSignal

DISTILL_LOSS_NAMES = {"kl", "jsd", "tvd", "hinge", "logistic_ranking"}


def create_loss_function(config: dict[str, Any]) -> tuple[str, float, Any]:
    func_name = config["function"]
    weight = config.get("weight", 1.0)

    if func_name == "cross_entropy":
        return func_name, weight, CrossEntropyLoss()
    if func_name == "kl":
        missing = MissingProbabilityHandling(
            config.get("missing_probability_handling", "zero")
        )
        return (
            func_name,
            weight,
            KLDLoss(
                temperature=config.get("temperature", 1.0),
                missing_probability_handling=missing,
                sparse_chunk_length=config.get("sparse_chunk_length"),
            ),
        )
    if func_name == "jsd":
        missing = MissingProbabilityHandling(
            config.get("missing_probability_handling", "zero")
        )
        return (
            func_name,
            weight,
            JSDLoss(
                temperature=config.get("temperature", 1.0),
                missing_probability_handling=missing,
                sparse_chunk_length=config.get("sparse_chunk_length"),
            ),
        )
    if func_name == "tvd":
        missing = MissingProbabilityHandling(
            config.get("missing_probability_handling", "zero")
        )
        return (
            func_name,
            weight,
            TVDLoss(
                temperature=config.get("temperature", 1.0),
                missing_probability_handling=missing,
                sparse_chunk_length=config.get("sparse_chunk_length"),
            ),
        )
    if func_name == "hinge":
        return func_name, weight, HingeLoss(margin=config.get("margin", 0.0))
    if func_name == "logistic_ranking":
        return func_name, weight, LogisticRankingLoss()
    raise ValueError(f"Unknown loss function: {func_name}")


class DistillationLossComputer:
    """Compute weighted loss combinations for offline distillation."""

    def __init__(
        self,
        loss_functions: list[dict[str, Any]],
        compressor_config: dict[str, Any],
        vocab_size: int,
    ):
        self.loss_funcs: list[tuple[str, float, Any]] = []
        self.has_distill_loss = False
        self.total_loss_weight = 0.0

        for cfg in loss_functions:
            name, weight, func = create_loss_function(cfg)
            self.loss_funcs.append((name, weight, func))
            self.total_loss_weight += weight
            if name in DISTILL_LOSS_NAMES:
                self.has_distill_loss = True

        if self.total_loss_weight <= 0:
            raise ValueError(
                f"Sum of loss weights must be > 0, got {self.total_loss_weight}"
            )

        if self.has_distill_loss:
            compression_cfg = DistributionQuantizationConfig.from_dict(
                compressor_config
            )
            compressor = LogprobCompressor(config=compression_cfg)
            self.signal_source = OfflineSignalSource(
                compressor=compressor,
                vocab_size=vocab_size,
                preapplied_temperature=1.0,
                log_values=True,
            )
        else:
            self.signal_source = None

    def get_teacher_signal(self, batch: dict[str, torch.Tensor]) -> SparseSignal:
        if self.signal_source is None:
            raise ValueError(
                "No distillation loss configured, cannot get teacher signal"
            )
        return self.signal_source.get_signal(batch)

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        model_loss: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_dict = {}
        total_loss = torch.tensor(
            0.0, device=student_logits.device, dtype=student_logits.dtype
        )

        signal = None
        if self.has_distill_loss:
            signal = self.get_teacher_signal(batch)
            mask = (labels[:, 1:] != -100).float()
            shifted_logits = student_logits[:, :-1, :]

        for name, weight, func in self.loss_funcs:
            if name == "cross_entropy":
                loss = func(model_loss)
            elif name in DISTILL_LOSS_NAMES:
                loss = func(student_logits=shifted_logits, signal=signal, mask=mask)
            else:
                raise ValueError(f"Unknown loss function: {name}")

            loss_dict[f"{name}_loss"] = loss.item()
            total_loss = total_loss + weight * loss

        total_loss = total_loss / self.total_loss_weight
        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
