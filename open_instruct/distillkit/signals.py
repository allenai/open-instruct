# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
from typing import Any

import torch

from open_instruct.distillkit.compression import LogprobCompressor


@dataclass
class SparseSignal:
    """Sparse teacher signal with top-k logprobs."""

    sparse_ids: torch.LongTensor
    sparse_values: torch.Tensor
    log_values: bool
    generation_temperature: float
    vocab_size: int


class OfflineSignalSource:
    """Signal source for offline distillation from precomputed compressed logprobs."""

    def __init__(
        self,
        compressor: LogprobCompressor,
        vocab_size: int,
        preapplied_temperature: float = 1.0,
        log_values: bool = True,
    ):
        self.compressor = compressor
        self.vocab_size = vocab_size
        self.preapplied_temperature = preapplied_temperature
        self.log_values = log_values

    def get_signal(self, batch: dict[str, Any]) -> SparseSignal:
        with torch.no_grad():
            sparse_ids, sparse_values = self.compressor.decompress_to_sparse(batch)
        return SparseSignal(
            sparse_ids=sparse_ids,
            sparse_values=sparse_values,
            log_values=self.log_values,
            generation_temperature=self.preapplied_temperature,
            vocab_size=self.vocab_size,
        )
