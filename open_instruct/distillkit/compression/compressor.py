# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch

from open_instruct.distillkit.compression.bitpack import pack_to_bytes, unpack_from_bytes
from open_instruct.distillkit.compression.config import DistributionQuantizationConfig
from open_instruct.distillkit.compression.monotonic_logprobs import (
    compress_monotonic_logprobs,
    decompress_monotonic_logprobs,
)


class LogprobCompressor:
    def __init__(self, config: DistributionQuantizationConfig):
        self.config = config
        self.vocab_index_bits = int(torch.log2(torch.tensor(self.config.d, dtype=torch.float32)).ceil().item())

    def compress_from_sparse(self, indices: torch.Tensor, logprobs: torch.Tensor) -> dict[str, torch.Tensor]:
        _, sorted_indices = torch.sort(logprobs, descending=True, dim=-1)
        sorted_values = logprobs.gather(-1, sorted_indices)
        sorted_indices = indices.gather(-1, sorted_indices)

        logprob_bytes = compress_monotonic_logprobs(sorted_values, self.config)
        index_bytes = pack_to_bytes(sorted_indices, self.vocab_index_bits)
        return {"compressed_logprobs": logprob_bytes, "bytepacked_indices": index_bytes}

    def decompress_to_sparse(self, row: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logprobs = decompress_monotonic_logprobs(row["compressed_logprobs"].to(torch.uint8), self.config)
        indices = unpack_from_bytes(
            row["bytepacked_indices"].to(torch.uint8), self.vocab_index_bits, original_num_elements=logprobs.shape[-1]
        )
        return indices, logprobs
