# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

from open_instruct.distillkit.compression import (
    DistributionQuantizationConfig,
    LogprobCompressor,
    QuantizationBin,
    SpecialTerm,
    pack_to_bytes,
    parse_torch_dtype,
    torch_dtype_bit_width,
    torch_dtype_to_name,
    unpack_from_bytes,
)
from open_instruct.distillkit.distill_loss import DistillationLossComputer, create_loss_function
from open_instruct.distillkit.lossfuncs import (
    CrossEntropyLoss,
    HingeLoss,
    JSDLoss,
    KLDLoss,
    LogisticRankingLoss,
    MissingProbabilityHandling,
    TVDLoss,
    sparse_hinge_loss,
    sparse_js_div,
    sparse_kl_div,
    sparse_logistic_ranking_loss,
    sparse_tvd,
)
from open_instruct.distillkit.signals import OfflineSignalSource, SparseSignal

__all__ = [
    "DistributionQuantizationConfig",
    "LogprobCompressor",
    "QuantizationBin",
    "SpecialTerm",
    "pack_to_bytes",
    "parse_torch_dtype",
    "torch_dtype_to_name",
    "torch_dtype_bit_width",
    "unpack_from_bytes",
    "SparseSignal",
    "OfflineSignalSource",
    "CrossEntropyLoss",
    "KLDLoss",
    "JSDLoss",
    "TVDLoss",
    "HingeLoss",
    "LogisticRankingLoss",
    "MissingProbabilityHandling",
    "sparse_kl_div",
    "sparse_js_div",
    "sparse_tvd",
    "sparse_hinge_loss",
    "sparse_logistic_ranking_loss",
    "DistillationLossComputer",
    "create_loss_function",
]
