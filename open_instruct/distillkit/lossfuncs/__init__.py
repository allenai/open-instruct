# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

from open_instruct.distillkit.lossfuncs.common import MissingProbabilityHandling
from open_instruct.distillkit.lossfuncs.cross_entropy import CrossEntropyLoss
from open_instruct.distillkit.lossfuncs.hingeloss import HingeLoss, sparse_hinge_loss
from open_instruct.distillkit.lossfuncs.jsd import JSDLoss, sparse_js_div
from open_instruct.distillkit.lossfuncs.kl import KLDLoss, sparse_kl_div
from open_instruct.distillkit.lossfuncs.logistic_ranking import (
    LogisticRankingLoss,
    sparse_logistic_ranking_loss,
)
from open_instruct.distillkit.lossfuncs.tvd import TVDLoss, sparse_tvd

__all__ = [
    "MissingProbabilityHandling",
    "CrossEntropyLoss",
    "KLDLoss",
    "JSDLoss",
    "TVDLoss",
    "HingeLoss",
    "LogisticRankingLoss",
    "sparse_kl_div",
    "sparse_js_div",
    "sparse_tvd",
    "sparse_hinge_loss",
    "sparse_logistic_ranking_loss",
]
