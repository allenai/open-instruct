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
from open_instruct.distillkit.losses import (
    ForwardKLTopKOutput,
    forward_kl_topk_from_logprobs,
    gather_student_logprobs_at_teacher_topk,
)
from open_instruct.distillkit.signals import SparseTeacherSignal
from open_instruct.distillkit.vllm_logprobs import extract_response_topk_from_prompt_logprobs, process_prompt_logprobs

__all__ = [
    "DistributionQuantizationConfig",
    "ForwardKLTopKOutput",
    "LogprobCompressor",
    "QuantizationBin",
    "SparseTeacherSignal",
    "SpecialTerm",
    "extract_response_topk_from_prompt_logprobs",
    "forward_kl_topk_from_logprobs",
    "gather_student_logprobs_at_teacher_topk",
    "pack_to_bytes",
    "parse_torch_dtype",
    "process_prompt_logprobs",
    "torch_dtype_to_name",
    "torch_dtype_bit_width",
    "unpack_from_bytes",
]
