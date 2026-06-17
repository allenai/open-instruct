"""Runtime patches loaded automatically by Python when enabled via env vars."""

from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes"}


def _empty_like_for_fake(tensor):
    import torch

    return torch.empty_strided(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _patch_vllm_rms_norm_fake_impls() -> None:
    """Avoid vLLM RMSNorm fake kernels reading real CUDA weights.

    vLLM 0.23's default IR fake implementation for RMSNorm reuses the native
    implementation. During fake/meta execution, activations are meta tensors but
    RMSNorm weights can still be real CUDA tensors, which crashes with a
    device-mismatch error. For fake execution, only output metadata is needed.
    """

    from vllm.ir.ops import layernorm

    @layernorm.rms_norm.register_fake
    def _rms_norm_fake(x, weight, epsilon, variance_size=None):
        return _empty_like_for_fake(x)

    @layernorm.fused_add_rms_norm.register_fake
    def _fused_add_rms_norm_fake(x, x_residual, weight, epsilon, variance_size=None):
        return _empty_like_for_fake(x), _empty_like_for_fake(x_residual)


if _truthy(os.environ.get("OPEN_INSTRUCT_PATCH_VLLM_RMS_NORM_FAKE_IMPL")):
    _patch_vllm_rms_norm_fake_impls()
