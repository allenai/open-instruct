"""Runtime patches loaded automatically by Python when enabled via env vars."""

from __future__ import annotations

import inspect
import os
from typing import Optional, Tuple


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

    import torch

    try:
        @torch.library.register_fake("_C::silu_and_mul")
        def _silu_and_mul_fake(out, x):
            return None
    except RuntimeError as exc:
        if "does not exist" not in str(exc):
            raise


def _patch_olmo_core_flash_attn_4_api() -> None:
    """Use keyword arguments for FA4 varlen calls with newer CUTE APIs.

    flash-attn-4 4.0.0b18 added an optional positional ``qv`` argument before
    ``cu_seqlens_q``. Older OLMo-core versions pass varlen metadata
    positionally, which shifts those arguments by one slot.
    """

    import torch
    from olmo_core.nn.attention import backend as attention_backend
    from olmo_core.nn.attention import flash_attn_api

    if flash_attn_api.flash_attn_4 is None:
        return

    signature = inspect.signature(flash_attn_api.flash_attn_4.flash_attn_varlen_func)
    if "qv" not in signature.parameters:
        return

    @torch._dynamo.disable()
    def _dispatch_flash_attn_4(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        seqused_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if window_size == (-1, -1) or (window_size == (-1, 0) and causal):
            window_size = (None, None)  # type: ignore

        if seqused_k is not None:
            B = q.shape[0]
            T = q.shape[1]
            cu_seqlens_q_cache = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
            return flash_attn_api.flash_attn_4.flash_attn_varlen_func(
                flash_attn_api._flatten_batch_dim(q),
                k,
                v,
                cu_seqlens_q=cu_seqlens_q_cache,
                cu_seqlens_k=None,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
            )[0]

        if cu_seqlens is not None:
            cu_seqlens_q = cu_seqlens if cu_seqlens_q is None else cu_seqlens_q
            cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
        if max_seqlen is not None:
            max_seqlen_q = max_seqlen if max_seqlen_q is None else max_seqlen_q
            max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k

        varlen = all(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k))

        if varlen:
            return flash_attn_api.flash_attn_4.flash_attn_varlen_func(
                flash_attn_api._flatten_batch_dim(q),
                flash_attn_api._flatten_batch_dim(k),
                flash_attn_api._flatten_batch_dim(v),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
            )[0]

        return flash_attn_api.flash_attn_4.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )[0]

    _dispatch_flash_attn_4._open_instruct_fa4_api_patch = True  # type: ignore[attr-defined]
    flash_attn_api.dispatch_flash_attn_4 = _dispatch_flash_attn_4
    attention_backend.dispatch_flash_attn_4 = _dispatch_flash_attn_4


if _truthy(os.environ.get("OPEN_INSTRUCT_PATCH_VLLM_RMS_NORM_FAKE_IMPL")):
    _patch_vllm_rms_norm_fake_impls()

if _truthy(os.environ.get("OPEN_INSTRUCT_PATCH_OLMO_CORE_FLASH_ATTN_4_API")):
    _patch_olmo_core_flash_attn_4_api()
