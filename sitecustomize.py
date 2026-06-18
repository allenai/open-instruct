"""Runtime patches loaded automatically by Python when enabled via env vars."""

from __future__ import annotations

import builtins
import inspect
import os
import sys
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


_OLMO_CORE_FA4_API_PATCHED = False
_OLMO_CORE_FA4_API_PATCHING = False


def _patch_olmo_core_flash_attn_4_api() -> None:
    """Use keyword arguments for FA4 varlen calls with newer CUTE APIs.

    flash-attn-4 4.0.0b18 added an optional positional ``qv`` argument before
    ``cu_seqlens_q``. Older OLMo-core versions pass varlen metadata
    positionally, which shifts those arguments by one slot.
    """

    global _OLMO_CORE_FA4_API_PATCHED, _OLMO_CORE_FA4_API_PATCHING
    if _OLMO_CORE_FA4_API_PATCHED or _OLMO_CORE_FA4_API_PATCHING:
        return

    flash_attn_api = sys.modules.get("olmo_core.nn.attention.flash_attn_api")
    attention_backend = sys.modules.get("olmo_core.nn.attention.backend")
    if flash_attn_api is None or attention_backend is None:
        return
    if not hasattr(flash_attn_api, "flash_attn_4") or not hasattr(attention_backend, "dispatch_flash_attn_4"):
        return

    if getattr(attention_backend.dispatch_flash_attn_4, "_open_instruct_fa4_api_patch", False):
        _OLMO_CORE_FA4_API_PATCHED = True
        return

    if flash_attn_api.flash_attn_4 is None:
        return

    signature = inspect.signature(flash_attn_api.flash_attn_4.flash_attn_varlen_func)
    if "qv" not in signature.parameters:
        return

    _OLMO_CORE_FA4_API_PATCHING = True
    try:
        import torch

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
        _OLMO_CORE_FA4_API_PATCHED = True
    finally:
        _OLMO_CORE_FA4_API_PATCHING = False


def _install_olmo_core_flash_attn_4_api_import_patch() -> None:
    """Patch OLMo-core FA4 after its attention modules are imported."""

    if getattr(builtins.__import__, "_open_instruct_olmo_core_fa4_import_patch", False):
        return

    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if level == 0 and name.startswith("olmo_core.nn.attention"):
            _patch_olmo_core_flash_attn_4_api()
        return module

    _import._open_instruct_olmo_core_fa4_import_patch = True  # type: ignore[attr-defined]
    builtins.__import__ = _import
    _patch_olmo_core_flash_attn_4_api()


if _truthy(os.environ.get("OPEN_INSTRUCT_PATCH_VLLM_RMS_NORM_FAKE_IMPL")):
    _patch_vllm_rms_norm_fake_impls()

if _truthy(os.environ.get("OPEN_INSTRUCT_PATCH_OLMO_CORE_FLASH_ATTN_4_API")):
    _install_olmo_core_flash_attn_4_api_import_patch()
