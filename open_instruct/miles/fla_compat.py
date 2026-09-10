#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-local FLA compatibility, carried forward from olmo-miles.compat.fla_compat."""

from __future__ import annotations

from importlib import import_module, metadata
from typing import Any

from open_instruct import logger_utils

FLASH_LINEAR_ATTENTION_VERSION = "0.5.2"

logger = logger_utils.setup_logger(__name__)
_FLA_PATCHED = False


def triton_needs_fla_patch(version: str) -> bool:
    """Return whether FLA 0.5.2 needs its KDA constexpr compatibility shim."""
    try:
        major_minor = tuple(int(part) for part in version.split(".")[:2])
    except ValueError as error:
        raise RuntimeError(f"Cannot parse Triton version {version!r}") from error
    return major_minor >= (3, 6)


def rewrite_fla_kernel_source(source: str, *, block_width: int) -> str:
    """Replace FLA's unsupported constexpr helper with an equivalent literal."""
    old_expression = "    BK: tl.constexpr = triton.next_power_of_2(K)\n"
    new_expression = f"    BK: tl.constexpr = {block_width}\n"
    if new_expression in source and old_expression not in source:
        return source
    if old_expression not in source:
        raise RuntimeError("FLA 0.5.2 KDA source did not match the expected Triton compatibility target")
    return source.replace(old_expression, new_expression)


def install_kda_triton_compat() -> bool:
    """Patch one FLA KDA constexpr expression through Triton's source API.

    Returns:
        Whether the trainer-local source adapter was newly installed.

    Raises:
        RuntimeError: If the pinned FLA version or expected kernel source is absent.

    """
    global _FLA_PATCHED

    fla_version = metadata.version("flash-linear-attention")
    if fla_version != FLASH_LINEAR_ATTENTION_VERSION:
        raise RuntimeError(
            f"OLMo KDA requires flash-linear-attention=={FLASH_LINEAR_ATTENTION_VERSION}; found {fla_version}"
        )
    triton_version = metadata.version("triton")
    if _FLA_PATCHED or not triton_needs_fla_patch(triton_version):
        return False

    chunk_intra_module = import_module("fla.ops.kda.chunk_intra")
    token_parallel_module = import_module("fla.ops.kda.chunk_intra_token_parallel")
    triton = import_module("triton")

    kernel = token_parallel_module.chunk_kda_fwd_kernel_intra_token_parallel
    jit_kernel = kernel
    while hasattr(jit_kernel, "fn") and not hasattr(jit_kernel, "_unsafe_update_src"):
        jit_kernel = jit_kernel.fn
    if not hasattr(jit_kernel, "_unsafe_update_src"):
        raise RuntimeError("Triton KDA kernel no longer exposes the controlled source-update API")

    original_launcher = token_parallel_module.chunk_kda_fwd_intra_token_parallel
    patched_block_width: int | None = None

    def chunk_kda_fwd_intra_token_parallel_compat(
        q: Any,
        k: Any,
        gk: Any,
        beta: Any,
        Aqk: Any,
        Akk: Any,
        scale: float,
        cu_seqlens: Any = None,
        chunk_size: int = 64,
        sub_chunk_size: int = 16,
    ) -> None:
        nonlocal patched_block_width

        block_width = triton.next_power_of_2(q.shape[-1])
        if patched_block_width is None:
            jit_kernel._unsafe_update_src(rewrite_fla_kernel_source(jit_kernel.src, block_width=block_width))
            patched_block_width = block_width
        elif block_width != patched_block_width:
            raise RuntimeError(
                "FLA 0.5.2 KDA compatibility shim cannot mix head widths in one process: "
                f"compiled {patched_block_width}, received {block_width}"
            )

        return original_launcher(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            sub_chunk_size=sub_chunk_size,
        )

    token_parallel_module.__dict__["chunk_kda_fwd_intra_token_parallel"] = chunk_kda_fwd_intra_token_parallel_compat
    chunk_intra_module.__dict__["chunk_kda_fwd_intra_token_parallel"] = chunk_kda_fwd_intra_token_parallel_compat
    _FLA_PATCHED = True
    logger.info("Installed trainer-local FLA 0.5.2 compatibility for Triton %s", triton_version)
    return True
