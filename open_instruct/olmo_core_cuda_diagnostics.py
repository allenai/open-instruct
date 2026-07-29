"""Opt-in CUDA synchronization diagnostics for OLMo-core expert parallelism."""

from collections.abc import Callable
from functools import wraps
from importlib import import_module
from itertools import count
from typing import Any

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def install_ep_operation_sync(rank: int) -> None:
    """Synchronize around sync-1D EP operations to identify the first failing CUDA kernel."""
    ep_sync_1d = import_module("olmo_core.nn.moe.v2.ep_sync_1d")
    op_counter = count()

    def wrap_ep_op(op_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def wrapped(*args, **kwargs):
            op_index = next(op_counter)
            first_arg = args[0] if args else kwargs.get("inp", kwargs.get("x"))
            shape = tuple(first_arg.shape) if isinstance(first_arg, torch.Tensor) else None
            logger.warning(
                "[CUDAEPSync] rank=%s op_index=%s op=%s boundary=enter shape=%s", rank, op_index, op_name, shape
            )
            torch.cuda.synchronize()
            logger.warning(
                "[CUDAEPSync] rank=%s op_index=%s op=%s boundary=enter-synchronized", rank, op_index, op_name
            )
            result = original(*args, **kwargs)
            logger.warning("[CUDAEPSync] rank=%s op_index=%s op=%s boundary=exit", rank, op_index, op_name)
            torch.cuda.synchronize()
            logger.warning(
                "[CUDAEPSync] rank=%s op_index=%s op=%s boundary=exit-synchronized", rank, op_index, op_name
            )
            return result

        return wrapped

    ep_sync_1d.moe_permute_no_compile = wrap_ep_op("permute", ep_sync_1d.moe_permute_no_compile)
    ep_sync_1d.moe_unpermute_no_compile = wrap_ep_op("unpermute", ep_sync_1d.moe_unpermute_no_compile)
    ep_sync_1d.ops.all_to_all_async = wrap_ep_op("all_to_all_async", ep_sync_1d.ops.all_to_all_async)
    ep_sync_1d.ops.all_to_all_wait = wrap_ep_op("all_to_all_wait", ep_sync_1d.ops.all_to_all_wait)
