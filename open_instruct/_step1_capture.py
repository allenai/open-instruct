"""One-shot training-step capture for trainer parity diagnosis.

Activates when OPEN_INSTRUCT_DUMP_DIR is set. Writes a torch-pickle to
$OPEN_INSTRUCT_DUMP_DIR/{trainer}_step{N}_rank{R}.pt with inputs, intermediates,
and per-parameter gradient summaries. See match-grpo.md for context.
"""

import dataclasses
import os
from typing import Any

import torch
import torch.distributed as dist

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def is_active(global_step: int) -> bool:
    if not os.environ.get("OPEN_INSTRUCT_DUMP_DIR"):
        return False
    target = int(os.environ.get("OPEN_INSTRUCT_DUMP_STEP", "1"))
    return global_step == target


def _to_cpu_list(tensors):
    return [t.detach().to("cpu", copy=True) for t in tensors]


def snapshot_inputs(data_BT) -> dict[str, Any]:
    out = {}
    for f in dataclasses.fields(data_BT):
        v = getattr(data_BT, f.name)
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[f.name] = _to_cpu_list(v)
        else:
            out[f.name] = v
    return out


def snapshot_param_grads(named_params) -> dict[str, dict[str, float | tuple]]:
    summary: dict[str, dict[str, float | tuple]] = {}
    for name, p in named_params:
        g = p.grad
        if g is None:
            continue
        g_local = g.to_local() if hasattr(g, "to_local") else g
        w_local = p.to_local() if hasattr(p, "to_local") else p
        g_local = g_local.detach()
        w_local = w_local.detach()
        summary[name] = {
            "grad_local_norm": float(g_local.float().norm(2).item()),
            "grad_local_abs_mean": float(g_local.float().abs().mean().item()),
            "grad_local_shape": tuple(g_local.shape),
            "weight_local_norm": float(w_local.float().norm(2).item()),
            "weight_local_shape": tuple(w_local.shape),
            "is_dtensor": hasattr(p, "to_local"),
        }
    return summary


def write(trainer: str, global_step: int, payload: dict[str, Any]) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    out_dir = os.environ["OPEN_INSTRUCT_DUMP_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{trainer}_step{global_step}_rank{rank}.pt")
    torch.save(payload, path)
    logger.info(f"[step1_capture] wrote {path} ({len(payload)} top-level keys)")
