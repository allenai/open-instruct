"""Per-rank allocator peaks spanning forward/backward and the optimizer step."""

import functools

import torch

from open_instruct import logger_utils
from open_instruct.miles.distillation import opd_timing

logger = logger_utils.setup_logger(__name__)


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    if not opd_timing.enabled():
        return
    device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer._oi_opd_memory_context = dict(rollout_id=rollout_id, step_id=step_id, rank=args.rank, device=device)
    if getattr(optimizer, "_oi_opd_memory_wrapped", False):
        return
    original = optimizer.step

    @functools.wraps(original)
    def measured_step(*positional, **keywords):
        try:
            return original(*positional, **keywords)
        finally:
            try:
                context = optimizer._oi_opd_memory_context
                free, total = torch.cuda.mem_get_info(context["device"])
                opd_timing.write(
                    dict(
                        stage="learner_memory",
                        **context,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(context["device"]),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(context["device"]),
                        allocated_bytes=torch.cuda.memory_allocated(context["device"]),
                        reserved_bytes=torch.cuda.memory_reserved(context["device"]),
                        device_free_bytes=free,
                        device_total_bytes=total,
                    )
                )
            except Exception:
                logger.exception("OPD allocator observation unavailable")

    optimizer.step = measured_step
    optimizer._oi_opd_memory_wrapped = True
