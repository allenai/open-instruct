"""Core trainer metrics using MILES' sample/token accounting and tracking run."""

import torch
from miles.backends.training_utils import log_utils
from miles.utils.tracking_utils import tracking
from torch import distributed as dist

_COUNT = "_miles_metric_count"
_OBJECTIVE = "normalized_policy_objective"


def init_tracking(args):
    """Join the driver's run from one trainer process, like the MILES trainers."""
    if dist.get_rank() == 0:
        tracking.init_tracking(args, primary=False)


def loss_metrics(payload, loss):
    """Preserve MILES' count alongside its unnormalized metric numerators."""
    return {
        **dict(zip(payload["keys"], payload["values"][1:], strict=True)),
        _COUNT: payload["values"][0],
        _OBJECTIVE: loss.detach(),
    }


def aggregate_losses(microbatches):
    """Use the same global sample/token averages as Megatron and FSDP MILES."""
    payloads = []
    for batch in microbatches:
        keys = [key for key in batch if key not in (_COUNT, _OBJECTIVE)]
        payloads.append({"keys": keys, "values": torch.stack([batch[_COUNT], *(batch[key] for key in keys)])})
    return log_utils.aggregate_train_losses(payloads)


def step_summary(microbatches, auxiliary, elapsed_seconds):
    """Average already globally normalized objectives; time the slowest rank.

    Core's policy and auxiliary gradients are averaged across the training
    world, so their rank-local scaled objectives must receive that same average.
    The raw MILES loss numerators instead use ``aggregate_losses`` above.
    """
    device = microbatches[0][_OBJECTIVE].device
    totals = torch.tensor(
        [sum(float(batch[_OBJECTIVE]) for batch in microbatches), auxiliary["load_balancing"], auxiliary["router_z"]],
        dtype=torch.float64,
        device=device,
    )
    elapsed = torch.tensor(elapsed_seconds, dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    totals /= dist.get_world_size()
    return {
        "policy_objective": float(totals[0]),
        "auxiliary_load_balancing": float(totals[1]),
        "auxiliary_router_z": float(totals[2]),
        "step_seconds": float(elapsed),
    }


def score_metrics(mean_abs, active_tokens, profile):
    """Collection pre-update drift is token weighted, unlike response loss means."""
    result = {
        "collection_score_mean_abs": mean_abs,
        "collection_score_active_tokens": active_tokens,
        "collection_score_max_abs": profile["max_abs"],
    }
    for quantile in ("p50_upper", "p95_upper", "p99_upper"):
        upper = profile[quantile]
        # None means the quantile lies beyond the histogram's last finite edge.
        result[f"collection_score_{quantile}_overflow"] = int(upper is None)
        if upper is not None:
            result[f"collection_score_{quantile}"] = upper
    return result


def log_step(
    args, *, losses, summary, scores, clock, rollout_id, lr_used, lr_next, optimizer_metrics, gradient_stats=None
):
    """Only rank zero logs; local probes never masquerade as global norms."""
    if dist.get_rank() != 0:
        return None
    values = {**losses, **summary, **scores}
    if args.entropy_coef == 0 and not args.observe_training_entropy:
        values.pop("entropy_loss", None)
    values.update(completed_steps=clock.completed_steps, published_step=clock.published_step, rollout_id=rollout_id)
    # MILES reports scheduled LR after the update; also retain the LR actually used.
    values.update({f"lr-pg_{index}": value for index, value in enumerate(lr_next)})
    values.update({f"lr_used-pg_{index}": value for index, value in enumerate(lr_used)})
    if "optim/total grad norm" in optimizer_metrics:
        # This is Core's optimizer-provided, already-reduced preclip norm.
        values["grad_norm"] = optimizer_metrics["optim/total grad norm"]
    for category, stats in (gradient_stats or {}).items():
        values[f"rank0_local_pre_optimizer/{category}/l2"] = stats["local_l2"]
    output = {f"train/{key}": value for key, value in values.items()}
    output["train/step"] = clock.completed_steps - 1
    tracking.log(args, output, step_key="train/step")
    return output
