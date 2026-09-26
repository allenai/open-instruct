"""Numerical contracts at the MILES/Core boundary; no model-sized diagnostic copies."""

import dataclasses
import json
import math
from pathlib import Path

import torch
from torch import distributed as dist
from torch.distributed.tensor import DTensor


@dataclasses.dataclass(frozen=True)
class StepNormalization:
    samples: int
    active_tokens: int
    token_denominator: int
    model_tokens: int
    world_size: int

    @property
    def auxiliary_denominator(self):
        return self.model_tokens / self.world_size

    def scale_token_loss(self, loss):
        return loss * self.world_size / self.token_denominator


def step_normalization(batches, global_batch_size):
    """Match MILES' clamped per-sample token denominator and Core rank averaging."""
    masks = [mask for batch in batches for mask in batch["loss_masks"]]
    counts = torch.tensor(
        [
            len(masks),
            sum(int(m.sum()) for m in masks),
            sum(max(int(m.sum()), 1) for m in masks),
            sum(batch["tokens"].numel() for batch in batches),
        ],
        device=masks[0].device,
        dtype=torch.int64,
    )
    dist.all_reduce(counts)
    samples, active, denominator, model_tokens = counts.tolist()
    if samples != global_batch_size or active == 0 or model_tokens < active:
        raise ValueError(f"Invalid optimizer batch counts: {counts.tolist()}, expected {global_batch_size} samples")
    return StepNormalization(samples, active, denominator, model_tokens, dist.get_world_size())


def validate_training_data(rollout):
    """Reject bad active values before MILES' defensive nan_to_num can conceal them."""
    masks = rollout["loss_masks"]
    for key in ("log_probs", "rollout_log_probs", "ref_log_probs", "advantages", "returns"):
        if rollout.get(key) is None:
            continue
        if len(rollout[key]) != len(masks):
            raise ValueError(f"{key} sample count differs from masks")
        for values, mask in zip(rollout[key], masks, strict=True):
            values = torch.as_tensor(values, device=mask.device)
            if values.shape != mask.shape or not bool(torch.isfinite(values[mask.bool()]).all()):
                raise ValueError(f"Invalid active {key}")
    if not all(math.isfinite(float(reward)) for reward in rollout["rewards"]):
        raise ValueError("Non-finite reward")


def probability_profile(rollout):
    """Globally reduced error tails, relative-position thirds and length buckets.

    Fixed histograms avoid gathering token arrays. Quantile values are upper
    bin edges (not exact quantiles); max and means are exact reductions.
    """
    device = rollout["log_probs"][0].device
    edges = torch.tensor([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0], device=device)
    hist = torch.zeros(10, dtype=torch.float64, device=device)
    groups = torch.zeros(6, 2, dtype=torch.float64, device=device)
    maximum = torch.zeros((), dtype=torch.float64, device=device)
    for current, old, mask in zip(
        rollout["log_probs"], rollout["rollout_log_probs"], rollout["loss_masks"], strict=True
    ):
        active = mask.bool()
        errors = (current.detach().float() - torch.as_tensor(old, device=device).float()).abs()[active]
        if not errors.numel():
            continue
        hist += torch.bincount(torch.bucketize(errors, edges), minlength=10)
        maximum = torch.maximum(maximum, errors.max().double())
        positions = torch.arange(len(mask), device=device)[active] * 3 // max(len(mask), 1)
        for index in range(3):
            values = errors[positions == index]
            groups[index, 0] += values.double().sum()
            groups[index, 1] += values.numel()
        length_bucket = 3 + int(len(mask) > 1024) + int(len(mask) > 4096)
        groups[length_bucket, 0] += errors.double().sum()
        groups[length_bucket, 1] += errors.numel()
    dist.all_reduce(hist)
    dist.all_reduce(groups)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    total = int(hist.sum())
    if not total:
        raise ValueError("No active tokens for probability comparison")
    edges_list = edges.tolist() + [None]
    quantiles = {}
    for name, fraction in (("p50_upper", 0.5), ("p95_upper", 0.95), ("p99_upper", 0.99)):
        index = int(torch.searchsorted(hist.cumsum(0), hist.sum() * fraction))
        quantiles[name] = edges_list[index]
    labels = ("first_third", "middle_third", "last_third", "length_le_1024", "length_1025_4096", "length_gt_4096")
    return {
        "active_tokens": total,
        "max_abs": float(maximum),
        **quantiles,
        "histogram_upper_edges": edges_list,
        "histogram_counts": hist.long().tolist(),
        "groups": {
            name: {"tokens": int(n), "mean_abs": float(s / n) if n else None}
            for name, (s, n) in zip(labels, groups.tolist(), strict=True)
        },
    }


SCORING_CHECK_EDGE = 1e-3


def scoring_check(standalone, training, masks):
    """Local sums comparing standalone scoring to the training forward on active tokens.

    Returns ``(sums, maximum)``: sums hold absolute error, token count and the
    number of tokens above ``SCORING_CHECK_EDGE``; maximum is the largest error.
    The caller reduces sums with SUM and maximum with MAX before validating.
    """
    device = masks[0].device
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    maximum = torch.zeros((), dtype=torch.float64, device=device)
    if len(standalone) != len(training) or len(standalone) != len(masks):
        raise ValueError("Scoring check requires one standalone and one training score per sample")
    for reference, current, mask in zip(standalone, training, masks, strict=True):
        reference = torch.as_tensor(reference, device=device)
        current = torch.as_tensor(current, device=device)
        if reference.ndim != 1 or reference.shape != current.shape or reference.shape != mask.shape:
            raise ValueError("Scoring check score and response-mask shapes differ")
        errors = (reference.float() - current.float()).abs()[mask.bool()]
        if not bool(torch.isfinite(errors).all()):
            raise ValueError("Non-finite active-token log probability in scoring check")
        if not errors.numel():
            continue
        sums[0] += errors.double().sum()
        sums[1] += errors.numel()
        sums[2] += (errors > SCORING_CHECK_EDGE).sum()
        maximum = torch.maximum(maximum, errors.max().double())
    return sums, maximum


def validate_scoring_check(sums, maximum, tolerance):
    """Fail on a globally reduced mean above tolerance; report the reduced statistics."""
    tokens = int(sums[1])
    if not tokens:
        raise ValueError("No active tokens for scoring check")
    mean_abs = float(sums[0] / tokens)
    report = {
        "active_tokens": tokens,
        "mean_abs": mean_abs,
        "max_abs": float(maximum),
        "tokens_above_edge": int(sums[2]),
        "edge": SCORING_CHECK_EDGE,
        "tolerance": tolerance,
    }
    if mean_abs > tolerance:
        raise ValueError(
            f"Standalone scoring differs from the training forward: mean_abs {mean_abs:.6g} exceeds {tolerance:.6g}"
        )
    return report


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _category(name):
    if "router" in name:
        return "router"
    if "routed_experts" in name or "shared_experts" in name:
        return "expert"
    return "dense"


class ParameterProbe:
    """Optional local gradient norms and sampled updates; never claim global EP norms.

    Gradient values are read before Core optimizer intake (and thus before its
    EP-MP rescaling). Updates sample at most 256 entries of each named parameter.
    FP8 weight stores outside named_parameters are explicitly outside coverage.
    Pass persistent named parameters captured at train-module initialization,
    before FSDP can replace model-visible parameters with temporary full views.
    """

    def __init__(self, named_parameters):
        self.parameters = list(named_parameters)
        self.before = {}
        for name, parameter in self.parameters:
            flat = _local(parameter.detach()).reshape(-1)
            stride = max(1, math.ceil(flat.numel() / 256))
            self.before[name] = (stride, flat[::stride].float().clone())

    def gradients(self):
        results = {}
        for name, parameter in self.parameters:
            category = _category(name)
            result = results.setdefault(category, {"sum_squares": 0.0, "elements": 0, "missing_parameters": 0})
            grad = getattr(parameter, "_olmo_ddp_reduced_grad_shard", None)
            if grad is None:
                grad = getattr(parameter, "_main_grad_fp32", None)
            if grad is None:
                grad = parameter.grad
            if grad is None:
                result["missing_parameters"] += 1
                continue
            grad = _local(grad.detach())
            norm = float(torch.linalg.vector_norm(grad, dtype=torch.float32))
            if not math.isfinite(norm):
                raise ValueError(f"Non-finite gradient: {name}")
            result["sum_squares"] += norm * norm
            result["elements"] += grad.numel()
        for result in results.values():
            result["local_l2"] = math.sqrt(result.pop("sum_squares"))
        return results

    def updates(self):
        results = {}
        for name, parameter in self.parameters:
            stride, before = self.before[name]
            current = _local(parameter.detach()).reshape(-1)[::stride].float()
            if not bool(torch.isfinite(current).all()):
                raise ValueError(f"Non-finite updated parameter: {name}")
            result = results.setdefault(_category(name), {"sum_squares": 0.0, "sampled_elements": 0})
            result["sum_squares"] += float((current - before).double().square().sum())
            result["sampled_elements"] += current.numel()
        for result in results.values():
            result["sampled_update_l2"] = math.sqrt(result.pop("sum_squares"))
        return results


def record(args, event):
    """Per-rank files avoid concurrent writers; JSON rejects non-finite diagnostics."""
    document = json.dumps({"schema_version": 1, "rank": dist.get_rank(), **event}, allow_nan=False, sort_keys=True)
    if args.save:
        path = Path(args.save) / f"training_contract_rank{dist.get_rank()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(document + "\n")
    return document


def auxiliary_metrics(model, *, reset=False):
    """Read Core's forward-only accumulators, excluding checkpoint recomputation."""
    result = {"load_balancing": 0.0, "router_z": 0.0}
    for module in model.modules():
        if not hasattr(module, "compute_aux_loss"):
            continue
        if reset:
            module.reset_metrics()
            continue
        for output, field, coefficient in (
            ("load_balancing", "load_balancing_loss", "lb_loss_weight"),
            ("router_z", "z_loss", "z_loss_weight"),
        ):
            value = getattr(module, field, None)
            weight = getattr(module, coefficient, None)
            if value is not None and weight is not None:
                result[output] += float(value) * weight
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite router auxiliary loss")
    return result


def validate_step_transition(clock, optimizer, device):
    """Agree before mutating any clock; a skipped rank must stop every rank."""
    state = torch.tensor(
        [clock.completed_steps, clock.published_step, int(bool(getattr(optimizer, "step_skipped", False)))],
        dtype=torch.int64,
        device=device,
    )
    minimum, maximum = state.clone(), state.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum[:2], maximum[:2]):
        raise ValueError("Trainer ranks disagree on policy clock")
    if int(maximum[2]):
        raise ValueError("Optimizer step skipped on at least one rank; no policy clock may advance")


def validate_batch_schedule(batch_count, local_batch_size, device):
    counts = torch.tensor([batch_count], device=device, dtype=torch.int64)
    minimum, maximum = counts.clone(), counts.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if int(minimum) != int(maximum) or batch_count % local_batch_size:
        raise ValueError("Trainer ranks must execute the same number of complete optimizer batches")


def validate_schedule(clock, scheduler):
    if scheduler.last_epoch != clock.completed_steps:
        raise ValueError("LR scheduler and policy clock disagree")
    if not all(math.isfinite(value) and value >= 0 for value in scheduler.get_last_lr()):
        raise ValueError("Invalid learning rate before optimizer step")
