"""Opt-in H235 half-group gradient probe, run only in an isolated diagnostic image.

One extra masked backward estimates half A. The ordinary full-batch backward
estimates F; half B is 2F-A. No optimizer step occurs between these measurements.
A CPU snapshot retains optimizer-owned gradient slices, including every expert.
"""

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import distributed as dist

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def build_frozen_module(args, builder):
    """Satisfy Core's positive constructor LR, then restore zero for its scheduler.

    No forward/backward or optimizer update occurs during construction. The
    actor creates CoreLRScheduler immediately afterwards using restored args.lr.
    """
    if args.lr != 0:
        raise ValueError("Fixed-weight noise diagnostics require zero scheduled LR")
    args.lr = 1e-6
    try:
        return builder(args)
    finally:
        args.lr = 0.0


def partition(indices_by_rank, siblings, seed):
    groups = Counter(index // siblings for indices in indices_by_rank for index in indices)
    if len(groups) % 2 or not groups or any(count != siblings for count in groups.values()):
        raise ValueError("Probe requires complete equal-sized groups and an even group count")
    indices = [index for rank in indices_by_rank for index in rank]
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate response identities in probe batch")
    identities = set(indices)
    for group in groups:
        if set(range(group * siblings, (group + 1) * siblings)) - identities:
            raise ValueError("Sample identities do not form complete sibling groups")
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    return set(order[: len(order) // 2]), len(groups)


def statistics(half_norm2, full_norm2, half_full_dot, groups, siblings):
    if not all(math.isfinite(x) for x in (half_norm2, full_norm2, half_full_dot)):
        raise ValueError("Non-finite gradient statistic")
    signal = 2 * half_full_dot - half_norm2
    noise = half_norm2 + full_norm2 - 2 * half_full_dot
    tolerance = 1e-5 * max(half_norm2, full_norm2, abs(half_full_dot), 1e-30)
    if noise < -tolerance:
        raise ValueError("Negative squared gradient difference")
    noise = max(noise, 0.0)
    return {
        "half_a_norm2": half_norm2,
        "full_norm2": full_norm2,
        "half_a_full_dot": half_full_dot,
        "signal_power": signal,
        "full_batch_noise_power": noise,
        "groups": groups,
        "responses": groups * siblings,
        # Negative signal estimates are retained, never clamped into a conclusion.
        "noise_to_signal": noise / signal if signal > 0 else None,
        "noise_scale_groups": groups * noise / signal if signal > 0 else None,
        "noise_scale_responses": groups * siblings * noise / signal if signal > 0 else None,
    }


def gradient_slices(optimizer):
    """Read pre-clipping gradients in exactly the optimizer's coordinate system.

    Do not call optimizer intake: it rescales expert buffers in place. Instead
    narrow the raw reducer buffers and apply EP-MP scaling to the diagnostic copy.
    Replica weights follow the optimizer's own DTensor mesh, not global rank count.
    """
    for group in optimizer.param_groups:
        for name, parameter in group["named_params"].items():
            if not parameter.requires_grad:
                continue
            main = optimizer.states[f"{name}.main"]
            if main.device_mesh.ndim != 1 or len(main.placements) != 1:
                raise ValueError("Probe supports the H235 one-dimensional optimizer meshes")
            grad = getattr(parameter, "_olmo_ddp_reduced_grad_shard", None)
            if grad is None:
                grad = getattr(parameter, "_main_grad_fp32", None)
                if grad is None:
                    grad = parameter.grad
                if grad is None:
                    raise ValueError(f"Missing gradient (including unsupported FP8 stores): {name}")
                grad = optimizer.narrow_tensor(grad.detach().reshape(-1), main.device_mesh, main.placements)
            if grad.numel() != main.to_local().numel():
                raise ValueError(f"Gradient does not match optimizer ownership: {name}")
            scale = 1.0
            if optimizer.moe_mesh is not None and group["pg"] == "ep_dp":
                scale /= dist.get_world_size(optimizer.ep_mp_mesh.get_group())
            replicas = main.device_mesh.size() if main.placements[0].is_replicate() else 1
            yield name, grad.detach().reshape(-1), scale, replicas


def snapshot(optimizer):
    return {
        name: (grad.float().cpu().clone(), scale, replicas)
        for name, grad, scale, replicas in gradient_slices(optimizer)
    }


def compare(optimizer, first):
    sums = torch.zeros(3, dtype=torch.float64)
    names = set()
    elements = 0.0
    for name, grad, scale, replicas in gradient_slices(optimizer):
        old, previous_scale, previous_replicas = first[name]
        if (scale, replicas) != (previous_scale, previous_replicas):
            raise ValueError("Gradient ownership changed between backwards")
        current = grad.float().cpu()
        if current.shape != old.shape:
            raise ValueError("Gradient shape changed between backwards")
        # Bounded FP64 temporaries avoid a second model-sized diagnostic copy.
        for offset in range(0, current.numel(), 1048576):
            a = old[offset : offset + 1048576].double()
            f = current[offset : offset + 1048576].double()
            sums += torch.stack((torch.dot(a, a), torch.dot(f, f), torch.dot(a, f))) * (scale**2 / replicas)
        elements += current.numel() / replicas
        names.add(name)
    if names != set(first):
        raise ValueError("Gradient parameter coverage differs between backwards")
    device = grad.device
    total = torch.cat((sums, torch.tensor([elements], dtype=torch.float64))).to(device)
    dist.all_reduce(total)
    return total.cpu().tolist()


def measure(actor, batches, objective, replay_context):
    """Wrap one native train_batch_with_loss, preserving its full-batch gradients."""
    args, module = actor.args, actor.train_module
    if args.calculate_per_token_loss or args.entropy_coef or args.kl_coef or args.use_kl_loss:
        raise ValueError("H235 probe requires response-mean policy loss without entropy/KL objectives")
    if args.olmo_core.router_aux_loss_weight or args.olmo_core.router_z_loss_weight:
        raise ValueError("H235 probe requires zero router auxiliary objectives")
    siblings = args.n_samples_per_prompt
    local_ids = [int(i) for batch in batches for i in batch["sample_indices"]]
    all_ids = [None] * dist.get_world_size()
    dist.all_gather_object(all_ids, local_ids)
    selected, groups = partition(all_ids, siblings, 17 + actor.clock.completed_steps)
    if groups * siblings != args.global_batch_size:
        raise ValueError("Probe expects exactly one optimizer batch per collection")
    originals = [batch["advantages"] for batch in batches]
    rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    started = time.perf_counter()
    try:
        for batch in batches:
            batch["advantages"] = [
                value if int(index) // siblings in selected else torch.zeros_like(value)
                for index, value in zip(batch["sample_indices"], batch["advantages"], strict=True)
            ]

        def half_objective(train_module, batch):
            loss, metrics = objective(train_module, batch)
            return loss * 2, metrics

        module.train_batch_with_loss(batches, half_objective, replay_context)
        first = snapshot(actor.optimizer)
    finally:
        for batch, advantages in zip(batches, originals, strict=True):
            batch["advantages"] = advantages
        module.zero_grads()
        torch.random.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
    probe_seconds = time.perf_counter() - started
    full_started = time.perf_counter()
    metrics = module.train_batch_with_loss(batches, objective, replay_context)
    full_seconds = time.perf_counter() - full_started
    reduction_started = time.perf_counter()
    a2, f2, af, elements = compare(actor.optimizer, first)
    del first
    row = statistics(a2, f2, af, groups, siblings)
    row.update(
        step=actor.clock.completed_steps + 1,
        optimizer_coordinates="pre-clipping; EP scaling applied; replica copies counted once",
        parameter_elements=elements,
        half_backward_and_snapshot_seconds=probe_seconds,
        full_backward_seconds=full_seconds,
        statistics_seconds=time.perf_counter() - reduction_started,
        sampled_half_group_ids=sorted(selected),
        source="fresh isolated diagnostic; actual accepted priority-sampled groups",
    )
    if dist.get_rank() == 0:
        root = Path(args.save).parent
        root.mkdir(parents=True, exist_ok=True)
        with (root / "gradient-noise.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        logger.info("Half-group gradient noise: %s", json.dumps(row, sort_keys=True))
    return metrics
