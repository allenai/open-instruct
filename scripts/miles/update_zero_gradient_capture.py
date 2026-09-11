"""Diagnostic-only policy/auxiliary router gradients at native optimizer boundaries.

No optimization takes place. Each arm uses the actual native training path and
stops after reduction/intake, before clipping or parameter/moment mutation.
"""

import contextlib
import copy
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from unittest import mock

import torch
from torch import distributed as dist


class CapturedBoundary(Exception):
    """Successful diagnostic stop, raised before any update."""


def local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def tensor_hash(value):
    value = local(value.detach()).contiguous().reshape(-1).view(torch.uint8)
    result = hashlib.sha256()
    for part in value.split(16 * 1024 * 1024):
        result.update(part.cpu().numpy().tobytes())
    return result.hexdigest()


def tree_hash(value):
    if isinstance(value, torch.Tensor):
        return [str(value.dtype), list(value.shape), tensor_hash(value)]
    if isinstance(value, dict):
        return {str(k): tree_hash(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_hash(x) for x in value]
    return repr(value)


def models(worker):
    return worker.model if isinstance(worker.model, list) else [worker.model]


def state_fingerprint(worker):
    """Hash all model parameters/buffers and persistent optimizer tensors, bounded copies."""
    parameters = {}
    for index, model in enumerate(models(worker)):
        parameters.update({f"{index}:{name}": tree_hash(v) for name, v in model.state_dict().items()})
    optim = worker.optimizer
    if worker.args.train_backend == "olmo_core":
        optimizer = tree_hash(optim.states)
    else:
        optimizer = []
        for child in getattr(optim, "chained_optimizers", [optim]):
            optimizer.append(
                {
                    "masters": tree_hash(child.shard_fp32_from_float16_groups),
                    "states": tree_hash(list(child.optimizer.state.values())),
                }
            )
    scheduler = getattr(worker, "lr_scheduler", getattr(worker, "opt_param_scheduler", None))
    return {
        "model": parameters,
        "optimizer": optimizer,
        "scheduler": tree_hash(scheduler.state_dict()) if scheduler else None,
        "clock": copy.deepcopy(vars(worker.clock)) if hasattr(worker, "clock") else None,
    }


def router_gradients(worker):
    """Preserve optimizer-owned Core shards or reduced Megatron model gradients."""
    result = {}
    backend = worker.args.train_backend
    for model in models(worker):
        for name, parameter in model.named_parameters():
            if not name.endswith(("routed_experts_router.weight", ".mlp.router.weight")):
                continue
            grad = worker.optimizer.main_grad.get(name) if backend == "olmo_core" else parameter.main_grad
            if grad is None:
                continue  # Core may assign this parameter to another optimizer rank.
            value = local(grad.detach())
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid optimizer-consumed router gradient: {name}")
            result[name] = {
                "gradient": value.cpu().clone(),
                "parameter_shape": list(parameter.shape),
                "gradient_shape": list(grad.shape),
                "placements": repr(getattr(grad, "placements", None)),
                "parameter_sha256": tensor_hash(parameter),
            }
    return result


@contextlib.contextmanager
def coefficients(worker, lb, z):
    """Temporarily select objective coefficients, preserving native auxiliary formulas."""
    with contextlib.ExitStack() as stack:
        seen = set()
        for model in models(worker):
            for name, module in model.named_modules():
                if worker.args.train_backend == "olmo_core" and name.endswith(".routed_experts_router"):
                    if module.global_load_balancing or module.orth_loss_weight:
                        raise ValueError("Gradient probe requires the original local LB/z recipe")
                    stack.enter_context(mock.patch.object(module, "lb_loss_weight", lb))
                    stack.enter_context(mock.patch.object(module, "z_loss_weight", z))
                elif worker.args.train_backend == "megatron" and name.endswith(".mlp.router"):
                    config = module.config
                    if id(config) in seen:
                        continue
                    seen.add(id(config))
                    stack.enter_context(mock.patch.object(config, "moe_aux_loss_coeff", lb))
                    stack.enter_context(mock.patch.object(config, "moe_z_loss_coeff", z))
        yield


@contextlib.contextmanager
def stop_at_boundary(worker, sink):
    """Both primary capture and independent update guards are restored on every exit."""
    optimizer = worker.optimizer
    calls = []

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Diagnostic attempted a forbidden optimizer/scheduler update")

    def capture(*_args, **_kwargs):
        if calls:
            raise ValueError("Expected exactly one finalized optimizer boundary")
        calls.append(True)
        sink.update(router_gradients(worker))
        raise CapturedBoundary()

    with contextlib.ExitStack() as stack:
        if worker.args.train_backend == "olmo_core":
            stack.enter_context(mock.patch.object(optimizer, "_clip_grad", side_effect=capture))
            stack.enter_context(mock.patch.object(optimizer, "_step_foreach", side_effect=forbidden))
        else:
            stack.enter_context(mock.patch.object(optimizer, "step", side_effect=capture))
            for child in getattr(optimizer, "chained_optimizers", [optimizer]):
                stack.enter_context(mock.patch.object(child.optimizer, "step", side_effect=forbidden))
        scheduler = getattr(worker, "lr_scheduler", getattr(worker, "opt_param_scheduler", None))
        if scheduler:
            stack.enter_context(mock.patch.object(scheduler, "step", side_effect=forbidden))
        yield calls


def validate_payload(payload, world, context):
    module = importlib.import_module(
        "update_zero_training_capture" if __package__ in (None, "") else "scripts.miles.update_zero_training_capture"
    )
    cases = module.validate_payload(payload, world, context)
    for case in cases:
        for key in ("old_log_probs", "advantages"):
            values = case[key]
            if len(values) != case["response_length"] or not all(math.isfinite(float(x)) for x in values):
                raise ValueError(f"Invalid explicit {key} response axis")
    if payload.get("auxiliary") != {"lb": 0.01, "z": 1e-5}:
        raise ValueError("Require explicit original LB .01 and z 1e-5 coefficients")
    return cases


def rollout_from_cases(cases, arm):
    return {
        "tokens": [torch.tensor(x["input_ids"], device="cuda") for x in cases],
        "total_lengths": [len(x["input_ids"]) for x in cases],
        "response_lengths": [x["response_length"] for x in cases],
        "loss_masks": [torch.tensor(x["loss_mask"], device="cuda", dtype=torch.float32) for x in cases],
        "rollout_log_probs": [torch.tensor(x["old_log_probs"], device="cuda", dtype=torch.float32) for x in cases],
        "advantages": [
            torch.tensor(x["advantages"], device="cuda", dtype=torch.float32) * (0 if arm == "auxiliary" else 1)
            for x in cases
        ],
        "rewards": [0.0 for _ in cases],
        "weight_versions": [["0"] for _ in cases],
    }


def assert_objective_batch(batch, rollout, index):
    if batch["total_lengths"] != [rollout["total_lengths"][index]] or batch["response_lengths"] != [
        rollout["response_lengths"][index]
    ]:
        raise ValueError("Native objective changed sample or response axes")
    if not torch.equal(batch["unconcat_tokens"][0], rollout["tokens"][index]):
        raise ValueError("Native objective changed immutable token IDs")
    for name in ("loss_masks", "advantages", "rollout_log_probs"):
        if not torch.equal(batch[name][0], rollout[name][index]):
            raise ValueError(f"Native objective changed fixed {name}")


@contextlib.contextmanager
def observe_objective(module, rollout):
    consumed = []
    original = module.loss_function

    def checked(args, batch, *other, **kwargs):
        assert_objective_batch(batch, rollout, len(consumed))
        consumed.append(len(consumed))
        return original(args, batch, *other, **kwargs)

    with mock.patch.object(module, "loss_function", side_effect=checked):
        try:
            yield
        finally:
            if sys.exc_info()[0] in (None, CapturedBoundary) and len(consumed) != len(rollout["tokens"]):
                raise ValueError("Native objective did not consume every sample exactly once")


def native_pass(worker, rollout):
    if worker.args.train_backend == "olmo_core":
        actor = importlib.import_module("open_instruct.miles.actor")
        advantages = rollout["advantages"]

        def inject(_args, batch):
            batch["advantages"] = advantages
            batch["returns"] = [x.clone() for x in advantages]

        with (
            mock.patch.object(actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())),
            mock.patch.object(actor.miles_loss, "compute_advantages_and_returns", side_effect=inject),
            observe_objective(actor.miles_loss, rollout),
        ):
            worker.train(0, None)
    else:
        module = importlib.import_module("miles.backends.megatron_utils.model")
        data = importlib.import_module("miles.backends.training_utils.data")
        rollout["max_seq_lens"] = [max(rollout["total_lengths"])] * len(rollout["tokens"])
        rollout["log_probs"] = rollout["rollout_log_probs"]
        rollout["returns"] = [x.clone() for x in rollout["advantages"]]
        with observe_objective(module, rollout):
            module.train(
                0,
                worker.model,
                worker.optimizer,
                worker.opt_param_scheduler,
                [data.DataIterator(rollout, micro_batch_size=1)],
                [len(rollout["tokens"])],
                [worker.args.global_batch_size],
                witness_info=None,
                attempt=0,
            )


def decomposition(arms):
    keys = set(arms["policy"])
    if any(set(values) != keys for values in arms.values()):
        raise ValueError("Router gradient ownership changed between objective arms")
    records = {}
    for name in sorted(keys):
        p, a, c = [arms[key][name]["gradient"].double().reshape(-1) for key in ("policy", "auxiliary", "combined")]
        residual = c - p - a
        denominator = float(p.norm() * a.norm())
        records[name] = {
            "policy_norm": float(p.norm()),
            "auxiliary_norm": float(a.norm()),
            "combined_norm": float(c.norm()),
            "policy_auxiliary_cosine": float(p.dot(a)) / denominator if denominator else None,
            "superposition_residual_norm": float(residual.norm()),
            "superposition_relative_to_combined": float(residual.norm()) / max(float(c.norm()), 1e-30),
        }
    return records


def diagnostic_gradient_probe(worker, payload, output):
    backend = worker.args.train_backend
    rank, world = dist.get_rank(), dist.get_world_size()
    # EP2 capture needs a dedicated reduced-ownership qualification before enabling.
    if world != 1:
        raise ValueError("Gradient diagnostic currently qualifies world1 only; EP2 is not yet qualified")
    if (
        worker.args.use_routing_replay
        or worker.args.use_rollout_routing_replay
        or not worker.args.use_rollout_logprobs
    ):
        raise ValueError("Require replay off and fixed serving policy anchors")
    if backend == "megatron" and float(worker.optimizer.get_loss_scale()) != 1.0:
        raise ValueError("Preclip Megatron capture requires unit loss scaling")
    if worker.args.global_batch_size != len(payload["cases"]):
        raise ValueError("Explicit cohort must contain exactly one full optimizer batch")
    cases = validate_payload(payload, world, worker.args.rollout_max_context_len)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    before = state_fingerprint(worker)
    arms = {}
    for arm in ("policy", "auxiliary", "combined"):
        rollout = rollout_from_cases(cases, arm)
        captured = {}
        with coefficients(worker, 0.0 if arm == "policy" else 0.01, 0.0 if arm == "policy" else 1e-5):
            with stop_at_boundary(worker, captured) as calls:
                try:
                    native_pass(worker, rollout)
                except CapturedBoundary:
                    pass
                else:
                    raise ValueError("Production training did not reach the guarded optimizer boundary")
            if len(calls) != 1 or not captured:
                raise ValueError("Missing finalized router gradient capture")
        arms[arm] = captured
        torch.save(captured, path / f"{arm}-rank{rank}.pt")
        if state_fingerprint(worker) != before:
            raise ValueError("Model/optimizer/scheduler/clock changed during gradient-only diagnostic")
    report = {
        "backend": backend,
        "world_size": world,
        "rank": rank,
        "optimizer_calls": 0,
        "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_unchanged": True,
        "router_gradients": decomposition(arms),
        "scope": "Finalized, preclip native optimizer input; EP1 only; native auxiliary padding and normalization preserved",
    }
    (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
