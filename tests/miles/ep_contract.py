"""Same fixed hybrid-MoE batch through native Core EP1/EP2; no serving or downloads.

First Adam moments are compared, since first-step Adam parameter updates alone
can conceal uniform gradient scaling errors. All input weights are random.
"""

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from miles.backends.training_utils import parallel
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from torch import distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from open_instruct.miles import actor, data, models, scheduler
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.state import PolicyClock


def bootstrap(root):
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(173)
    _register_olmo3moe_auto_classes()
    hf = Olmo3MoeConfig(
        vocab_size=256,
        hidden_size=128,
        attention_hidden_size=128,
        head_dim=64,
        dense_mlp_intermediate_size=256,
        dense_mlp_uses_shared_experts=True,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_head_qk_norm=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        latent_moe_dim=64,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        use_peri_ln=True,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(hf).to(torch.bfloat16)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                value.zero_()
    model.save_pretrained(root / "hf")
    (root / "prompts.jsonl").write_text('{"input": "fixture", "label": "fixture"}\n')


def rollout_for(rank, world, mode):
    result = {
        key: [] for key in ("tokens", "total_lengths", "response_lengths", "loss_masks", "rewards", "weight_versions")
    }
    for index in range(rank, 4, world):
        length = 8 + index * 2
        response = 3 + index
        result["tokens"].append((torch.arange(length, device="cuda") + 11 * index) % 256)
        result["total_lengths"].append(length)
        result["response_lengths"].append(response)
        mask = torch.ones(response, device="cuda")
        mask[1] = 0
        result["loss_masks"].append(mask)
        result["rewards"].append(0.0 if mode == "auxiliary" else [1.0, -1.0, 0.5, -0.5][index])
        result["weight_versions"].append(["0"])
    return result


def full_optimizer_state(worker, *, gradients=False):
    """Gather optimizer-DP shards, then expert-MP shards in native expert order."""
    result = {}
    for group in worker.optimizer.param_groups:
        for name in group["named_params"]:
            owner = worker.model.get_submodule(name.rsplit(".", 1)[0])
            for suffix in ("grad",) if gradients else ("exp_avg", "exp_avg_sq", "main"):
                if gradients:
                    template = worker.optimizer.states[f"{name}.main"]
                    value = DTensor.from_local(
                        worker.optimizer.main_grad[name].detach().clone(),
                        device_mesh=template.device_mesh,
                        placements=template.placements,
                        shape=template.shape,
                        stride=template.stride(),
                    )
                else:
                    value = worker.optimizer.states[f"{name}.{suffix}"]
                value = value.full_tensor() if isinstance(value, DTensor) else value
                value = value.detach().reshape(-1).contiguous()
                if getattr(owner, "_ep_sharded", False):
                    pg = owner.ep_mesh["ep_mp"].get_group()
                    pieces = [torch.empty_like(value) for _ in range(dist.get_world_size(pg))]
                    dist.all_gather(pieces, value, group=pg)
                    value = torch.cat(pieces)
                result[f"{name}.{suffix}"] = value.cpu()
    return result


def run(root, mode, checkpointing, *, token_average=False, clip_grad=1e9, capture_gradients=False):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    try:
        group = GroupInfo(rank=rank, size=world, group=dist.group.WORLD, gloo_group=gloo)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=group,
                intra_dp_cp=group,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        config = RunConfig(
            CoreConfig(
                expert_parallel_size=world,
                attention_backend="torch",
                max_sequence_length=128,
                activation_checkpointing=checkpointing,
                diagnostic_interval=1,
                router_aux_loss_weight=0.0 if mode == "policy" else 0.01,
                router_z_loss_weight=0.0 if mode == "policy" else 1e-5,
            ),
            dict(
                hf_checkpoint=str(root / "hf"),
                global_batch_size=4,
                rollout_batch_size=1,
                n_samples_per_prompt=4,
                num_rollout=2,
                actor_num_gpus_per_node=world,
                debug_train_only=True,
                save=str(root / f"ep{world}-{mode}-ac{int(checkpointing)}"),
                rollout_global_dataset=True,
                prompt_data=str(root / "prompts.jsonl"),
                lr=1e-4,
                clip_grad=clip_grad,
                use_rollout_routing_replay=True,
                use_miles_router=True,
            ),
        )
        if token_average:
            config.miles["calculate_per_token_loss"] = True
        sys.argv = ["ep-contract", *config.arguments()]
        args = arguments.parse_args()
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = args
        worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
        worker.model = worker.train_module.model
        worker.optimizer = worker.train_module.optim
        worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
        worker.clock = PolicyClock()
        worker.clock.published()
        worker.ref_module = None
        worker._heartbeat = SimpleNamespace(bump=lambda: None)
        rollout = rollout_for(rank, world, mode)
        route_path = root / "routes.pt"
        if world == 1 and not route_path.exists():
            observed = []

            def capture(module, inputs, output):
                observed.append(output[1].detach().clone())

            router = next(
                module for name, module in worker.model.named_modules() if name.endswith("routed_experts_router")
            )
            handle = router.register_forward_hook(capture)
            worker._score(worker.train_module, data.sample_batches(rollout, 128), use_replay=False)
            handle.remove()
            routes = []
            for ids in observed:
                packed = torch.zeros(ids.shape[1] - 1, 2, 2, dtype=torch.long)
                packed[:, 0, 1] = 1
                packed[:, 1] = ids[0, :-1].cpu()
                routes.append(packed)
            assert len(routes) == 4
            torch.save(routes, route_path)
        all_routes = torch.load(route_path, weights_only=True)
        rollout["rollout_routed_experts"] = [all_routes[i].cuda() for i in range(rank, 4, world)]
        rollout["rollout_log_probs"] = worker._score(
            worker.train_module, data.sample_batches(rollout, 128), use_replay=True
        )
        gradient_evidence = {}
        original_clip = worker.optimizer._clip_grad

        def capture_clip():
            gradient_evidence["before"] = full_optimizer_state(worker, gradients=True)
            norm = original_clip()
            gradient_evidence["after"] = full_optimizer_state(worker, gradients=True)
            gradient_evidence["norm"] = float(norm)
            return norm

        clip_context = (
            mock.patch.object(worker.optimizer, "_clip_grad", side_effect=capture_clip)
            if capture_gradients
            else contextlib.nullcontext()
        )
        with (
            clip_context,
            mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo),
            mock.patch.object(actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())),
        ):
            worker.train(0, None)
        state = full_optimizer_state(worker)
        if capture_gradients:
            gradient_evidence.update(
                state=state,
                clip_grad=clip_grad,
                token_average=token_average,
                betas=worker.optimizer.param_groups[0]["betas"],
                rank=rank,
                world=world,
            )
            torch.save(gradient_evidence, root / f"stress-ep{world}-rank{rank}.pt")
        if rank == 0:
            torch.save(state, root / f"ep{world}-{mode}-ac{int(checkpointing)}.pt")
    finally:
        dist.destroy_process_group()


def _category(name):
    return "router" if "router" in name else "expert" if "experts" in name else "dense"


def _state_errors(reference, actual):
    """Return finite aggregate evidence or a structural error, without raising."""
    if set(reference) != set(actual):
        return {}, "state keys differ"
    groups = {}
    for name, expected in reference.items():
        value = actual[name]
        if value.shape != expected.shape:
            return {}, f"shape differs for {name}"
        if not bool(torch.isfinite(value).all() and torch.isfinite(expected).all()):
            return {}, f"non-finite state for {name}"
        group = groups.setdefault(_category(name) + "/" + name.rsplit(".", 1)[1], [0.0, 0.0, 0.0])
        delta = value.double() - expected.double()
        group[0] += float(delta.square().sum())
        group[1] += float(expected.double().square().sum())
        group[2] = max(group[2], float(delta.abs().max()))
    return {
        name: {
            "relative_l2_error": math.sqrt(a / max(b, 1e-20)),
            "difference_l2": math.sqrt(a),
            "reference_l2": math.sqrt(b),
            "max_abs_error": c,
        }
        for name, (a, b, c) in groups.items()
    }, None


def _moment_signals(state):
    groups = {}
    for name, value in state.items():
        if not name.endswith(".exp_avg"):
            continue
        group = groups.setdefault(_category(name), {"sum_squares": 0.0, "elements": 0})
        if not bool(torch.isfinite(value).all()):
            return {}, f"non-finite first moment for {name}"
        group["sum_squares"] += float(value.double().square().sum())
        group["elements"] += value.numel()
    return {
        name: {"l2": math.sqrt(values["sum_squares"]), "elements": values["elements"]}
        for name, values in groups.items()
    }, None


def compare(root):
    tolerance = 0.05
    report = dict(
        passed=False,
        relative_l2_tolerance=tolerance,
        comparisons=[],
        first_moment_signals=[],
        first_moment_superposition=[],
        failures=[],
        notes=[
            "Master parameter comparisons measure state agreement, not relative update agreement.",
            "Superposition divides residual L2 by policy L2 + auxiliary L2 to remain stable near cancellation.",
            "Nonzero first moments reject disconnected native policy or auxiliary router gradients.",
        ],
    )
    states = {}
    settings = ((1, False), (1, True), (2, False), (2, True))
    for mode in ("policy", "auxiliary", "combined"):
        for world, checkpointing in settings:
            key = (mode, world, checkpointing)
            path = root / f"ep{world}-{mode}-ac{int(checkpointing)}.pt"
            try:
                states[key] = torch.load(path, weights_only=True)
            except (OSError, RuntimeError) as exc:
                report["failures"].append(f"Cannot load {path.name}: {exc}")
                continue
            signal, error = _moment_signals(states[key])
            report["first_moment_signals"].append(
                dict(mode=mode, world=world, activation_checkpointing=checkpointing, categories=signal, error=error)
            )
            if error:
                report["failures"].append(f"{key}: {error}")
            if mode in ("policy", "auxiliary") and signal.get("router", {}).get("l2", 0.0) <= 0:
                report["failures"].append(f"{key}: missing or zero router first-moment signal")
        reference = states.get((mode, 1, False))
        if reference is None:
            continue
        for world, checkpointing in settings[1:]:
            actual = states.get((mode, world, checkpointing))
            if actual is None:
                continue
            measured, error = _state_errors(reference, actual)
            report["comparisons"].append(
                dict(mode=mode, world=world, activation_checkpointing=checkpointing, errors=measured, error=error)
            )
            if error:
                report["failures"].append(f"{mode}, EP{world}, checkpointing={checkpointing}: {error}")
            for name, values in measured.items():
                if values["relative_l2_error"] >= tolerance:
                    report["failures"].append(
                        f"{mode}, EP{world}, checkpointing={checkpointing}: {name} state error exceeds tolerance"
                    )
    for world, checkpointing in settings:
        modes = [states.get((mode, world, checkpointing)) for mode in ("policy", "auxiliary", "combined")]
        if any(state is None for state in modes):
            continue
        policy, auxiliary, combined = [
            {name: value.double() for name, value in state.items() if name.endswith(".exp_avg")} for state in modes
        ]
        if set(policy) != set(auxiliary) or any(policy[name].shape != auxiliary[name].shape for name in policy):
            report["failures"].append(
                f"EP{world}, checkpointing={checkpointing}: component first-moment shapes differ"
            )
            continue
        expected = {name: policy[name] + auxiliary[name] for name in policy}
        measured, error = _state_errors(expected, combined)
        component_signals = [_moment_signals(state)[0] for state in (policy, auxiliary)]
        if error:
            report["failures"].append(f"EP{world}, checkpointing={checkpointing}: superposition {error}")
        for name, values in measured.items():
            category = name.split("/")[0]
            denominator = sum(signal[category]["l2"] for signal in component_signals)
            values["component_l2_sum"] = denominator
            values["relative_component_l2_error"] = values["difference_l2"] / max(denominator, 1e-10)
            if values["relative_component_l2_error"] >= tolerance:
                report["failures"].append(
                    f"EP{world}, checkpointing={checkpointing}: {name} superposition error exceeds tolerance"
                )
        report["first_moment_superposition"].append(
            dict(world=world, activation_checkpointing=checkpointing, errors=measured, error=error)
        )
    report["passed"] = not report["failures"]
    (root / "ep-contract.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("EP_CONTRACT_PASSED" if report["passed"] else "EP_CONTRACT_FAILED", json.dumps(report, allow_nan=False))
    if not report["passed"]:
        raise AssertionError("; ".join(report["failures"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "run", "compare"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=("policy", "auxiliary", "combined"), default="combined")
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap(args.root)
    elif args.command == "run":
        run(args.root, args.mode, args.checkpointing)
    else:
        compare(args.root)


if __name__ == "__main__":
    main()
