"""Same fixed hybrid-MoE batch through native Core EP1/EP2; no serving or downloads.

First Adam moments are compared, since first-step Adam parameter updates alone
can conceal uniform gradient scaling errors. All input weights are random.
"""

import argparse
import contextlib
import json
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


def full_optimizer_state(worker):
    """Gather optimizer-DP shards, then expert-MP shards in native expert order."""
    result = {}
    for group in worker.optimizer.param_groups:
        for name in group["named_params"]:
            owner = worker.model.get_submodule(name.rsplit(".", 1)[0])
            for suffix in ("exp_avg", "exp_avg_sq", "main"):
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


def run(root, mode, checkpointing):
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
                clip_grad=1e9,
                use_rollout_routing_replay=True,
            ),
        )
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
        with (
            mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo),
            mock.patch.object(actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())),
        ):
            worker.train(0, None)
        state = full_optimizer_state(worker)
        if rank == 0:
            torch.save(state, root / f"ep{world}-{mode}-ac{int(checkpointing)}.pt")
    finally:
        dist.destroy_process_group()


def compare(root):
    report = []
    for mode in ("policy", "auxiliary", "combined"):
        reference = torch.load(root / f"ep1-{mode}-ac0.pt", weights_only=True)
        for world, checkpointing in ((1, True), (2, False), (2, True)):
            actual = torch.load(root / f"ep{world}-{mode}-ac{int(checkpointing)}.pt", weights_only=True)
            assert set(reference) == set(actual)
            groups = {}
            for name, expected in reference.items():
                value = actual[name]
                assert value.shape == expected.shape and torch.isfinite(value).all()
                category = "router" if "router" in name else "expert" if "experts" in name else "dense"
                suffix = name.rsplit(".", 1)[1]
                group = groups.setdefault(category + "/" + suffix, [0.0, 0.0, 0.0])
                group[0] += float((value.double() - expected.double()).square().sum())
                group[1] += float(expected.double().square().sum())
                group[2] = max(group[2], float((value - expected).abs().max()))
            measured = {
                name: {"relative_l2_error": (a / max(b, 1e-20)) ** 0.5, "max_abs_error": c}
                for name, (a, b, c) in groups.items()
            }
            # BF16 distributed kernel ordering can differ; moments catch factor-of-world scaling errors.
            for name, values in measured.items():
                assert values["relative_l2_error"] < 0.05, (mode, world, checkpointing, name, values)
            report.append(dict(mode=mode, world=world, activation_checkpointing=checkpointing, errors=measured))
    (root / "ep-contract.json").write_text(
        json.dumps({"passed": True, "relative_l2_tolerance": 0.05, "comparisons": report}, indent=2)
    )
    print("EP_CONTRACT_PASSED", json.dumps(report))


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
