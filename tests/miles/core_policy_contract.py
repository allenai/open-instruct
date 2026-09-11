"""Capture one real Core actor/Adam step on the shared immutable Megatron fixture.

This qualifies a fixed-input objective and optimizer, not online rewards/scoring.
Only the rollout provider and advantage producer are replaced by fixture inputs.
Run with the comparison olmo-miles checkout's src directory on PYTHONPATH.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss_hub import losses as policy_losses
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from olmo_core.nn.hf import convert
from olmo_core.nn.moe.v2 import olmo3
from olmo_miles.evaluation import policy_contract_schema as schema
from olmo_miles.evaluation.policy_contract_capture import validate_capture
from torch import distributed as dist
from torch.distributed.tensor import DTensor

from open_instruct.miles import actor, models, moe_models, scheduler
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.state import PolicyClock


def load_fixture(root):
    fixture = schema.validate_fixture(json.loads((root / "fixture.json").read_text()))
    digest = schema.fixture_digest(fixture)
    if (root / "fixture.sha256").read_text().strip() != digest:
        raise ValueError("Immutable fixture digest changed")
    if schema.checkpoint_inventory(root / "hf") != fixture["checkpoint"]:
        raise ValueError("Immutable HF checkpoint changed")
    return fixture, digest


def rollout_from_fixture(fixture, device):
    samples = fixture["samples"]
    return {
        "tokens": [torch.tensor(sample["tokens"], device=device) for sample in samples],
        "total_lengths": [len(sample["tokens"]) for sample in samples],
        "response_lengths": [sample["response_length"] for sample in samples],
        "loss_masks": [torch.tensor(sample["loss_mask"], dtype=torch.float32, device=device) for sample in samples],
        "rollout_log_probs": [
            torch.tensor(sample["old_log_probs"], dtype=torch.float32, device=device) for sample in samples
        ],
        "rewards": [0.0] * len(samples),
        "weight_versions": [["0"] for _ in samples],
    }


def inject_advantages(fixture, rollout):
    if [value.tolist() for value in rollout["tokens"]] != [sample["tokens"] for sample in fixture["samples"]]:
        raise ValueError("Actor changed immutable fixture sample order")
    rollout["advantages"] = [
        torch.tensor(sample["advantages"], dtype=torch.float32, device=tokens.device)
        for sample, tokens in zip(fixture["samples"], rollout["tokens"], strict=True)
    ]
    rollout["returns"] = [value.clone() for value in rollout["advantages"]]


def assert_objective_inputs(fixture, batch):
    tokens = batch["tokens"][0].tolist()
    matches = [sample for sample in fixture["samples"] if sample["tokens"] == tokens]
    if len(matches) != 1:
        raise ValueError("Objective tokens do not identify exactly one fixture sample")
    sample = matches[0]
    if batch["total_lengths"] != [len(tokens)] or batch["response_lengths"] != [sample["response_length"]]:
        raise ValueError("Objective response lengths changed")
    for key, source in (
        ("loss_masks", "loss_mask"),
        ("advantages", "advantages"),
        ("rollout_log_probs", "old_log_probs"),
    ):
        if len(batch[key]) != 1 or batch[key][0].tolist() != sample[source]:
            raise ValueError(f"Objective {key} changed")
    if len(batch["unconcat_tokens"]) != 1 or batch["unconcat_tokens"][0].tolist() != tokens:
        raise ValueError("Objective unconcat tokens changed")
    return sample


def canonical_state(worker, native):
    """Apply Core's invertible layout conversion; preserve every input dtype."""
    inventory = {name.removeprefix("module."): value for name, value in native.items()}
    if len(inventory) != len(native):
        raise ValueError("Ambiguous wrapped and unwrapped parameter names")
    q_dim = worker.hf_config.num_attention_heads * worker.hf_config.head_dim
    kv_dim = worker.hf_config.num_key_value_heads * worker.hf_config.head_dim
    for name in list(inventory):
        if name.endswith(".attention.w_qkv.weight"):
            pieces = inventory.pop(name).split((q_dim, kv_dim, kv_dim), dim=0)
            for suffix, value in zip(("w_q", "w_k", "w_v"), pieces, strict=True):
                inventory[name.replace("w_qkv", suffix)] = value
    hf = olmo3._config_for_native_dense_layout(olmo3._unwrap_model(worker.model), worker.hf_config)
    return {name: value.detach().cpu().clone() for name, value in convert.iter_olmo3moe_state_to_hf(hf, inventory)}


def optimizer_native(worker, suffix):
    result = {}
    parameters = dict(worker.model.named_parameters())
    for group in worker.optimizer.param_groups:
        for name in group["named_params"]:
            value = (
                worker.optimizer.main_grad[name] if suffix == "grad" else worker.optimizer.states[f"{name}.{suffix}"]
            )
            if isinstance(value, DTensor):
                value = value.full_tensor()
            result[name] = value.detach().reshape(parameters[name].shape).cpu().clone()
    if set(result) != set(parameters):
        raise ValueError("Optimizer does not cover every model parameter")
    return result


def configuration(root, output, fixture):
    opt = fixture["optimizer"]
    count = len(fixture["samples"])
    config = RunConfig(
        CoreConfig(
            expert_parallel_size=1,
            attention_backend="torch",
            max_sequence_length=128,
            activation_checkpointing=False,
            diagnostic_interval=1,
            router_aux_loss_weight=0.0,
            router_z_loss_weight=0.0,
        ),
        dict(
            hf_checkpoint=str(root / "hf"),
            global_batch_size=count,
            rollout_batch_size=1,
            n_samples_per_prompt=count,
            num_rollout=1,
            actor_num_gpus_per_node=1,
            debug_train_only=True,
            save=str(output / "checkpoint"),
            rollout_global_dataset=True,
            prompt_data=str(root / "prompts.jsonl"),
            lr=opt["lr"],
            lr_decay_style="constant",
            adam_beta1=opt["betas"][0],
            adam_beta2=opt["betas"][1],
            adam_eps=opt["eps"],
            weight_decay=0.0,
            clip_grad=opt["clip_grad"],
            eps_clip=fixture["eps_clip"],
            eps_clip_high=fixture["eps_clip_high"],
            seed=fixture["seed"],
            use_rollout_logprobs=True,
            entropy_coef=0.0,
            rollout_temperature=1.0,
        ),
    )

    if fixture["reduction"] == "token_mean":
        config.miles["calculate_per_token_loss"] = True
    return config


def run(root, output):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This capture qualifies world1/EP1 only")
    fixture, digest = load_fixture(root)
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous evidence")
    output.mkdir(parents=True)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    try:
        group = GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=gloo)
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
        moe_models.register_hf_classes()
        config = configuration(root, output, fixture)
        sys.argv = ["core-policy-contract", *config.arguments()]
        args = arguments.parse_args()
        if (
            args.use_rollout_routing_replay
            or args.use_kl_loss
            or args.normalize_advantages
            or args.use_opsm
            or args.recompute_loss_function
            or args.advantage_estimator != "grpo"
        ):
            raise ValueError("Fixture requires GRPO with replay/KL/normalization/OPSM/loss recomputation disabled")
        (output / "arguments.json").write_text(json.dumps(config.arguments(), indent=2))
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = args
        worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
        worker.model, worker.optimizer = worker.train_module.model, worker.train_module.optim
        if getattr(worker.hf_config, "attention_dropout", 0) != 0 or any(
            isinstance(module, torch.nn.Dropout) and module.p != 0 for module in worker.model.modules()
        ):
            raise ValueError("Fixed-input comparison requires zero model dropout")
        if args.calculate_per_token_loss != (fixture["reduction"] == "token_mean"):
            raise ValueError("Parsed objective reduction differs from immutable fixture")
        worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
        worker.clock = PolicyClock()
        worker.clock.published()
        worker.ref_module = None
        worker._heartbeat = SimpleNamespace(bump=lambda: None)
        rollout = rollout_from_fixture(fixture, "cuda")
        snapshots = {
            "initial_weights": models.export_state(worker.train_module, worker.hf_config),
            "initial_masters": canonical_state(worker, optimizer_native(worker, "main")),
        }
        clip_norms, consumed, shifted_scores = [], [], []
        original_clip = worker.optimizer._clip_grad

        def clip():
            snapshots["preclip_gradients"] = canonical_state(worker, optimizer_native(worker, "grad"))
            norm = original_clip()
            snapshots["postclip_gradients"] = canonical_state(worker, optimizer_native(worker, "grad"))
            clip_norms.append(float(norm))
            return norm

        original_loss = actor.miles_loss.loss_function
        original_logprobs = policy_losses.get_log_probs_and_entropy

        def loss(loss_args, batch, microbatches, logits, **kwargs):
            sample = assert_objective_inputs(fixture, batch)
            consumed.append(sample["id"])
            return original_loss(loss_args, batch, microbatches, logits, **kwargs)

        def logprobs(logits, **kwargs):
            result = original_logprobs(logits, **kwargs)
            tokens = kwargs["unconcat_tokens"][0]
            response = kwargs["response_lengths"][0]
            expected = logits[0, -response - 1 : -1].detach().float().log_softmax(-1)
            expected = expected.gather(-1, tokens[-response:].unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(result["log_probs"][0].detach(), expected, rtol=1e-5, atol=1e-5)
            shifted_scores.append(
                {
                    "response_target_ids": tokens[-response:].tolist(),
                    "logit_positions": list(range(len(tokens) - response - 1, len(tokens) - 1)),
                    "actual_log_probs": result["log_probs"][0].detach().cpu().tolist(),
                }
            )
            return result

        with (
            mock.patch.object(actor.miles_loss, "loss_function", side_effect=loss),
            mock.patch.object(policy_losses, "get_log_probs_and_entropy", side_effect=logprobs),
            mock.patch.object(worker.optimizer, "_clip_grad", side_effect=clip),
            mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo),
            mock.patch.object(actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())),
            mock.patch.object(
                actor.miles_loss,
                "compute_advantages_and_returns",
                side_effect=lambda _args, batch: inject_advantages(fixture, batch),
            ),
        ):
            worker.train(0, None)
        if consumed != [sample["id"] for sample in fixture["samples"]] or len(shifted_scores) != len(consumed):
            raise ValueError("Actual objective did not consume each fixture sample exactly once")
        if worker.clock.completed_steps != 1 or len(clip_norms) != 1:
            raise ValueError("Expected exactly one real actor optimizer step and clip operation")
        for category, suffix in (("exp_avg", "exp_avg"), ("exp_avg_sq", "exp_avg_sq"), ("final_masters", "main")):
            snapshots[category] = canonical_state(worker, optimizer_native(worker, suffix))
        snapshots["final_weights"] = models.export_state(worker.train_module, worker.hf_config)
        payload = {
            "schema_version": 1,
            "fixture_sha256": digest,
            "backend": "core",
            "world_size": 1,
            "tensors": snapshots,
            "optimizer_gradient_norm": clip_norms[0],
        }
        if set(snapshots) != set(schema.CATEGORIES):
            raise ValueError("Incomplete capture categories")
        torch.save(payload, output / "capture.pt")
        validation = validate_capture(payload, fixture)
        report = {
            "fixture_sha256": digest,
            "validation": validation,
            "consumed_sample_ids": consumed,
            "shifted_scores": shifted_scores,
            "optimizer_steps": worker.clock.completed_steps,
            "production_optimizer": type(worker.optimizer).__name__,
            "preclip_norm": clip_norms[0],
            "schema_source_sha256": hashlib.sha256(Path(schema.__file__).read_bytes()).hexdigest(),
            "core_actor_source_sha256": hashlib.sha256(Path(actor.__file__).read_bytes()).hexdigest(),
            "core_factory_source_sha256": hashlib.sha256(Path(olmo3.__file__).read_bytes()).hexdigest(),
            "model_config": worker.model_config.as_dict(),
            "fixture": copy.deepcopy(fixture),
            "objective_inputs": "immutable advantages/rollout old_log_probs, actual current Core forward",
            "limitations": "EP1 fixed-input objective/optimizer; excludes rewards, generation, and replay",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({"output": str(output), "optimizer_steps": 1, "fixture_sha256": digest}))
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
