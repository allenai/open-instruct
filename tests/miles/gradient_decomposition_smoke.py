"""Actual EP1 Core/Megatron gradient-only qualification on immutable fixture02.

Set GRADIENT_PROBE_BACKEND, GRADIENT_PROBE_FIXTURE, GRADIENT_PROBE_OUTPUT.
Megatron must use its normal compat entrypoint and fixture miles-args.json.
"""

import importlib
import json
import os
import sys
from dataclasses import replace
from importlib import util
from pathlib import Path
from types import SimpleNamespace

import torch
from scripts.miles import update_zero_gradient_capture as capture
from torch import distributed as dist


def core_worker(root, output, fixture):
    spec = util.spec_from_file_location("core_policy_fixture", Path(__file__).with_name("core_policy_contract.py"))
    helper = util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    gloo = helper.actor.distributed_utils.init_gloo_group()
    group = helper.GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=gloo)
    trivial = helper.GroupInfo(rank=0, size=1, group=None)
    helper.parallel.set_parallel_state(
        helper.parallel.ParallelState(
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
    helper.moe_models.register_hf_classes()
    config = helper.configuration(root, output, fixture)
    config = replace(config, core=replace(config.core, diagnostic_interval=0))
    config.miles["rollout_max_context_len"] = 128
    sys.argv = ["gradient-probe", *config.arguments()]
    args = helper.arguments.parse_args()
    worker = helper.actor.OLMoCoreTrainRayActor.__new__(helper.actor.OLMoCoreTrainRayActor)
    worker.args = args
    worker.train_module, worker.hf_config, worker.model_config = helper.models.build_train_module(args)
    worker.model, worker.optimizer = worker.train_module.model, worker.train_module.optim
    worker.lr_scheduler = helper.scheduler.CoreLRScheduler(args, worker.optimizer)
    worker.clock = helper.PolicyClock()
    worker.clock.published()
    worker.ref_module = None
    worker._heartbeat = SimpleNamespace(bump=lambda: None)
    return worker


def megatron_worker():
    arguments = importlib.import_module("miles.utils.arguments")
    initialize = importlib.import_module("miles.backends.megatron_utils.initialize")
    model_utils = importlib.import_module("miles.backends.megatron_utils.model")
    actor = importlib.import_module("miles.backends.megatron_utils.actor")
    args = arguments.parse_args()
    args.rank = 0
    args.use_distributed_optimizer = True
    args.overlap_grad_reduce = False
    args.overlap_param_gather = False
    args.use_wandb = False
    args.use_tensorboard = False
    initialize.init(args)
    model, optimizer, scheduler, _ = model_utils.initialize_model_and_optimizer(args)
    worker = actor.MegatronTrainRayActor.__new__(actor.MegatronTrainRayActor)
    worker.args, worker.model, worker.optimizer, worker.opt_param_scheduler = args, model, optimizer, scheduler
    return worker


def main():
    root, output = Path(os.environ["GRADIENT_PROBE_FIXTURE"]), Path(os.environ["GRADIENT_PROBE_OUTPUT"])
    fixture = json.loads((root / "fixture.json").read_text())
    schema = importlib.import_module("olmo_miles.evaluation.policy_contract_schema")
    if schema.fixture_digest(fixture) != (root / "fixture.sha256").read_text().strip():
        raise ValueError("Immutable fixture digest changed")
    if schema.checkpoint_inventory(root / "hf") != fixture["checkpoint"]:
        raise ValueError("Immutable HF checkpoint inventory changed")
    fixture["auxiliary"] = {"lb": 0.01, "z": 1e-5}
    payload = {
        "schema_version": 1,
        "source": "immutable-policy-fixture02",
        "auxiliary": fixture["auxiliary"],
        "cases": [
            {
                "case_id": x["id"],
                "input_ids": x["tokens"],
                "response_length": x["response_length"],
                "loss_mask": x["loss_mask"],
                "old_log_probs": x["old_log_probs"],
                "advantages": x["advantages"],
            }
            for x in fixture["samples"]
        ],
    }
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    try:
        worker = (
            core_worker(root, output, fixture)
            if os.environ["GRADIENT_PROBE_BACKEND"] == "olmo_core"
            else megatron_worker()
        )
        report = capture.diagnostic_gradient_probe(worker, payload, output)
        for arm in ("policy", "auxiliary", "combined"):
            if not any(row[f"{arm}_norm"] > 0 for row in report["router_gradients"].values()):
                raise ValueError(f"Tiny fixture failed to exercise nonzero {arm} router gradients")
        print(json.dumps(report))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
