"""Diagnostic-only fixed cohorts and explicit synchronous publication cadence.

This deliberately removes the asynchronous scheduler to separate a frozen prompt
selection effect from policy age. It does not reproduce asynchronous throughput
or selection among alternative responses to the same prompt.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.miles import opd_control

from miles.rollout import sglang_rollout


def record(event, **values):
    path = Path(os.environ["OI_OPD_OUTPUT"]) / "controlled-exposure.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"event": event, "time_ns": time.time_ns(), **values}) + "\n")


def install(args, train, path):
    config = json.loads(Path(path).read_text())
    if args.fully_async or args.start_rollout_id not in (None, 0):
        raise ValueError("Controlled OPD requires a fresh synchronous run")
    if args.dynamic_sampling_filter_path or args.rollout_sample_filter_path or args.partial_rollout:
        raise ValueError("Controlled OPD disallows filtering and partial rollouts")
    if args.global_batch_size != args.rollout_batch_size * args.n_samples_per_prompt:
        raise ValueError("Controlled OPD requires one optimizer update per cohort")
    period = config["publication_period"]
    opd_control.publication_due(None, period, args.num_rollout)
    source = Path(config["cohort"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["cohort_sha256"]:
        raise ValueError("Frozen cohort changed")
    if len(source.read_text().splitlines()) != args.num_rollout * args.rollout_batch_size:
        raise ValueError("Cohort size must equal the exact run budget")
    args.oi_opd_prompt_hashes = [
        opd_control.prompt_hash(json.loads(line)["input"]) for line in source.read_text().splitlines()
    ]
    args.prompt_data = str(source)
    args.rollout_shuffle = False
    args.eval_interval = None  # All endpoints use the separate common evaluator.
    args.over_sampling_batch_size = args.rollout_batch_size
    args.rollout_function_path = "scripts.miles.opd_control_runtime.generate"
    native_create = train.__globals__["create_training_models"]
    published_step = 0

    class PublicationActor:
        def __init__(self, actor):
            self.actor = actor

        def __getattr__(self, name):
            return getattr(self.actor, name)

        async def update_weights(self, *, rollout_id=None):
            nonlocal published_step
            due = opd_control.publication_due(rollout_id, period, args.num_rollout)
            weight_version = None
            if due:
                weight_version = await self.actor.update_weights(rollout_id=rollout_id)
                published_step = 0 if rollout_id is None else rollout_id + 1
            record(
                "publication_completed" if due else "publication_skipped",
                rollout_id=rollout_id,
                served_optimizer_step=published_step,
                publication_period=period,
                weight_version=weight_version,
            )
            return weight_version

    async def create(*values, **kwargs):
        actor, critic = await native_create(*values, **kwargs)
        if critic is not None:
            raise ValueError("Controlled OPD requires an actor-only learner")
        return PublicationActor(actor), critic

    train.__globals__["create_training_models"] = create
    record("control_installed", config=config, updates=args.num_rollout, groups_per_update=args.rollout_batch_size)


def generate(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError("Use the separate common checkpoint evaluator")
    groups = data_source.get_samples(args.rollout_batch_size)
    expected = [opd_control.prompt_hash(group[0].prompt) for group in groups]
    offset = rollout_id * args.rollout_batch_size
    if expected != args.oi_opd_prompt_hashes[offset : offset + args.rollout_batch_size]:
        raise ValueError("Data source filtered, reordered or substituted the frozen cohort")
    requested = False

    def take(count):
        nonlocal requested
        if requested or count != len(groups):
            raise ValueError("Generation attempted to replace or oversample the fixed cohort")
        requested = True
        return groups

    def unused(values):
        if values:
            raise ValueError("Fixed-cohort generation aborted or left unused groups")

    output = sglang_rollout.generate_rollout(args, rollout_id, SimpleNamespace(get_samples=take, add_samples=unused))
    actual = [opd_control.prompt_hash(group[0].prompt) for group in output.samples]
    if actual != expected or any(len(group) != args.n_samples_per_prompt for group in output.samples):
        raise ValueError("Delivered cohort differs from the predetermined membership/order")
    events = [
        json.loads(line)
        for line in (Path(os.environ["OI_OPD_OUTPUT"]) / "controlled-exposure.jsonl").read_text().splitlines()
    ]
    publication = [row for row in events if row["event"] == "publication_completed"][-1]
    age = rollout_id - publication["served_optimizer_step"]
    if age != opd_control.expected_age(rollout_id, publication["publication_period"]):
        raise ValueError("Observed publication age differs from the assigned schedule")
    record(
        "cohort_delivered",
        rollout_id=rollout_id,
        prompt_hashes=actual,
        age=age,
        response_lengths=[s.response_length for group in output.samples for s in group],
        statuses=[s.status.value for group in output.samples for s in group],
    )
    return output
