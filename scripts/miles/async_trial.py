"""Four-update live scheduling check on the frozen SFT/GSM8K preparation."""

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import ray
import torch
from miles.utils import arguments
from scripts.miles import analyze_gsm8k_parity as evidence
from scripts.miles import gsm8k_parity, prepare_gsm8k_parity

from open_instruct.miles.driver import train

UPDATES = 4


def configuration(campaign, output, *, asynchronous):
    config = gsm8k_parity.configuration(campaign)
    config = dataclasses.replace(
        config, core=dataclasses.replace(config.core, max_policy_lag=int(asynchronous), diagnostic_interval=1)
    )
    for key in list(config.miles):
        if key.startswith(("eval_", "wandb_")) or key in ("n_samples_per_eval_prompt", "use_wandb"):
            del config.miles[key]
    config.miles.update(
        use_rollout_logprobs=True,
        num_rollout=UPDATES,
        lr_decay_iters=UPDATES,
        sglang_cuda_graph_backend_decode="disabled",
        save=str(output / "metrics"),
        save_debug_rollout_data=str(output / "rollouts/{rollout_id}.pt"),
    )
    if asynchronous:
        config.miles.update(
            fully_async=True,
            async_data_buffer_capacity_factor=1.0,
            async_unused_samples_handler="retry",
            rollout_submission_granularity="group",
        )
    return config


def batch_membership(samples, prepared, *, rollout, asynchronous, consumed, updates=UPDATES):
    """Completion order may vary; identities and complete groups may not."""
    counts, groups, versions = Counter(), {}, {}
    for sample in samples:
        key = evidence.identity(sample)
        group = sample.get("group_index")
        if key not in prepared or type(group) is not int or group < 0:
            raise ValueError("Unprepared prompt or invalid group identity")
        if key in consumed or (group in groups and groups[group] != key):
            raise ValueError("Repeated prompt or merged prompt group")
        groups[group] = key
        counts[key] += 1
        raw_versions = sample.get("weight_versions")
        if not raw_versions or any(str(v) not in {str(i) for i in range(updates + 1)} for v in raw_versions):
            raise ValueError("Missing or malformed policy version")
        sample_versions = {int(v) for v in raw_versions}
        if len(sample_versions) != 1:
            raise ValueError("A response mixes policy versions")
        version = next(iter(sample_versions))
        if key in versions and versions[key] != version:
            raise ValueError("A prompt group mixes policy versions")
        if not 0 <= rollout - version <= int(asynchronous):
            raise ValueError("Training collection exceeds its policy-lag budget")
        versions[key] = version
    if len(counts) != 4 or set(counts.values()) != {4} or len(groups) != 4 or len(set(groups.values())) != 4:
        raise ValueError("Expected four distinct four-response prompt groups")
    return list(counts), versions, sorted(groups)


def rank_versions(samples, versions, *, world=2):
    """Pinned MILES unbalanced DP split preserves order and takes rank::world."""
    return {
        str(rank): sorted({versions[evidence.identity(sample)] for sample in samples[rank::world]})
        for rank in range(world)
    }


def audit(campaign, output, *, asynchronous):
    preparation = prepare_gsm8k_parity.verify_preparation(campaign)
    config = configuration(campaign, output, asynchronous=asynchronous)
    if json.loads((output / "arguments.json").read_text()) != config.arguments():
        raise ValueError("Executed arguments differ from the scheduling protocol")
    prepared = {evidence.identity(row): row for row in evidence.read_rows(campaign / "train.jsonl")}
    proofs = {row["prepared_sample_id"]: row for row in preparation["partitions"]["train"]["rows"]}
    # This fixture uses the pinned standard rank-strided split. Fail rather than
    # silently infer membership for a different MILES balancing/custom schedule.
    if config.miles.get("balance_data", False) or config.miles.get("custom_convert_samples_to_train_data_path"):
        raise ValueError("Scheduling audit requires standard unbalanced DP partitioning")
    consumed, consumed_groups, reports = set(), set(), []
    if {p.name for p in (output / "rollouts").glob("*.pt")} != {f"{i}.pt" for i in range(UPDATES)}:
        raise ValueError("Missing or additional retained training collections")
    for rollout in range(UPDATES):
        path = output / f"rollouts/{rollout}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        selected, version, groups = batch_membership(
            payload["samples"], prepared, rollout=rollout, asynchronous=asynchronous, consumed=consumed
        )
        if consumed_groups.intersection(groups):
            raise ValueError("Prompt group identity was consumed twice")
        report = evidence.audit_dump(
            path, [prepared[key] for key in selected], version=version, multiplicity=4, token_proofs=proofs
        )
        if not report["valid"]:
            raise ValueError(f"Rollout {rollout} failed independent audit: {report['errors']}")
        report.update(
            optimizer_step_before=rollout,
            policy_lags={key: rollout - value for key, value in version.items()},
            groups=groups,
            rank_behavior_versions=rank_versions(payload["samples"], version),
        )
        consumed.update(selected)
        consumed_groups.update(groups)
        reports.append(report)
    contracts = {}
    for rank in range(2):
        rows = [
            json.loads(line)
            for line in (output / f"metrics/training_contract_rank{rank}.jsonl").read_text().splitlines()
        ]
        steps = [row for row in rows if row.get("event") == "optimizer"]
        if [row["step"] for row in steps] != list(range(1, UPDATES + 1)) or any(
            row["optimizer_skipped"] for row in steps
        ):
            raise ValueError("Missing, repeated, or skipped optimizer step")
        for row, rollout in zip(steps, reports, strict=True):
            if (
                row["local_behavior_versions"] != rollout["rank_behavior_versions"][str(rank)]
                or row["normalization"]["samples"] != 16
            ):
                raise ValueError("Trainer consumed another policy version or batch size")
        contracts[str(rank)] = steps
    publications = [json.loads(line) for line in (output / "metrics/publication.jsonl").read_text().splitlines()]
    expected = [(0, False)] + [(step, repeated) for step in range(1, UPDATES + 1) for repeated in (False, True)]
    if [(row["version"], row["repeated_version"]) for row in publications] != expected:
        raise ValueError("Incomplete publication and current-weight diagnostic sequence")
    return dict(
        passed=True,
        asynchronous=asynchronous,
        completed_updates=UPDATES,
        preparation_sha256=evidence.digest(campaign / "preparation.json"),
        unique_consumed_prompts=len(consumed),
        training=reports,
        contracts=contracts,
        publications=publications,
        interpretation="Bounded scheduling and publication check; no evaluation, restart, endurance, or learning claim. "
        "Async completion order may select different prepared prompts than synchronous execution.",
    )


def run(campaign, output, *, asynchronous):
    prepare_gsm8k_parity.verify_preparation(campaign)
    config = configuration(campaign, output, asynchronous=asynchronous)
    sys.argv = ["scheduling-trial", *config.arguments()]
    args = arguments.parse_args()
    if (
        args.eval_interval is not None
        or args.save_interval is not None
        or args.rollout_shuffle
        or args.apply_chat_template
    ):
        raise ValueError("Scheduling trial requires prepared completions and no eval/checkpoint saves")
    output.mkdir(parents=True, exist_ok=False)
    (output / "arguments.json").write_text(json.dumps(config.arguments(), indent=2) + "\n")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    started = time.monotonic()
    ray.init(num_gpus=3, num_cpus=16, include_dashboard=False, object_store_memory=1024**3)
    try:
        asyncio.run(train(args))
    finally:
        ray.shutdown()
    report = audit(campaign, output, asynchronous=asynchronous)
    report["elapsed_seconds"] = time.monotonic() - started
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        "SCHEDULING_TRIAL_COMPLETED",
        json.dumps({k: v for k, v in report.items() if k not in ("training", "contracts", "publications")}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--audit-only", type=Path, metavar="FRESH_REPORT")
    args = parser.parse_args()
    if args.audit_only is not None:
        if args.audit_only.exists() or args.audit_only.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Audit-only evidence must use a fresh report outside the retained run")
        report = audit(args.campaign, args.output, asynchronous=not args.synchronous)
        report["retained_output"] = str(args.output)
        args.audit_only.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_only.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("SCHEDULING_REAUDIT_PASSED", args.audit_only)
    else:
        run(args.campaign, args.output, asynchronous=not args.synchronous)


if __name__ == "__main__":
    main()
