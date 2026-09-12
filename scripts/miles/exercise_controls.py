"""Bounded matched throughput and grouped CLI-control qualification."""

import argparse
import dataclasses
import json
import runpy
import statistics
import sys
import time
from pathlib import Path

import ray
import torch
from scripts.miles import analyze_gsm8k_parity as evidence
from scripts.miles import async_trial, gsm8k_parity, prepare_gsm8k_parity
from scripts.miles.check_gsm8k_checkpoints import check_boundaries

from open_instruct.miles import replay_diagnostics
from open_instruct.miles.config import RunConfig

ARMS = ("sync", "async", "controls", "sync-admission64", "async-admission64", "replay-admission64")
ADMISSION_KEYS = (
    "rollout_batch_size",
    "global_batch_size",
    "sglang_log_level",
    "sglang_server_concurrency",
    "sglang_max_running_requests",
    "sglang_max_total_tokens",
    "sglang_max_mamba_cache_size",
    "sglang_cuda_graph_max_bs_decode",
)


def is_async(arm):
    return arm in ("async", "controls", "async-admission64")


def configuration(campaign, output, arm, updates):
    config = gsm8k_parity.configuration(campaign, updates=updates)
    for key in list(config.miles):
        if key.startswith(("eval_", "wandb_")) or key in ("n_samples_per_eval_prompt", "use_wandb"):
            del config.miles[key]
    config = dataclasses.replace(
        config,
        core=dataclasses.replace(
            config.core,
            row_specialization="dynamic",
            max_policy_lag=int(is_async(arm)),
            diagnostic_interval=0,
            scoring_pass_required=True,
        ),
    )
    config.miles.update(
        num_rollout=updates,
        lr_decay_iters=updates,
        use_rollout_logprobs=True,
        save=str(output / "metrics"),
        save_debug_rollout_data=str(output / "rollouts/{rollout_id}.pt"),
        use_wandb=True,
        wandb_mode="offline",
        wandb_project="olmo-rl-comparison",
        wandb_group="core-controls-20260911",
        wandb_dir=str(output / "wandb"),
    )
    if is_async(arm):
        config.miles.update(
            fully_async=True,
            async_data_buffer_capacity_factor=1.0,
            async_unused_samples_handler="retry",
            rollout_submission_granularity="group",
        )
    if arm == "controls":
        config = dataclasses.replace(config, core=dataclasses.replace(config.core, diagnostic_interval=updates))
        config.miles.update(
            use_rollout_logprobs=False,
            use_tis=True,
            tis_clip=2.0,
            tis_clip_low=0.5,
            async_data_buffer_capacity_factor=2.0,
            async_unused_samples_handler="drop",
            lr_decay_style="cosine",
            lr_warmup_iters=1,
            entropy_coef=0.001,
            sglang_max_running_requests=8,
            sglang_server_concurrency=8,
            sglang_cuda_graph_max_bs_decode=8,
            sglang_max_mamba_cache_size=16,
            update_weight_buffer_size=256 * 1024**2,
            save_interval=updates,
            eval_prompt_data=["gsm8k", str(output / "eval.jsonl")],
            eval_interval=updates,
            eval_temperature=0.0,
            n_samples_per_eval_prompt=1,
            eval_max_response_len=4096,
        )
    if arm.endswith("-admission64"):
        starter = RunConfig.load(
            Path(__file__).resolve().parents[2] / "configs/miles/profiles/train-disaggregated.toml"
        )
        config.miles.update({key: starter.miles[key] for key in ADMISSION_KEYS})
        # Preserve this historical 16 x 4 measurement recipe as starter defaults evolve.
        # It also measures standalone scoring on every collection by design.
        config.miles.update(rollout_batch_size=16, n_samples_per_prompt=4, global_batch_size=64)
    if arm == "replay-admission64":
        config = dataclasses.replace(
            config, core=dataclasses.replace(config.core, replay_diagnostics=True, diagnostic_interval=1)
        )
        config.miles.update(use_rollout_routing_replay=True, use_miles_router=True)
    config.validate()
    return config


def write_config(path, config):
    # This campaign deliberately uses scalar/list controls; JSON encodings of
    # those values are also valid TOML. None means omit the optional setting.
    sections = {"core": dataclasses.asdict(config.core), "miles": config.miles}
    path.write_text(
        "\n".join(
            f"[{section}]\n"
            + "\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items() if value is not None)
            for section, values in sections.items()
        )
        + "\n"
    )
    if RunConfig.load(path).arguments() != config.arguments():
        raise ValueError("TOML round trip changed runtime arguments")


def prepare(campaign, output, arm, updates):
    prepare_gsm8k_parity.verify_preparation(campaign)
    output.mkdir(parents=True, exist_ok=False)
    if arm == "controls":
        (output / "eval.jsonl").write_text("\n".join((campaign / "eval.jsonl").read_text().splitlines()[:8]) + "\n")
    config = configuration(campaign, output, arm, updates)
    if updates * config.miles["rollout_batch_size"] > len(evidence.read_rows(campaign / "train.jsonl")):
        raise ValueError("Trial would repeat prompts from the frozen training set")
    write_config(output / "run.toml", config)
    (output / "arguments.json").write_text(json.dumps(config.arguments(), indent=2) + "\n")
    (output / "protocol.json").write_text(
        json.dumps(
            dict(
                arm=arm,
                updates=updates,
                campaign=str(campaign),
                preparation_sha256=evidence.digest(campaign / "preparation.json"),
            ),
            indent=2,
        )
        + "\n"
    )


def train_cli(output):
    # Exercise the real facade after initializing a bounded local Ray cluster.
    started = time.perf_counter()
    ray.init(num_gpus=3, num_cpus=16, include_dashboard=False, object_store_memory=1024**3)
    try:
        sys.argv = [
            "open-instruct-miles",
            "train",
            str(output / "run.toml"),
            "--set",
            'core.row_specialization="dynamic"',
        ]
        runpy.run_module("open_instruct.miles", run_name="__main__")
    finally:
        ray.shutdown()
        (output / "elapsed.json").write_text(json.dumps({"seconds": time.perf_counter() - started}) + "\n")


def describe(values):
    return (
        {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        if values
        else None
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_publications(config, publications, updates):
    expected = [(0, False)]
    interval = config.core.diagnostic_interval
    for version in range(1, updates + 1):
        expected.append((version, False))
        if config.miles.get("check_weight_update_equal", False) and interval > 0 and version % interval == 0:
            expected.append((version, True))
    if [(row["version"], row["repeated_version"]) for row in publications] != expected:
        raise ValueError("Missing or unexpected publication/verification round trip")


def audit(campaign, output, arm, updates, report_path=None):
    config = configuration(campaign, output, arm, updates)
    if RunConfig.load(output / "run.toml").arguments() != config.arguments():
        raise ValueError("Run config differs from protocol")
    preparation = prepare_gsm8k_parity.verify_preparation(campaign)
    prepared = {evidence.identity(row): row for row in evidence.read_rows(campaign / "train.jsonl")}
    proofs = {row["prepared_sample_id"]: row for row in preparation["partitions"]["train"]["rows"]}
    collection_size = config.miles["rollout_batch_size"] * config.miles["n_samples_per_prompt"]
    consumed, groups_seen, reports = set(), set(), []
    for rollout in range(updates):
        path = output / f"rollouts/{rollout}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        selected, versions, groups = async_trial.batch_membership(
            payload["samples"],
            prepared,
            rollout=rollout,
            asynchronous=is_async(arm),
            consumed=consumed,
            updates=updates,
            groups_per_collection=config.miles["rollout_batch_size"],
            samples_per_prompt=config.miles["n_samples_per_prompt"],
        )
        if groups_seen.intersection(groups):
            raise ValueError("Duplicate consumed group")
        report = evidence.audit_dump(
            path,
            [prepared[key] for key in selected],
            version=versions,
            multiplicity=config.miles["n_samples_per_prompt"],
            token_proofs=proofs,
        )
        if not report["valid"]:
            raise ValueError(f"Independent reward/token audit failed: {report['errors']}")
        report.update(
            rollout_id=rollout,
            policy_lags=[rollout - v for v in versions.values()],
            rank_versions=async_trial.rank_versions(payload["samples"], versions),
        )
        reports.append(report)
        consumed.update(selected)
        groups_seen.update(groups)
    contracts = {}
    for rank in (0, 1):
        rows = read_jsonl(output / f"metrics/training_contract_rank{rank}.jsonl")
        steps = [row for row in rows if row["event"] == "optimizer"]
        if [row["step"] for row in steps] != list(range(1, updates + 1)) or any(
            row["optimizer_skipped"] for row in steps
        ):
            raise ValueError("Missing or skipped optimizer step")
        for step, report in zip(steps, reports, strict=True):
            if (
                step["local_behavior_versions"] != report["rank_versions"][str(rank)]
                or step["normalization"]["samples"] != config.miles["global_batch_size"]
            ):
                raise ValueError("Trainer consumed different versions or sample count")
        contracts[str(rank)] = rows
    publications = read_jsonl(output / "metrics/publication.jsonl")
    validate_publications(config, publications, updates)
    stages = read_jsonl(output / "metrics/driver_timing.jsonl")
    if any(not row["passed"] for row in stages):
        raise ValueError("A measured driver stage failed")
    score_rows = [[row for row in contracts[str(rank)] if row["event"] == "score_timing"] for rank in (0, 1)]
    if any([row["rollout_id"] for row in rows] != list(range(updates)) for rows in score_rows):
        raise ValueError("Missing score timings")
    if any(row["row_specialization"] != "dynamic" for rows in score_rows for row in rows):
        raise ValueError("Static scoring unexpectedly configured")
    cycles = []
    for rollout in range(updates):
        rows = [
            row
            for row in stages
            if row["rollout_id"] == rollout and row["stage"] in {"generation_wait", "training", "publication"}
        ]
        if {row["stage"] for row in rows} != {"generation_wait", "training", "publication"} or len(rows) != 3:
            raise ValueError("Incomplete cycle timings")
        cycles.append(sum(row["seconds"] for row in rows))
    report = dict(
        passed=True,
        arm=arm,
        updates=updates,
        training=reports,
        cycle_seconds=describe(cycles),
        post_first_four_cycle_seconds=describe(cycles[4:]),
        score_seconds=describe([max(a["seconds"], b["seconds"]) for a, b in zip(*score_rows, strict=True)]),
        stages={
            name: describe([r["seconds"] for r in stages if r["stage"] == name])
            for name in {r["stage"] for r in stages}
        },
        evaluation_timings=[row for row in stages if row["stage"] in {"evaluation", "evaluation_dispatch"}],
        measured_cycle_seconds=sum(cycles),
        consumed_samples=collection_size * updates,
        consumed_response_tokens=sum(row["summary"]["mean_response_tokens"] * collection_size for row in reports),
        publication_count=len(publications),
        scoring_by_rank=score_rows,
        elapsed=json.loads((output / "elapsed.json").read_text()),
        native_log_timing=evidence.parse_timing_log(output / "run.log", warmup_updates=4, eval_steps=()),
        interpretation="Cycle excludes startup, eval and saving; async wait is consumer stall, not total inference work. Different completion order means different consumed prompts. No learning-quality claim.",
    )
    if arm == "controls":
        report["checkpoints"] = check_boundaries(output / "metrics", updates=updates, save_interval=updates)
        eval_rows = evidence.read_rows(output / "eval.jsonl")
        eval_proofs = {row["prepared_sample_id"]: row for row in preparation["partitions"]["eval"]["rows"]}
        report["evaluation"] = [
            evidence.audit_dump(
                output / f"rollouts/{name}.pt", eval_rows, version=version, multiplicity=1, token_proofs=eval_proofs
            )
            for name, version in (("eval_0", 0), (f"eval_{updates - 1}", updates))
        ]
        if not all(row["valid"] for row in report["evaluation"]):
            raise ValueError("Heldout eval audit failed")
    if arm == "replay-admission64":
        from_replay = replay_diagnostics.audit_contracts(contracts, updates, collection_size // 2)
        report["replay"] = from_replay
    destination = report_path or output / "audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CONTROL_EXERCISE_PASSED", arm, json.dumps({k: v for k, v in report.items() if k != "training"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "audit"))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--report", type=Path, help="Fresh independent audit report path")
    args = parser.parse_args()
    if not 4 <= args.updates <= 80:
        parser.error("Use 4..80 updates within the frozen 400-prompt training set")
    if args.command == "prepare":
        prepare(args.campaign, args.output, args.arm, args.updates)
    elif args.command == "train":
        train_cli(args.output)
    else:
        audit(args.campaign, args.output, args.arm, args.updates, args.report)


if __name__ == "__main__":
    main()
