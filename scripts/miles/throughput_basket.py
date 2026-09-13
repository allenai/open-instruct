"""Reproducible topology basket and warm-window analysis; safe to import on CPU."""

import argparse
import json
import math
import statistics
from pathlib import Path

from open_instruct.miles.run_spec import RunSpec

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path("/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1")
CASES = {
    "dev": {"profile": "dev", "updates": 4},
    "tiny": {"profile": "tiny", "updates": 4},
    "small-2t4i-group": {"profile": "small"},
    "small-4t2i-group": {"profile": "small", "trainer": 4, "inference": 2},
    "small-2t4i-sample": {"profile": "small", "submission": "sample"},
    "small-2t4i-c8": {"profile": "small", "concurrency": 8},
    "bridge-2t6i": {"profile": "small", "inference": 6, "capacity": 8},
    "bridge-8t8i": {"profile": "large", "inference": 8, "updates": 4},
    "large-8t56i": {"profile": "large", "updates": 16},
    "steady-8t24i-c8-b256": {"profile": "large", "inference": 24, "concurrency": 8, "updates": 16},
    "steady-2t6i-c8-b32": {"profile": "small", "inference": 6, "capacity": 8, "concurrency": 8, "updates": 24},
    "steady-2t6i-c8-b128": {
        "profile": "small",
        "inference": 6,
        "capacity": 8,
        "concurrency": 8,
        "batch": 128,
        "updates": 24,
    },
    "steady-2t6i-c8-b128-graphs": {
        "profile": "small",
        "inference": 6,
        "capacity": 8,
        "concurrency": 8,
        "batch": 128,
        "decode_graphs": True,
        "updates": 16,
    },
    "steady-2t16i-c8-b128": {
        "profile": "small",
        "inference": 16,
        "capacity": 8,
        "concurrency": 8,
        "batch": 128,
        "updates": 16,
    },
}


def specification(case, output):
    settings = CASES[case]
    output = Path(output)
    run = RunSpec.load(ROOT / f"configs/miles/examples/{settings['profile']}.toml").to_dict()
    mechanics = settings["profile"] in ("dev", "tiny")
    source = output.parent / "fixture" if mechanics else CAMPAIGN
    run["name"] = "throughput-" + case
    run["model"]["source"] = str(source / "hf")
    run["output"] = {"root": str(output), "export_hf": False}
    run["conversion"] = {"hf_output": str(output / "prepared/hf")}
    run["data"] = {
        "seed": 17,
        "shuffle": True,
        "prompt_data": str(source / ("prompts.jsonl" if mechanics else "train.jsonl")),
        "reward_config": str(source / "verifiers.json"),
        "eval_prompt_data": [],
    }
    run["training"] = {
        "num_rollouts": settings.get("updates", 12),
        "save_checkpoints": mechanics,
        "skip_eval_before_train": True,
    }
    if mechanics:
        run["training"]["save_interval"] = 2
    run["launch"]["auto_resume"] = False
    run["launch"]["timeout"] = "2h" if mechanics else "3h"
    run["launch"]["min_runtime"] = "15m" if mechanics else "1h"
    run["launch"]["secrets"]["WANDB_API_KEY"] = "robertb_WANDB_API_KEY"
    run["tracking"].update(
        wandb_mode="online",
        wandb_team="ai2-llm",
        wandb_project="olmo-rl-comparison",
        wandb_group="throughput-profiles-20260913-v1",
    )
    run["miles"]["lr_decay_iters"] = settings.get("updates", 12)
    if "trainer" in settings:
        run["trainer"]["gpus"] = settings["trainer"]
        run["trainer"]["expert_parallel_size"] = settings["trainer"]
    if "inference" in settings:
        run["inference"]["gpus"] = settings["inference"]
    if "capacity" in settings:
        run["launch"]["gpus_per_replica"] = settings["capacity"]
    if "batch" in settings:
        run["inference"]["global_batch_size"] = settings["batch"]
        run["inference"]["rollout_batch_size"] = settings["batch"] // run["inference"]["samples_per_prompt"]
    run["core"]["pipeline_observation_interval"] = 2.0
    run["miles"]["sglang_enable_metrics"] = True
    if "submission" in settings:
        run["async"]["rollout_submission_granularity"] = settings["submission"]
    if "concurrency" in settings:
        run["inference"]["sglang_server_concurrency"] = settings["concurrency"]
        run["inference"]["sglang_max_running_requests"] = settings["concurrency"]
    if settings.get("decode_graphs"):
        run["inference"]["sglang_cuda_graph_backend_decode"] = "full"
        run["inference"]["sglang_cuda_graph_max_bs_decode"] = settings["concurrency"]
        run["core"]["replay_diagnostics"] = True
    # The benchmark observes drift without a run-ending threshold established on a different batch distribution.
    run["core"].pop("max_train_rollout_logprob_abs_diff", None)
    return RunSpec.from_dict(run, config_path=output / "trial.json")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(root, *, warmup=3, allow_incomplete_workflow=False):
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    root = Path(root)
    metrics = root / "checkpoints"
    stages = rows(metrics / "driver_timing.jsonl")
    flow = rows(metrics / "rollout_flow.jsonl")
    if any(not r["passed"] for r in stages):
        raise ValueError("Failed driver stage; do not report a successful throughput run")
    options = json.loads((root / "plan.json").read_text())["runtime"]["miles"]
    expected = options["num_rollout"]
    if [r["rollout_id"] for r in flow] != list(range(expected)):
        raise ValueError("Missing or repeated consumed collections")
    ranks = options.get("actor_num_nodes", 1) * options.get("actor_num_gpus_per_node", 1)
    contracts = []
    for rank in range(ranks):
        path = metrics / f"training_contract_rank{rank}.jsonl"
        if not path.exists():
            raise ValueError(f"Missing trainer contract for rank {rank}")
        records = rows(path)
        updates = [r for r in records if r["event"] == "optimizer"]
        if [r["step"] for r in updates] != list(range(1, expected + 1)) or any(
            r["optimizer_skipped"] for r in updates
        ):
            raise ValueError(f"Missing or skipped optimizer updates on rank {rank}")
        contracts.append(records)
    contract = contracts[0]
    chosen = [r for r in flow if r["rollout_id"] >= warmup]
    durations = {
        name: [
            r["seconds"]
            for r in stages
            if r["stage"] == name and r["rollout_id"] is not None and r["rollout_id"] >= warmup
        ]
        for name in ("generation_wait", "training", "publication")
    }
    if not chosen or any(len(values) != len(chosen) for values in durations.values()):
        raise ValueError("Incomplete warm window")
    seconds = sum(sum(v) for v in durations.values())
    prefix = "rollout/fully_async/completed_queue/"
    if options.get("fully_async", False):
        for row in flow:
            if not all(
                prefix + key in row["queue_metrics"]
                for key in ("dropped_response_tokens", "delivered_response_tokens")
            ):
                raise ValueError("Missing async queue counters; absence is not a zero discard rate")
            if row["queue_metrics"][prefix + "delivered_response_tokens"] != row["response_tokens"]:
                raise ValueError(
                    f"Queue delivery accounting differs from consumed tokens at rollout {row['rollout_id']}"
                )
    if seconds <= 0 or any(not math.isfinite(v) or v < 0 for values in durations.values() for v in values):
        raise ValueError("Invalid measured duration")
    dropped = sum(r["queue_metrics"].get(prefix + "dropped_response_tokens", 0) for r in chosen)
    delivered = sum(r["response_tokens"] for r in chosen)
    components = {
        "standalone_scoring": [
            r["seconds"] for r in contract if r["event"] == "score_timing" and r["rollout_id"] >= warmup
        ],
        "forward_backward_optimizer": [
            r["elapsed_seconds"]
            for r in contract
            if r["event"] == "optimizer" and r.get("rollout_id", r["step"] - 1) >= warmup and "elapsed_seconds" in r
        ],
    }
    provenance = [
        r for records in contracts for r in records if r["event"] == "refresh_scores" and r["rollout_id"] >= warmup
    ]
    provenance_tokens = sum(r["active_tokens"] for r in provenance)
    result = {
        "current_policy_token_fraction": (
            sum(r["active_tokens"] * r["current_version_token_fraction"] for r in provenance) / provenance_tokens
            if provenance_tokens
            else None
        ),
        "provenance_active_tokens": provenance_tokens,
        "scope": "Warm awaited driver-cycle throughput, excluding startup, checkpoints, evaluation and final drain; generation overlaps training.",
        "completed_updates": expected,
        "validated_trainer_ranks": ranks,
        "training_components_rank0": {
            name: {
                "count": len(values),
                "total_seconds": sum(values),
                "median_seconds": statistics.median(values) if values else None,
            }
            for name, values in components.items()
        },
        "warmup_updates": warmup,
        "measured_updates": len(chosen),
        "warm_cycle_seconds": seconds,
        "useful_response_tokens": delivered,
        "useful_response_tokens_per_second": delivered / seconds,
        "trainer_wait_fraction": sum(durations["generation_wait"]) / seconds,
        "discarded_response_tokens": dropped,
        "discarded_token_fraction": dropped / max(1, dropped + delivered),
        "mixed_responses": sum(r["mixed_responses"] for r in chosen),
        "per_update": [
            {
                "rollout_id": row["rollout_id"],
                "response_tokens": row["response_tokens"],
                "mixed_responses": row["mixed_responses"],
                **{name + "_seconds": values[index] for name, values in durations.items()},
            }
            for index, row in enumerate(chosen)
        ],
        "median_seconds": {name: statistics.median(values) for name, values in durations.items()},
        "all_driver_stage_seconds": {
            name: sum(r["seconds"] for r in stages if r["stage"] == name)
            for name in sorted({r["stage"] for r in stages})
        },
        "workflow": json.loads((root / "workflow.json").read_text()),
    }
    result["end_to_end_passed"] = result["workflow"]["status"] == "complete"
    if not result["end_to_end_passed"] and not allow_incomplete_workflow:
        raise ValueError("Workflow did not complete")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    report = analyze(args.root, warmup=args.warmup)
    (args.root / "throughput-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
