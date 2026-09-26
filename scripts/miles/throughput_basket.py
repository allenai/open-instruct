"""Reproducible topology basket and warm-window analysis; safe to import on CPU."""

import argparse
import json
import math
import statistics
from pathlib import Path

from open_instruct.miles.configuration.run_spec import RunSpec

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
    "steady-2t2i-c32-b128-graphs": {
        "profile": "small",
        "inference": 2,
        "capacity": 4,
        "concurrency": 32,
        "batch": 128,
        "token_pool": 262144,
        "state_pool": 256,
        "decode_graphs": True,
        "updates": 16,
    },
    "steady-2t4i-c8-b128-graphs": {
        "profile": "small",
        "concurrency": 8,
        "batch": 128,
        "decode_graphs": True,
        "updates": 16,
    },
    "steady-2t4i-c8-b32-graphs": {"profile": "small", "concurrency": 8, "decode_graphs": True, "updates": 16},
    "steady-2t4i-c8-b32-graphs-p32": {
        "profile": "small",
        "concurrency": 8,
        "decode_graphs": True,
        "producer_samples": 32,
        "updates": 16,
    },
    "steady-8t8i-c16-b256-graphs": {
        "profile": "large",
        "inference": 8,
        "concurrency": 16,
        "decode_graphs": True,
        "updates": 16,
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

# Keep historical cases frozen; probe packing and serving capacity independently.
for concurrency in (32, 64, 128):
    CASES[f"packed-2t2i-c{concurrency}-p512-b128"] = {
        "profile": "small",
        "inference": 2,
        "capacity": 4,
        "concurrency": concurrency,
        "batch": 128,
        "producer_samples": 512,
        "token_pool": 786432,
        "state_pool": 1024,
        "decode_graphs": True,
        "packing_tokens": 6144,
        "updates": 24,
    }


CASES["packed-2t2i-c256-p1024-b128"] = {
    **CASES["packed-2t2i-c128-p512-b128"],
    "concurrency": 256,
    "producer_samples": 1024,
    "token_pool": 1572864,
    "state_pool": 2048,
}


CASES["packed-fast-2t2i-c32-p512-b128"] = {
    **CASES["packed-2t2i-c32-p512-b128"],
    "activation_recompute": False,
    "scoring_pass_required": False,
    "replay_diagnostics": False,
}


# Matched follow-up after qualifying the faster trainer; only engine count and
# run duration change. Keep producer admission at 512 (128 engine slots total).
CASES["packed-fast-2t4i-c32-p512-b128"] = {
    **CASES["packed-fast-2t2i-c32-p512-b128"],
    "inference": 4,
    "capacity": 6,
    "updates": 48,
    "observe_compiler_cache": True,
}


def specification(case, output):
    settings = CASES[case]
    output = Path(output)
    mechanics = settings["profile"] in ("dev", "tiny")
    profile = settings["profile"]
    if mechanics:
        filename = "dev" if profile == "dev" else "small"
        run = RunSpec.load(ROOT / f"configs/miles/examples/{filename}.toml").to_dict()
    else:
        run = RunSpec.load(ROOT / "configs/miles/examples/medium.toml").to_dict()
        # Historical 4K benchmark geometry is independent of the maintained starters.
        run["judges"] = {}
        run["rubrics"] = {}
        run["judging"] = {}
        run["launch"]["gpus_per_replica"] = 8 if profile == "large" else 6
        if profile == "small":
            run["trainer"] = {
                "gpus": 2,
                "trainer_num_nodes": 1,
                "expert_parallel_size": 2,
                "micro_batch_size": 1,
                "sequence_packing": False,
                "activation_recompute": True,
                "trainer_flash_attention_version": 4,
            }
            run["inference"] = {
                "placement_mode": "disaggregated",
                "gpus": 4,
                "rollout_tensor_parallel_size": 1,
                "rollout_batch_size": 8,
                "samples_per_prompt": 4,
                "global_batch_size": 32,
                "max_response_length": 4096,
                "max_context_length": 6144,
                "sglang_server_concurrency": 16,
                "sglang_max_running_requests": 16,
                "sglang_max_total_tokens": 131072,
                "sglang_max_mamba_cache_size": 128,
                "sglang_mem_fraction_static": 0.6,
                "radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
                "sglang_cuda_graph_backend_decode": "disabled",
                "sglang_cuda_graph_max_bs_decode": 16,
                "sglang_cuda_graph_backend_prefill": "disabled",
                "sglang_sampling_backend": "pytorch",
                "sglang_attention_backend": "triton",
                "check_weight_update_equal": True,
                "update_weight_buffer_size": 1073741824,
            }
            run["core"] = {
                "max_train_rollout_logprob_abs_diff": 0.05,
                "publication_mode": "refresh",
                "refresh_request_timeout": 1800.0,
                "engine_drain_timeout": 900.0,
                "scoring_pass_required": True,
                "router_aux_loss_weight": 0.01,
                "router_z_loss_weight": 1e-05,
                "stream_moe_export": True,
                "weight_sync_mode": "flattened",
                "expert_publication": "per_expert",
                "scoring_check_interval": 50,
            }
            run["async"] = {
                "fully_async": True,
                "max_weight_staleness": 2,
                "async_data_buffer_capacity_factor": 1.0,
                "async_unused_samples_handler": "retry",
                "rollout_submission_granularity": "group",
                "off_policy_correction": "tis",
            }
            run["miles"] = {
                "use_rollout_routing_replay": True,
                "use_miles_router": True,
                "sglang_chunked_prefill_size": 8192,
                "rollout_temperature": 1.0,
                "rollout_seed": 17,
                "rollout_max_prompt_len": 2048,
                "seed": 17,
                "disable_grpo_std_normalization": True,
            }
        if profile == "large":
            run["trainer"] = {
                "gpus": 8,
                "trainer_num_nodes": 1,
                "expert_parallel_size": 8,
                "micro_batch_size": 1,
                "sequence_packing": False,
                "activation_recompute": True,
                "trainer_flash_attention_version": 4,
            }
            run["inference"] = {
                "placement_mode": "disaggregated",
                "gpus": 56,
                "rollout_tensor_parallel_size": 1,
                "rollout_batch_size": 64,
                "samples_per_prompt": 4,
                "global_batch_size": 256,
                "max_response_length": 4096,
                "max_context_length": 6144,
                "sglang_server_concurrency": 16,
                "sglang_max_running_requests": 16,
                "sglang_max_total_tokens": 131072,
                "sglang_max_mamba_cache_size": 128,
                "sglang_mem_fraction_static": 0.6,
                "radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
                "sglang_cuda_graph_backend_decode": "disabled",
                "sglang_cuda_graph_max_bs_decode": 16,
                "sglang_cuda_graph_backend_prefill": "disabled",
                "sglang_sampling_backend": "pytorch",
                "sglang_attention_backend": "triton",
                "check_weight_update_equal": True,
                "update_weight_buffer_size": 1073741824,
            }
            run["core"] = {
                "max_train_rollout_logprob_abs_diff": 0.05,
                "publication_mode": "refresh",
                "refresh_request_timeout": 1800.0,
                "engine_drain_timeout": 900.0,
                "scoring_pass_required": True,
                "router_aux_loss_weight": 0.01,
                "router_z_loss_weight": 1e-05,
                "stream_moe_export": True,
                "weight_sync_mode": "flattened",
                "expert_publication": "per_expert",
                "scoring_check_interval": 50,
            }
            run["async"] = {
                "fully_async": True,
                "max_weight_staleness": 2,
                "async_data_buffer_capacity_factor": 1.0,
                "async_unused_samples_handler": "retry",
                "rollout_submission_granularity": "group",
                "off_policy_correction": "tis",
            }
            run["miles"] = {
                "use_rollout_routing_replay": True,
                "use_miles_router": True,
                "sglang_chunked_prefill_size": 8192,
                "rollout_temperature": 1.0,
                "rollout_seed": 17,
                "rollout_max_prompt_len": 2048,
                "seed": 17,
                "disable_grpo_std_normalization": True,
            }
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
    if "producer_samples" in settings:
        run["async"]["async_max_concurrent_samples"] = settings["producer_samples"]
    if "concurrency" in settings:
        run["inference"]["sglang_server_concurrency"] = settings["concurrency"]
        run["inference"]["sglang_max_running_requests"] = settings["concurrency"]
    if "packing_tokens" in settings:
        run["trainer"].update(sequence_packing=True, packing_max_tokens=settings["packing_tokens"])
    if "token_pool" in settings:
        run["inference"]["sglang_max_total_tokens"] = settings["token_pool"]
    if "state_pool" in settings:
        run["inference"]["sglang_max_mamba_cache_size"] = settings["state_pool"]
    if settings.get("decode_graphs"):
        run["inference"]["sglang_cuda_graph_backend_decode"] = "full"
        run["inference"]["sglang_cuda_graph_max_bs_decode"] = settings["concurrency"]
        run["core"]["replay_diagnostics"] = True
    if "activation_recompute" in settings:
        run["trainer"]["activation_recompute"] = settings["activation_recompute"]
    for option in ("scoring_pass_required", "replay_diagnostics"):
        if option in settings:
            run["core"][option] = settings[option]
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
    queue_waits = [r["queue_metrics"].get(prefix + "consumer_wait_seconds") for r in chosen]
    wait_breakdown = None
    if all(value is not None for value in queue_waits):
        if any(
            not math.isfinite(value) or value < 0 or value > outer + 0.001
            for value, outer in zip(queue_waits, durations["generation_wait"], strict=True)
        ):
            raise ValueError("Completed-queue wait is invalid or exceeds its enclosing collection stage")
        queue_seconds = sum(queue_waits)
        wait_breakdown = {
            "scope": "Completed-buffer get time includes expiry filtering; the remainder includes collection and handoff, not solely transfer.",
            "completed_queue_get_seconds": queue_seconds,
            "completed_queue_get_cycle_fraction": queue_seconds / seconds,
            "other_collection_seconds": max(0, sum(durations["generation_wait"]) - queue_seconds),
            "other_collection_cycle_fraction": max(0, sum(durations["generation_wait"]) - queue_seconds) / seconds,
        }
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
        "batch_collection_breakdown": wait_breakdown,
        "discarded_response_tokens": dropped,
        "discarded_token_fraction": dropped / max(1, dropped + delivered),
        "mixed_responses": sum(r["mixed_responses"] for r in chosen),
        "per_update": [
            {
                "rollout_id": row["rollout_id"],
                "response_tokens": row["response_tokens"],
                "mixed_responses": row["mixed_responses"],
                "completed_queue_get_seconds": queue_waits[index],
                "other_collection_seconds": (
                    max(0, durations["generation_wait"][index] - queue_waits[index])
                    if wait_breakdown is not None
                    else None
                ),
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
    inventory = root / "checkpoints/pipeline_lifecycle.jsonl"
    result["terminal_inventory"] = rows(inventory) if inventory.exists() else None
    result["terminal_unused_work"] = terminal_unused_work(
        result["terminal_inventory"],
        delivered_tokens=sum(row["response_tokens"] for row in flow),
        stale_dropped_tokens=sum(row["queue_metrics"].get(prefix + "dropped_response_tokens", 0) for row in flow),
    )
    result["end_to_end_passed"] = result["workflow"]["status"] == "complete"
    if not result["end_to_end_passed"] and not allow_incomplete_workflow:
        raise ValueError("Workflow did not complete")
    return result


def terminal_unused_work(lifecycle, *, delivered_tokens, stale_dropped_tokens):
    """Do not mistake unobserved final buffers for empty buffers."""
    if not lifecycle or lifecycle[-1]["event"] != "shutdown_complete":
        return None
    terminal = lifecycle[-1]
    sections = [terminal.get(key) for key in ("completed_queue", "producer_ready", "shutdown_unqueued")]
    if any(section is None for section in sections):
        return None
    remaining = {key: sum(section[key] for section in sections) for key in ("groups", "samples", "response_tokens")}
    accounted = delivered_tokens + stale_dropped_tokens + remaining["response_tokens"]
    return {
        **remaining,
        "delivered_response_tokens_all_updates": delivered_tokens,
        "stale_dropped_response_tokens_all_updates": stale_dropped_tokens,
        "fraction_of_accounted_response_tokens": remaining["response_tokens"] / accounted if accounted else None,
        "scope": "Final buffered, producer-ready and shutdown-unqueued completions; distinct from stale drops. Denominator excludes unobserved aborted or dynamically filtered work.",
    }


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
