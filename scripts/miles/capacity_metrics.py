"""Reconstruct capacity metrics from completed runs; no training or W&B dependency.

Phase rates and full-cycle useful rates have deliberately different names.
Hardware observations include coverage; unavailable values are never zero-filled.
"""

import json
from pathlib import Path

from scripts.miles import throughput_basket, throughput_occupancy

from open_instruct.miles.performance import training_rates

PREFIX = "rollout/fully_async/completed_queue/"


def measurements(root, *, warmup=6):
    root = Path(root)
    report = throughput_basket.analyze(root, warmup=0)
    plan = json.loads((root / "plan.json").read_text())
    allocation = plan["allocation"]
    if plan["runtime"]["miles"].get("colocate", False):
        raise ValueError("Capacity role attribution currently requires disaggregated trainer and inference GPUs")
    roles = throughput_occupancy.node_roles(root)
    trainer_gpus = sum(node["trainer_gpus"] for node in allocation["nodes"])
    inference_gpus = sum(node["rollout_gpus"] for node in allocation["nodes"])
    expected_engines = inference_gpus // plan["runtime"]["miles"]["rollout_num_gpus_per_engine"]
    contracts = [
        throughput_basket.rows(root / "checkpoints" / f"training_contract_rank{r}.jsonl") for r in range(trainer_gpus)
    ]
    stages = throughput_basket.rows(root / "checkpoints/driver_timing.jsonl")
    flows = {r["rollout_id"]: r for r in throughput_basket.rows(root / "checkpoints/rollout_flow.jsonl")}
    output = []
    for cycle in report["per_update"]:
        rollout_id = cycle["rollout_id"]
        selected = [
            s
            for s in stages
            if s["rollout_id"] == rollout_id and s["stage"] in ("generation_wait", "training", "publication")
        ]
        start = min(s["started_unix"] for s in selected)
        end = max(s["started_unix"] + s["seconds"] for s in selected)
        cycle_seconds = sum(s["seconds"] for s in selected)
        flow = flows[rollout_id]
        values = {
            "update": rollout_id + 1,
            "is_warm": int(rollout_id >= warmup),
            "pipeline/cycle_seconds": cycle_seconds,
            "pipeline/collection_seconds": cycle["generation_wait_seconds"],
            "pipeline/publication_seconds": cycle["publication_seconds"],
            "pipeline/trainer_occupied_fraction": (cycle["training_seconds"] + cycle["publication_seconds"])
            / cycle_seconds,
            "pipeline/useful_response_tokens_per_second": cycle["response_tokens"] / cycle_seconds,
            "pipeline/useful_response_tokens_per_allocated_gpu_second": cycle["response_tokens"]
            / cycle_seconds
            / allocation["allocated_gpus"],
            "trainer/score_train_seconds": cycle["training_seconds"],
            "trainer/useful_response_tokens_per_gpu_cycle_second": cycle["response_tokens"]
            / cycle_seconds
            / trainer_gpus,
            "inference/useful_response_tokens_per_gpu_cycle_second": cycle["response_tokens"]
            / cycle_seconds
            / inference_gpus,
        }
        for key in ("completed_queue_get_seconds", "other_collection_seconds"):
            if cycle[key] is not None:
                values["pipeline/" + key] = cycle[key]
        optimizers = [
            [r for r in rows if r["event"] == "optimizer" and r["rollout_id"] == rollout_id] for rows in contracts
        ]
        seconds = sum(
            max(rank[index]["elapsed_seconds"] for rank in optimizers) for index in range(len(optimizers[0]))
        )
        model_tokens = sum(r["normalization"]["model_tokens"] for r in optimizers[0])
        active = sum(r["normalization"]["active_tokens"] for r in optimizers[0])
        values.update(
            {
                "trainer/" + key: value
                for key, value in training_rates(model_tokens, active, seconds, trainer_gpus).items()
            }
        )
        values["trainer/forward_backward_optimizer_seconds"] = seconds
        scores = [
            [r for r in rows if r["event"] == "score_timing" and r["rollout_id"] == rollout_id] for rows in contracts
        ]
        if all(scores):
            score_seconds = max(sum(r["seconds"] for r in rank) for rank in scores)
            score_tokens = sum(r["model_tokens"] for rank in scores for r in rank)
            values["trainer/scoring_seconds"] = score_seconds
            values["trainer/scoring_model_tokens_per_gpu_second"] = score_tokens / score_seconds / trainer_gpus
        for key, value in flow["queue_metrics"].items():
            if key.startswith(PREFIX):
                values["quality/" + key.removeprefix(PREFIX)] = value
        values["quality/mixed_responses"] = flow["mixed_responses"]
        occupancy = throughput_occupancy.analyze(root, warmup=0, window=(start, end))
        for key, summary in occupancy["pipeline"].items():
            values["pipeline/" + key + "_coverage"] = summary["coverage_fraction"]
            if summary["mean"] is not None:
                values["pipeline/" + key] = summary["mean"]
        engine_values = {}
        hardware_values = {"trainer": [], "inference": []}
        rate_sum = 0.0
        rate_engines = 0
        for index, (_, engine) in enumerate(sorted(occupancy["engines"].items())):
            for metric in engine["series"]:
                key = metric["name"]
                # Do not combine ambiguous rank-labelled gauges into a fleet total.
                if sum(s["name"] == key for s in engine["series"]) != 1:
                    continue
                if key in ("utilization", "fwd_occupancy"):
                    continue
                values[f"inference/engine_{index}/{key}_coverage"] = metric["coverage_fraction"]
                if metric["mean"] is not None:
                    values[f"inference/engine_{index}/{key}"] = metric["mean"]
                    values[f"inference/engine_{index}/{key}_peak"] = metric["maximum"]
                    engine_values.setdefault(key, []).append(metric)
                    if key == "gen_throughput" and metric["coverage_fraction"] >= 0.95:
                        rate_sum += metric["mean"]
                        rate_engines += 1
        for key, observations in engine_values.items():
            if len(observations) == expected_engines and min(m["coverage_fraction"] for m in observations) >= 0.95:
                values[f"inference/{key}_mean_per_engine"] = sum(m["mean"] for m in observations) / expected_engines
                values[f"inference/{key}_peak_any_engine"] = max(m["maximum"] for m in observations)
        if rate_engines == expected_engines:
            values["inference/observed_decode_tokens_per_gpu_second"] = rate_sum / inference_gpus
        for node_name, devices in occupancy["hardware"].items():
            node = node_name.removeprefix("gpu_usage_node")
            role = roles.get(node)
            if role is None:
                continue
            for device in devices:
                index = int(device["index"])
                category = (
                    "trainer"
                    if index < role["trainer_gpus"]
                    else "inference"
                    if index < role["trainer_gpus"] + role["rollout_gpus"]
                    else "unused"
                )
                if category in hardware_values:
                    hardware_values[category].append(device["metrics"])
                base = f"{category}/node{node}_gpu{index}"
                for source, label, divisor in (
                    ("utilization.gpu", "kernel_activity_percent", 1),
                    ("memory.used", "memory_used_gib", 1024),
                    ("memory.total", "memory_total_gib", 1024),
                ):
                    metric = device["metrics"][source]
                    values[f"{base}/{label}_coverage"] = metric["coverage_fraction"]
                    if metric["mean"] is not None:
                        values[f"{base}/{label}"] = metric["mean"] / divisor
                        values[f"{base}/{label}_peak"] = metric["maximum"] / divisor
        for category, devices in hardware_values.items():
            expected = trainer_gpus if category == "trainer" else inference_gpus
            if len(devices) != expected:
                continue
            for source, label, divisor in (
                ("utilization.gpu", "kernel_activity_percent", 1),
                ("memory.used", "memory_used_gib", 1024),
                ("memory.total", "memory_total_gib", 1024),
            ):
                observations = [d[source] for d in devices]
                coverage = min(m["coverage_fraction"] for m in observations)
                values[f"{category}/{label}_minimum_coverage"] = coverage
                if coverage >= 0.95:
                    values[f"{category}/{label}_mean"] = sum(m["mean"] for m in observations) / expected / divisor
                    values[f"{category}/{label}_peak_any_gpu"] = max(m["maximum"] for m in observations) / divisor
        refresh = [
            r for rank in contracts for r in rank if r["event"] == "refresh_scores" and r["rollout_id"] == rollout_id
        ]
        if refresh and sum(r["active_tokens"] for r in refresh):
            values["quality/current_policy_token_fraction"] = sum(
                r["active_tokens"] * r["current_version_token_fraction"] for r in refresh
            ) / sum(r["active_tokens"] for r in refresh)
        output.append(values)
    return {
        "scope": "Postprocessed completed-run capacity metrics; update is 1-based. Warmup is a timing exclusion, not a compiler guarantee.",
        "allocation": allocation,
        "warmup_updates": warmup,
        "rows": output,
    }
