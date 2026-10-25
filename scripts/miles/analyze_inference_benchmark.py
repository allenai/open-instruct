"""Audit complete fixed-policy benchmark artifacts without a GPU runtime."""

import argparse
import csv
import json
from pathlib import Path

from scripts.miles import throughput_occupancy

from open_instruct.miles.rollout import benchmark_metrics


def read_json(path):
    return json.loads(path.read_text())


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        **{
            name: values[round(fraction * (len(values) - 1))]
            for name, fraction in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("maximum", 1.0))
        },
    }


def analyze(root):
    root = Path(root)
    if read_json(root / "workflow.json")["status"] != "complete":
        raise ValueError("Only completed benchmark workflows can qualify throughput")
    summary = read_json(root / "benchmark-summary.json")
    options = read_json(root / "resolved-plan.json")["miles"]
    batches = throughput_occupancy.records(root / "benchmark-batches.jsonl")
    recomputed = benchmark_metrics.summarize(batches, options["rollout_num_gpus"])
    if any(summary[key] != value for key, value in recomputed.items()):
        raise ValueError("Benchmark summary disagrees with complete collection records")
    measured = [row for row in batches if row["phase"] == "measure"]
    first = measured[0]
    start = first["finished_unix"] - (first["finished"] - first["started"])
    end = measured[-1]["finished_unix"]
    responses = [
        r for r in throughput_occupancy.records(root / "benchmark-responses.jsonl") if r["phase"] == "measure"
    ]
    for field in ("response_tokens", "route_bytes"):
        if sum(r[field] for r in responses) != summary[field]:
            raise ValueError(f"Response records disagree with measured {field}")
    if len(responses) != summary["samples"]:
        raise ValueError("Response records disagree with measured sample count")
    occupancy = throughput_occupancy.analyze(root, window=(start, end))
    occupancy["scope"] = "Sampled occupancy during fixed-policy measurement; missing coverage is not idle time."
    resources = throughput_occupancy.records(root / "benchmark-resources.jsonl")
    devices = {}
    for row in resources:
        if row["returncode"] != 0:
            continue
        for device in csv.reader(row["gpu_csv"].splitlines(), skipinitialspace=True):
            index, identity, name, *numbers = device
            points = devices.setdefault(identity, {"index": index, "name": name, "observations": []})
            points["observations"].append((row["time_unix"], [float(value) for value in numbers]))
    for device in devices.values():
        points = device.pop("observations")
        device["metrics"] = {
            key: throughput_occupancy.summarize([(t, values[i]) for t, values in points], start, end)
            for i, key in enumerate(
                ("memory_used_mib", "memory_total_mib", "utilization_percent", "power_watts", "clock_mhz")
            )
        }
    return {
        "root": str(root),
        "summary": summary,
        "resolved_engines": [
            {
                "url": server["url"],
                "settings": {
                    key: server["info"].get(key)
                    for key in (
                        "context_length",
                        "max_running_requests",
                        "max_total_tokens",
                        "max_mamba_cache_size",
                        "mem_fraction_static",
                        "chunked_prefill_size",
                        "max_prefill_tokens",
                        "enable_mixed_chunk",
                        "cuda_graph_config",
                        "schedule_policy",
                        "schedule_conservativeness",
                        "disable_radix_cache",
                        "mamba_radix_cache_strategy",
                        "dtype",
                        "attention_backend",
                        "sampling_backend",
                    )
                },
                "memory": [state.get("memory_usage") for state in server["info"].get("internal_states", [])],
                "startup": [state.get("startup_time") for state in server["info"].get("internal_states", [])],
            }
            for server in read_json(root / "benchmark-server-start.json")
        ],
        "prompt_lengths": quantiles([r["prompt_tokens"] for r in responses]),
        "response_lengths": quantiles([r["response_tokens"] for r in responses]),
        "truncated_fraction": sum(r["status"] == "truncated" for r in responses) / len(responses),
        "occupancy": occupancy,
        "hardware": devices,
        "boundary_inventory": read_json(root / "benchmark-boundaries.json"),
        "shutdown_inventory": read_json(root / "benchmark-shutdown.json"),
        "qualification_limits": "Fixed checkpoint, no trainer, rewards as configured. Delivered tokens can cross boundaries; inspect retained and unfinished work. Payload validity alone does not prove numerical equivalence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.root)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
