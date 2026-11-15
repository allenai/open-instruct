"""Validate same-allocation fixed-length serving pilots before selecting capacity."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def buffer_bytes(value):
    return sum(buffer_bytes(item) for item in value) if isinstance(value, list) else value["bytes"]


def analyze(root):
    root = Path(root).resolve()
    summary = json.loads((root / "pilot-complete.json").read_text())
    attempt = Path(summary["attempt"]).resolve()
    if attempt.parent != root:
        raise ValueError("Pilot attempt must belong to this output root")
    rows, artifacts = [], {}
    identity = None

    def read(path):
        artifacts[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    for concurrency in summary["measured_capacities"]:
        folder = attempt / str(concurrency)
        complete = read(folder / "complete.json")
        options = complete["options"]
        current_identity = (complete["job_id"], complete["device"], complete["source_sha256"])
        if identity is None:
            identity = current_identity
        if current_identity != identity:
            raise ValueError("Capacities were measured on different allocations or sources")
        if options["max_running_requests"] != concurrency or options["context_length"] != 18434:
            raise ValueError("Unexpected capacity or context")
        if options["max_total_tokens"] < concurrency * 18434:
            raise ValueError("Insufficient full-length token capacity")
        repeats = []
        for name in ["warmup", "repeat-0", "repeat-1", "repeat-2"]:
            record = read(folder / f"{name}.json")
            counts = record["completion_counts"]
            if set(counts) != {str(i) for i in range(concurrency)} or set(counts.values()) != {16384}:
                raise ValueError("Missing requests or incorrect completion lengths")
            if record["num_retractions"] != 0:
                raise ValueError("Retracted or unobserved requests cannot qualify stable capacity")
            seconds = record["seconds"]
            rate = record["output_tokens_per_second"]
            if not math.isfinite(seconds) or seconds <= 0 or not math.isclose(rate, concurrency * 16384 / seconds):
                raise ValueError("Invalid timing or throughput")
            if name == "warmup":
                continue
            state = record["after"]["internal_states"][0]
            telemetry = state["capacity_pilot"]
            repeats.append(
                {
                    "seconds": seconds,
                    "output_tokens_per_second": rate,
                    "peak_allocated_bytes": telemetry["peak_allocated_bytes"],
                    "free_bytes": telemetry["free_bytes"],
                    "kv_bytes": telemetry["full_kv_bytes"],
                    "recurrent_bytes": sum(
                        buffer_bytes(telemetry[key]) for key in ("recurrent_temporal", "recurrent_conv")
                    ),
                    "request_latency_seconds": record["request_latency_seconds"],
                    "decode_seconds_per_token": record["decode_seconds_per_token"],
                    "num_retractions": record["num_retractions"],
                    "resolved_decode_graph": state.get("cuda_graph_config", {}).get("decode"),
                    "reported_captured_graph_sizes": state.get("decode_cuda_graph_bs"),
                    "graph_memory_gb": state.get("decode_cuda_graph_memory_gb"),
                }
            )
        mean = sum(row["output_tokens_per_second"] for row in repeats) / 3
        if not complete["stable"] or not math.isclose(mean, complete["mean_output_tokens_per_second"]):
            raise ValueError("Capacity summary disagrees with raw measurements")
        rows.append(
            {"concurrency": concurrency, "options": options, "mean_output_tokens_per_second": mean, "repeats": repeats}
        )
    best = max(rows, key=lambda row: row["mean_output_tokens_per_second"])
    if best["options"] != summary["best_measured_stable_options"]:
        raise ValueError("Selected settings do not maximize measured stable throughput")
    return {
        "passed": True,
        "root": str(root),
        "allocation": identity,
        "capacities": rows,
        "best_concurrency": best["concurrency"],
        "best_options": best["options"],
        "artifact_sha256": artifacts,
        "caveats": [
            "Synthetic exact-length throughput; representative AIME batching remains required.",
            "Configured graph sizes and graph memory are retained; an empty captured-size report does not prove graph coverage.",
            "Serving pools/graph capacity differ; gains across capacities are not quantization-only speedups.",
            "Three repeats on one allocation; no cross-device generalization or accuracy qualification.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.root), indent=2) + "\n")


if __name__ == "__main__":
    main()
