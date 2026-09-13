"""Time-weighted summaries of sampled pipeline occupancy, with explicit coverage."""

import json
import math
from pathlib import Path

from open_instruct.miles import topology


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(points, start, end, *, max_hold=10.0):
    """Hold a sample only until the next observation or max_hold; gaps stay missing."""
    if end <= start or max_hold <= 0:
        raise ValueError("Observation window and maximum hold must be positive")
    points = sorted(points, key=lambda point: point[0])
    weighted = []
    for i, (timestamp, value) in enumerate(points):
        next_time = points[i + 1][0] if i + 1 < len(points) else end
        duration = min(next_time, end, timestamp + max_hold) - max(timestamp, start)
        if duration > 0 and value is not None and math.isfinite(value):
            weighted.append((float(value), duration))
    coverage = sum(duration for _, duration in weighted)
    if not coverage:
        return {"coverage_fraction": 0, "mean": None, "p95": None, "maximum": None, "empty_fraction": None}
    accumulated = 0.0
    percentile = None
    for value, duration in sorted(weighted):
        accumulated += duration
        if accumulated >= 0.95 * coverage:
            percentile = value
            break
    return {
        "coverage_fraction": coverage / (end - start),
        "mean": sum(value * duration for value, duration in weighted) / coverage,
        "p95": percentile,
        "maximum": max(value for value, _ in weighted),
        "empty_fraction": sum(duration for value, duration in weighted if value == 0) / coverage,
    }


def node_roles(root):
    """Resolve observed replica indices through runtime IP assignment, not plan order."""
    root = Path(root)
    placements = list((root / "cluster").glob("*/placement-*.json"))
    if not placements:
        plan = root / "plan.json"
        if plan.exists():
            allocation = json.loads(plan.read_text()).get("allocation", {})
            if allocation.get("replicas") == 1:
                return {"0": allocation["nodes"][0]}
        return {}
    result = {}
    for path in placements:
        placement = json.loads(path.read_text())
        nodes = [json.loads(p.read_text()) for p in path.parent.glob("node-*.json")]
        assignments = topology.assign(placement["layout"], [node["address"] for node in nodes])
        index = path.stem.removeprefix("placement-")
        value = assignments[placement["address"]]
        if index in result and result[index] != value:
            raise ValueError("Replica roles differ across retained attempts; analyze one attempt at a time")
        result[index] = value
    return result


def analyze(root, *, warmup=6):
    root = Path(root)
    stages = records(root / "checkpoints/driver_timing.jsonl")
    chosen = [
        r for r in stages if r["stage"] in ("generation_wait", "training", "publication") and r["rollout_id"] >= warmup
    ]
    if not chosen or not all(r["passed"] for r in chosen):
        raise ValueError("No complete warm-window driver stages")
    start = min(r["started_unix"] for r in chosen)
    end = max(r["started_unix"] + r["seconds"] for r in chosen)
    pipeline_path = root / "checkpoints/pipeline_occupancy.jsonl"
    pipeline = records(pipeline_path) if pipeline_path.exists() else []
    result = {
        "scope": "Sampled occupancy during warm normal cycles; not hardware GPU utilization or exact per-request waits.",
        "start_unix": start,
        "end_unix": end,
        "max_sample_hold_seconds": 10,
        "pipeline": {
            key: summarize([(r["time_unix"], r.get(key)) for r in pipeline], start, end)
            for key in (
                "producer_owned_groups",
                "producer_active_group_tasks",
                "producer_admission_open",
                "producer_unfinished_samples",
                "completed_queue_groups",
                "completed_queue_capacity_groups",
                "http_active_requests",
                "http_waiting_requests",
                "http_capacity_requests",
            )
        },
        "engines": {},
        "node_roles": node_roles(root),
    }
    engine_rows = [r for path in (root / "checkpoints").glob("engine_occupancy*.jsonl") for r in records(path)]
    for identity in sorted({r["engine"] for r in engine_rows if "engine" in r}):
        values = [r for r in engine_rows if r.get("engine") == identity]
        series_keys = sorted({(s["name"], s.get("labels") or "") for r in values for s in r.get("series", [])})
        summaries = []
        for key in series_keys:
            points = []
            for r in values:
                matches = [s for s in r.get("series", []) if (s["name"], s.get("labels") or "") == key]
                points.append((r["time_unix"], matches[0]["value"] if len(matches) == 1 else None))
            summaries.append({"name": key[0], "labels": key[1], **summarize(points, start, end)})
        result["engines"][identity] = {
            "series": summaries,
            "failed_observations": sum("error" in r for r in values if start <= r["time_unix"] <= end),
        }
    result["hardware"] = {}
    for path in sorted((root / "checkpoints").glob("gpu_usage_node*.jsonl")):
        observations = records(path)
        identities = sorted({d["uuid"] for r in observations for d in r.get("devices", [])})
        devices = []
        for identity in identities:
            values = []
            index = None
            for row in observations:
                matches = [d for d in row.get("devices", []) if d["uuid"] == identity]
                device = matches[0] if len(matches) == 1 else {}
                index = device.get("index", index)
                values.append((row["time_unix"], device))
            devices.append(
                {
                    "uuid": identity,
                    "index": index,
                    "metrics": {
                        key: summarize([(t, d.get(key)) for t, d in values], start, end)
                        for key in ("utilization.gpu", "utilization.memory", "memory.used", "memory.total")
                    },
                }
            )
        result["hardware"][path.stem] = devices
    return result
