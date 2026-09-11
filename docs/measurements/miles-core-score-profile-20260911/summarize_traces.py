import collections
import hashlib
import json
import pathlib

ROOT = pathlib.Path(
    "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1/score-profile-20260911-v1"
)


def union(xs):
    total = 0.0
    end = None
    for a, b in sorted(xs):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


result = []
for rank in (0, 1):
    p = ROOT / f"rank{rank}-trace.json"
    raw = p.read_bytes()
    d = json.loads(raw)
    events = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    cats = collections.Counter(e.get("cat", "") for e in events)
    ranges = {}
    for name in ("core_score_forward", "core_score_logprobs"):
        es = [e for e in events if e.get("name") == name]
        xs = [(e["ts"], e["ts"] + e["dur"]) for e in es]
        ranges[name] = {
            "count": len(es),
            "cpu_sum_seconds": sum(e["dur"] for e in es) / 1e6,
            "cpu_union_seconds": union(xs) / 1e6,
            "durations_seconds": [e["dur"] / 1e6 for e in es],
        }
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    collective = [e for e in gpu if "nccl" in e.get("name", "").lower()]
    other = [e for e in gpu if "nccl" not in e.get("name", "").lower()]

    def interval(es):
        return [(e["ts"], e["ts"] + e["dur"]) for e in es]

    gu = union(interval(gpu)) / 1e6
    cu = union(interval(collective)) / 1e6
    ou = union(interval(other)) / 1e6
    groups = collections.defaultdict(lambda: {"events": 0, "duration_sum_seconds": 0})
    for e in gpu:
        g = groups[e["name"]]
        g["events"] += 1
        g["duration_sum_seconds"] += e["dur"] / 1e6
    result.append(
        {
            "rank": rank,
            "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "trace_bytes": len(raw),
            "gpu_devices": sorted({str(e.get("args", {}).get("device")) for e in gpu}),
            "duration_event_categories": dict(cats),
            "cpu_ranges": ranges,
            "gpu": {
                "event_count": len(gpu),
                "active_union_seconds": gu,
                "nccl_union_seconds": cu,
                "non_nccl_union_seconds": ou,
                "nccl_non_nccl_overlap_seconds": cu + ou - gu,
                "nccl_event_count": len(collective),
                "span_seconds": (max(e["ts"] + e["dur"] for e in gpu) - min(e["ts"] for e in gpu)) / 1e6,
                "top_kernels_by_summed_duration": sorted(
                    groups.items(), key=lambda pair: pair[1]["duration_sum_seconds"], reverse=True
                )[:20],
            },
            "scope": "Separate warm instrumented pass. CPU ranges include host launches/waits, not isolated compute. GPU active time is union of kernel/memcpy/memset intervals, not inclusive event sum. NCCL classified by kernel name; includes synchronization/waiting and is not all EP overhead. No attribution of GPU kernels to CPU ranges.",
        }
    )
print(json.dumps({"ranks": result}, indent=2))
