"""Summarize retained MILES KDA measurements without treating missing work as success."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
        "min": values[0],
        "max": values[-1],
    }


def window_details(row, path):
    """Keep sampled occupancy separate from HTTP and application timings."""
    samples = row["samples"]
    if not samples or "admitted" not in samples[0]:
        return {}
    details = {
        "responses_per_second": row["responses"] / row["seconds"],
        "output_lengths": distribution([s["output_tokens"] for s in samples]),
        "prompt_lengths": distribution([s["prompt_tokens"] for s in samples]),
        "admission_seconds": distribution([s["admitted"] - s["submitted"] for s in samples]),
        "admitted_to_processed_seconds": distribution([s["response_processed"] - s["admitted"] for s in samples]),
        "reward_seconds": distribution([s["reward_seconds"] for s in samples]),
        "processed_to_group_assembled_seconds": distribution(
            [s["assembled"] - s["response_processed"] for s in samples]
        ),
        "group_tail_seconds": distribution([g["tail_seconds"] for g in row["groups"]]),
        "timing_definition": "Admission is the MILES generate hook; admitted-to-processed includes HTTP dispatch, engine generation, transfer and response decoding. It is not pure GPU decode time.",
    }
    timeline = path / "engine-timeline.jsonl"
    if not timeline.exists():
        return details
    start = row["time_origin_monotonic"]
    finish = start + row["seconds"]
    gauges = {name: [] for name in ("num_running_reqs", "num_queue_reqs", "num_retracted_reqs", "token_usage")}
    rows = [json.loads(line) for line in timeline.read_text().splitlines()]
    selected = [r for r in rows if start <= r["monotonic"] <= finish]
    for r in selected:
        for name, values in gauges.items():
            matches = re.findall(r"^sglang:" + name + r"(?:\{[^\n]*?\})? ([^\n]+)$", r.get("metrics", ""), re.M)
            if matches:
                values.append(sum(float(v) for v in matches))
    details["engine_gauges"] = {k: distribution(v) for k, v in gauges.items()}
    details["engine_sampling"] = {
        "polls": len(selected),
        "errors": sum("error" in r for r in selected),
        "definition": "Approximately 1s HTTP metric snapshots; distributions are sampled, not exact time-weighted occupancy. Engine gauges can update less often than polling.",
    }
    if selected:
        offset = statistics.median([r["unix"] - r["monotonic"] for r in selected])
        http = []
        for file in path.glob("http_post_worker_*.jsonl"):
            for line in file.read_text().splitlines():
                record = json.loads(line)
                if start <= record["request_started_unix"] - offset <= finish:
                    http.append(record)
        details["http_worker"] = {
            "requests": len(http),
            "json_decode_seconds": distribution([r["json_decode_seconds"] for r in http]),
            "response_bytes": sum(r["response_bytes"] for r in http),
            "definition": "Worker records matched by measured-window wall time; request_id is absent, so individual response-processing stages cannot be joined exactly to sample IDs.",
        }
    return details


def runtime_audit(path, mode):
    result = {}
    for name in ("warmed-state.json", "fixed-0-state.json", "fixed-1-state.json", "final-state.json"):
        if not (path / name).exists():
            continue
        state = json.loads((path / name).read_text())
        audits = [s["kda_benchmark"] for s in state.get("internal_states", []) if "kda_benchmark" in s]
        for audit in audits:
            if not audit["loader_context_fix"]:
                raise ValueError("Loader-context fix missing")
            if any(bool(t["installed_policy"]) != (mode == "pinned") for t in audit["targets"]):
                raise ValueError("Actual policy installation differs")
        result[name] = [
            {k: a[k] for k in ("targets", "loader_context_fix", "gpu_name", "allocated_bytes", "peak_allocated_bytes")}
            for a in audits
        ]
    return result


def summarize(root):
    result = {"status": "IN PROGRESS", "precisions": {}}
    for precision in ("bf16", "combined"):
        folder = root / "speed/kda-throughput-20261116" / precision
        provenance = root / "prepared/kda-throughput-provenance.json"
        if provenance.exists():
            cases = json.loads(provenance.read_text())["cases"]
            folder = Path(next(case["output_root"] for case in cases if case["precision"] == precision))
        modes = {}
        for mode in ("pinned", "native"):
            path = folder / mode
            records = [json.loads(p.read_text()) for p in sorted(path.glob("fixed-[0-9].json"))]
            for row in records:
                if row["warmup"] or row["responses"] != 64 or row["output_tokens"] != 64 * 4096:
                    raise ValueError("Incomplete or incompatible fixed-token record")
                if any(sample["output_tokens"] != 4096 for sample in row["samples"]):
                    raise ValueError("Per-response fixed counts differ")
            modes[mode] = {
                "repeats": len(records),
                "records": records,
                "window_details": [window_details(r, path) for r in records],
                "runtime_audit": runtime_audit(path, mode),
                "seconds": [r["seconds"] for r in records],
                "output_tokens_per_second": [r["output_tokens_per_second"] for r in records],
                "provenance": {},
            }
            for name in (
                "requested-server-options.json",
                "warmed-state.json",
                "cache-provenance.json",
                "final-state.json",
            ):
                if (path / name).exists():
                    modes[mode]["provenance"][name] = {
                        "path": str(path / name),
                        "sha256": hashlib.sha256((path / name).read_bytes()).hexdigest(),
                    }
            if (path / "natural.json").exists():
                modes[mode]["natural_length_workload"] = json.loads((path / "natural.json").read_text())
                modes[mode]["natural_details"] = window_details(modes[mode]["natural_length_workload"], path)
        item = {"status": "IN PROGRESS", "conditions": modes}
        if provenance.exists():
            item["launch_provenance"] = next(case for case in cases if case["precision"] == precision)
        for name in ("gpu.json", "inputs.json", "preflight.json", "completed.json"):
            file = folder / name
            if file.exists():
                item.setdefault("artifacts", {})[name] = {
                    "path": str(file),
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                }
        if (folder / "gpu.json").exists():
            item["gpu"] = json.loads((folder / "gpu.json").read_text())
        if all(modes[m]["repeats"] == 2 for m in modes):
            pinned, native = modes["pinned"]["seconds"], modes["native"]["seconds"]
            pflags = json.loads((folder / "pinned/requested-server-options.json").read_text())
            nflags = json.loads((folder / "native/requested-server-options.json").read_text())
            if pflags != nflags:
                raise ValueError("Server settings differ within a precision pair")
            ratios = [n / p for n, p in zip(native, pinned, strict=True)]
            item.update(
                status="MEASURED",
                pinned_over_native_throughput_ratio=statistics.median(native) / statistics.median(pinned),
                paired_ratios=ratios,
                ratio_range=[min(ratios), max(ratios)],
                definition="native seconds / pinned seconds for equal fixed output counts; >1 favors pinned",
            )
        result["precisions"][precision] = item
    if all(item["status"] == "MEASURED" for item in result["precisions"].values()):
        result["status"] = "MEASURED"
    result["limitations"] = [
        "One precision pair per allocation, at most one benchmark B300 at a time; the two precision pairs may use different physical GPUs. Every pinned/native ratio uses one unchanged GPU within its allocation.",
        "Each precision retains its own production image, backend, KV/head precision, graphs and memory pools. No cross-precision ratio isolates KDA or quantization.",
        "64 responses/window, engine concurrency32 and pending budget128; finite backlog drains at the tail. This is not a steady-state infinite-backlog measurement.",
        "Fixed4096 output tokens rather than full16384, with actual prepared Dolci prompt lengths. Full cap retained only for separate optional natural-stopping probes.",
        "Two warmed repeats per condition, one load per condition and no reversed-order repeat within a precision; drift and order effects remain.",
        "MILES rollout components include native admission, HTTP workers, candidate20/behavior decoding, application reward and group assembly. No learner, 3-server fleet or real weight publication is measured.",
        "Natural-stopping lengths can differ with native autotuning. Their throughput is descriptive and is never used as a kernel speed ratio.",
        "Triton/Inductor/CUDA/XDG caches are fresh per condition; the image-inherited TILELANG_CACHE_DIR is shared. No cache directory isolation is claimed for TileLang.",
        "Occupancy is sampled engine request occupancy, not SM utilization. No separate pure prefill/decode GPU timer was collected; fixed-token duration includes the full MILES response path.",
        "Warmup and load/compile/autotune are outside fixed-window timing. Native finite behavior/candidate outputs are required; bitwise equality is not a gate.",
    ]
    return result


def write(root):
    result = summarize(root)
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    (output / "kda-throughput-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    text = [
        f"KDA throughput comparison — {result['status']}",
        "",
        "Measured through MILES rollout components on isolated B300 allocations. Production runs are unchanged.",
        "",
    ]
    for precision, item in result["precisions"].items():
        text += [f"{precision}: {item['status']}", ""]
        provenance = item.get("launch_provenance", {})
        if provenance:
            text += [
                f"Image `{provenance['image']}`; harness source `{provenance['source']}`; "
                f"[actual spec]({provenance['spec']}), [submission receipt]({provenance.get('submission_receipt', provenance['spec'].replace('.spec.json', '.submission.json'))}).",
                "",
            ]
        if "gpu" in item:
            text += [f"Physical GPU: `{item['gpu']['nvidia_smi'].strip()}`.", ""]
        for mode, condition in item["conditions"].items():
            values = condition["output_tokens_per_second"]
            if values:
                text += [
                    f"- {mode}: {len(values)} repeats, median {statistics.median(values):.1f} output tokens/s; range {min(values):.1f}–{max(values):.1f}."
                ]
        if item["status"] == "MEASURED":
            text += [
                f"- Pinned/native throughput: {item['pinned_over_native_throughput_ratio']:.4f}×; paired range {item['ratio_range'][0]:.4f}–{item['ratio_range'][1]:.4f}. Each window contains exactly 262144 output tokens."
            ]
        for mode, condition in item["conditions"].items():
            for repeat, detail in enumerate(condition["window_details"]):
                if not detail:
                    continue
                gauges = detail.get("engine_gauges", {})
                active = gauges.get("num_running_reqs", {})
                queue = gauges.get("num_queue_reqs", {})
                text += [
                    f"- {mode} repeat{repeat}: {detail['responses_per_second']:.3f} responses/s; "
                    f"admission median {detail['admission_seconds']['median']:.4f}s, "
                    f"reward median {detail['reward_seconds']['median']:.4f}s, "
                    f"group-tail median {detail['group_tail_seconds']['median']:.4f}s; "
                    f"sampled active mean {active.get('mean', 'unavailable')}, queue mean {queue.get('mean', 'unavailable')}."
                ]
            audit = condition.get("runtime_audit", {}).get("warmed-state.json", [])
            for scheduler in audit:
                text += [
                    f"- {mode} audit: loader-context fix enabled; peak allocator {scheduler['peak_allocated_bytes']} bytes."
                ]
                for target in scheduler["targets"]:
                    text += [
                        f"  - `{target['target']}`: installed policy `{target['installed_policy']}`; "
                        f"actual warmed configs `{json.dumps(target['cache'], sort_keys=True)}`."
                    ]
            natural = condition.get("natural_details")
            if natural:
                text += [
                    f"- {mode} separate natural-stopping probe: {natural['output_lengths']['count']} responses, "
                    f"mean output {natural['output_lengths']['mean']:.1f} tokens, "
                    f"range {natural['output_lengths']['min']}–{natural['output_lengths']['max']}; excluded from ratios."
                ]
        text += [""]
    text += [
        "Limitations:",
        "",
        *["- " + value for value in result["limitations"]],
        "",
        "Raw durations, per-response counts/timings, group tails and configuration/cache receipt hashes are in [the machine-readable report](kda-throughput-comparison.json).",
    ]
    (output / "kda-throughput-comparison.md").write_text("\n".join(text) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    write(parser.parse_args().study)
