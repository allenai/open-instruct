"""Summarize retained MILES KDA measurements without treating missing work as success."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(root):
    result = {"status": "IN PROGRESS", "precisions": {}}
    for precision in ("bf16", "combined"):
        folder = root / "speed/kda-throughput-20261116" / precision
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
        item = {"status": "IN PROGRESS", "conditions": modes}
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
