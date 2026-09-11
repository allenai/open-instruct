"""Summarize retained control-exercise evidence without importing the GPU runtime."""

import argparse
import json
import statistics
from pathlib import Path


def describe(values):
    return (
        dict(
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        )
        if values
        else None
    )


def correlation(left, right):
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(left, right)


def scoring(root):
    comparison = json.loads((root / "comparison.json").read_text())
    result = {
        "exact_score_gate": comparison["valid"],
        "compared_tokens": sum(row["tokens"] for row in comparison["comparisons"]),
    }
    for mode, arm in (("static", "parent"), ("dynamic", "candidate")):
        ranks = [json.loads((root / arm / f"rank{rank}.json").read_text()) for rank in (0, 1)]
        points = []
        for index in range(len(ranks[0]["passes"])):
            rows = [rank["passes"][index] for rank in ranks]
            if rows[0]["name"] != rows[1]["name"]:
                raise ValueError("Rank timing schedules differ")
            batch = index // 2
            tokens = sum(sum(rank["inputs"][batch]["lengths"]) for rank in ranks)
            misses = [
                [miss for miss in row["jit_in_memory_misses"] if "swiglu_valid_prefix" in miss["kernel"]]
                for row in rows
            ]
            points.append(
                dict(
                    name=rows[0]["name"],
                    model_tokens=tokens,
                    seconds=max(row["wall_seconds"] for row in rows),
                    swiglu_misses_by_rank=[len(value) for value in misses],
                    all_jit_misses_by_rank=[row["jit_miss_count"] for row in rows],
                    jit_miss_wall_seconds_by_rank=[row["jit_miss_wall_seconds"] for row in rows],
                )
            )
        firsts, repeats = points[::2], points[1::2]
        result[mode] = dict(
            points=points,
            first_batch_seconds=firsts[0]["seconds"],
            changing_batch_seconds=describe([row["seconds"] for row in firsts[1:]]),
            repeated_batch_seconds=describe([row["seconds"] for row in repeats]),
            changing_batch_token_time_correlation=correlation(
                [row["model_tokens"] for row in firsts[1:]], [row["seconds"] for row in firsts[1:]]
            ),
        )
    result["dynamic_no_new_swiglu_variants_after_first"] = all(
        not any(point["swiglu_misses_by_rank"]) for point in result["dynamic"]["points"][1:]
    )
    result["changing_batch_speedup"] = (
        result["static"]["changing_batch_seconds"]["mean"] / result["dynamic"]["changing_batch_seconds"]["mean"]
    )
    return result


def live(root):
    document = json.loads((root / "audit.json").read_text())
    times = document["scoring_by_rank"]
    points = [
        dict(
            rollout_id=a["rollout_id"],
            seconds=max(a["seconds"], b["seconds"]),
            model_tokens=a["model_tokens"] + b["model_tokens"],
        )
        for a, b in zip(*times, strict=True)
    ]
    report = {
        key: value
        for key, value in document.items()
        if key not in ("training", "scoring_by_rank", "native_log_timing")
    }
    report["scoring_points"] = points
    report["post_first_four_score_seconds"] = describe([row["seconds"] for row in points[4:]])
    report["post_first_four_score_token_correlation"] = correlation(
        [row["model_tokens"] for row in points[4:]], [row["seconds"] for row in points[4:]]
    )
    report["consumed_response_tokens_per_cycle_second"] = (
        document["consumed_response_tokens"] / document["measured_cycle_seconds"]
    )
    report["updates_per_cycle_hour"] = document["updates"] * 3600 / document["measured_cycle_seconds"]
    report["policy_lag_counts"] = {
        str(lag): sum(row["policy_lags"].count(lag) for row in document["training"]) for lag in (0, 1)
    }
    warm = document["training"][4:]
    if warm:
        samples_per_collection = document["consumed_samples"] // document["updates"]
        warm_seconds = document["post_first_four_cycle_seconds"]["mean"] * len(warm)
        warm_tokens = sum(row["summary"]["mean_response_tokens"] * samples_per_collection for row in warm)
        report["post_first_four_consumed_response_tokens"] = warm_tokens
        report["post_first_four_consumed_tokens_per_cycle_second"] = warm_tokens / warm_seconds
        report["samples_per_collection"] = samples_per_collection
    report["native_generation_timing"] = document["native_log_timing"]["phases"]["generation"]
    return report


def analyze(root):
    result = {"scope": "Bounded performance/configuration trial; cold isolated caches; no learning or restart claim."}
    paths = {
        "scoring": root / "scores/scoring",
        "sync": root / "scheduling/sync",
        "async": root / "scheduling/async",
        "controls": root / "controls/controls",
    }
    for name, path in paths.items():
        try:
            result[name] = scoring(path) if name == "scoring" else live(path)
        except FileNotFoundError as error:
            result[name] = {"incomplete": True, "missing": str(error.filename)}
    if all(result[name].get("passed") for name in ("sync", "async")):
        result["async_vs_sync"] = {
            "post_first_four_update_speedup": result["sync"]["post_first_four_cycle_seconds"]["mean"]
            / result["async"]["post_first_four_cycle_seconds"]["mean"],
            "all_update_speedup": result["sync"]["cycle_seconds"]["mean"] / result["async"]["cycle_seconds"]["mean"],
            "consumed_token_throughput_ratio": result["async"]["consumed_response_tokens_per_cycle_second"]
            / result["sync"]["consumed_response_tokens_per_cycle_second"],
            "note": "Different completion order may change sampled prompts and lengths. Cycle excludes startup/eval/save.",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission-results", type=Path, help="Downloaded admission task directory")
    args = parser.parse_args()
    report = analyze(args.results)
    if args.admission_results:
        report["admission64"] = {}
        for mode in ("sync", "async"):
            root = args.admission_results / f"{mode}-admission64"
            if (root / "audit.json").exists():
                measured = live(root)
                baseline = report.get(mode, {})
                if baseline.get("passed"):
                    measured["warm_token_throughput_vs_batch16"] = (
                        measured["post_first_four_consumed_tokens_per_cycle_second"]
                        / baseline["post_first_four_consumed_tokens_per_cycle_second"]
                    )
                    measured["baseline_comparison_note"] = (
                        "Both admission and optimizer batch grow; this is useful throughput, not an isolated admission effect."
                    )
                report["admission64"][mode] = measured
            else:
                report["admission64"][mode] = {"incomplete": True, "missing": str(root / "audit.json")}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: {key: value for key, value in entry.items() if key not in ("points", "scoring_points")}
                if isinstance(entry, dict)
                else entry
                for name, entry in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
