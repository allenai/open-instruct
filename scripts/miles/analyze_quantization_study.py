"""Offline fixed-token pairing and separate-year pinned olmo-eval AIME32 analysis.

Reads artifacts only. Missing/invalid evidence raises rather than becoming zero.
Bootstrap intervals resample questions, never individual draws. They describe this
small question set and cannot establish general mathematical capability.
"""

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

KEYS = ("prompt_tokens", "output_tokens_per_request", "concurrency", "repeat")
GRID = set(itertools.product((512, 2048), (256, 4096, 16384), (1, 8, 16), range(3)))
SCORER = "minerva_math_flex"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def quantiles(values):
    ordered = sorted(values)
    return {str(p): ordered[round((len(ordered) - 1) * p)] for p in (0, 0.5, 0.95, 1)}


def question_interval(values, *, seed=17, repeats=10000):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [means[int(0.025 * repeats)], means[min(repeats - 1, int(0.975 * repeats))]]


def speed_cases(path, precision, *, legacy_original=None, legacy_validation=None):
    """Legacy cases have source assertions but no persisted per-request counts."""
    rows = read_rows(path)
    legacy = []
    if legacy_original is not None:
        validation = json.loads(Path(legacy_validation).read_text())
        if digest(legacy_original) != validation["original_measurements_sha256"]:
            raise ValueError("Original 45 measurements hash mismatch")
        original = Path(legacy_original).read_bytes()
        legacy = read_rows(legacy_original)
        if len(legacy) != 45 or not Path(path).read_bytes().startswith(original):
            raise ValueError("Original 45 measurements not preserved")
    cases = {}
    for index, row in enumerate(rows):
        key = tuple(row[k] for k in KEYS)
        if key not in GRID or key in cases or row["precision"] != precision:
            raise ValueError("Duplicate, unexpected or wrong-precision speed case")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("Invalid speed duration")
        if index < len(legacy):
            context = 18432
            count_evidence = "legacy benchmark runtime assertion; per-request counts not persisted"
        else:
            context = row["context_length"]
            counts = row["completion_counts"]
            if context != 18434 or len(counts) != key[2] or set(counts.values()) != {key[1]}:
                raise ValueError("Truncated speed output or wrong context")
            if not row.get("provenance"):
                raise ValueError("Missing speed provenance")
            count_evidence = "persisted per-request exact counts"
        cases[key] = {
            **dict(zip(KEYS, key)),
            "seconds": row["seconds"],
            "output_tokens_per_second": key[1] * key[2] / row["seconds"],
            "context_length": context,
            "count_evidence": count_evidence,
            "provenance": row.get("provenance"),
            "ttft_seconds": row.get("ttft_seconds"),
        }
    if set(cases) != GRID:
        raise ValueError(f"Incomplete speed grid: {len(cases)}/54")
    return cases


def warmed_speed_cases(path, precision):
    """Require durable same-process exact-workload warmup before every measurement."""
    path = Path(path)
    cases = speed_cases(path, precision)
    complete = json.loads(path.with_name("complete.json").read_text())
    if complete["cases"] != 54 or complete["options"]["context_length"] != 18434:
        raise ValueError("Missing complete warmed grid receipt")
    warmups = read_rows(path.with_name("warmups.jsonl"))
    by_id = {row["id"]: row for row in warmups}
    if len(by_id) != len(warmups):
        raise ValueError("Duplicate warmup ID")
    referenced = set()
    for row in read_rows(path):
        warmup = by_id[row["warmup_id"]]
        referenced.add(warmup["id"])
        provenance = row["provenance"]
        if (
            provenance != warmup["provenance"]
            or provenance["timing_protocol"] != "exact-workload-warmup-v1"
            or provenance["context_length"] != 18434
            or not provenance.get("process_id")
            or any(row[key] != warmup[key] for key in KEYS[:3])
            or set(warmup["completion_counts"]) != {str(i) for i in range(row["concurrency"])}
            or set(warmup["completion_counts"].values()) != {row["output_tokens_per_request"]}
            or set(row["completion_counts"]) != {str(i) for i in range(row["concurrency"])}
        ):
            raise ValueError("Warmup workload, counts or process provenance mismatch")
        for record in (warmup, row):
            times = [record[key] for key in ("wall_start", "wall_end", "monotonic_start", "monotonic_end", "seconds")]
            if not all(math.isfinite(value) for value in times) or record["seconds"] <= 0:
                raise ValueError("Invalid timing interval")
            if record["wall_end"] < record["wall_start"] or not math.isclose(
                record["monotonic_end"] - record["monotonic_start"], record["seconds"], abs_tol=1e-6
            ):
                raise ValueError("Inconsistent timing interval")
        if warmup["monotonic_end"] > row["monotonic_start"] or warmup["wall_end"] > row["wall_start"]:
            raise ValueError("Warmup overlaps measurement")
    if len(referenced) < 18:
        raise ValueError("Missing exact-workload warmups")
    return cases, {
        "warmups": len(warmups),
        "referenced_warmups": len(referenced),
        "sources": {
            str(p): digest(p) for p in (path, path.with_name("complete.json"), path.with_name("warmups.jsonl"))
        },
    }


def paired_speed(cases):
    """Ratio >1 means the named arm is faster than BF16 auto for equal work."""
    baseline = cases["bf16"]
    comparisons = {}
    for precision, arm in cases.items():
        if precision == "bf16":
            continue
        pairs = []
        for key in sorted(GRID):
            a, b = baseline[key], arm[key]
            pairs.append(
                {
                    **dict(zip(KEYS, key)),
                    "speed_ratio_vs_bf16": a["seconds"] / b["seconds"],
                    "bf16_context": a["context_length"],
                    "arm_context": b["context_length"],
                    "same_context": a["context_length"] == b["context_length"],
                }
            )
        groups = []
        for p, o, c in itertools.product((512, 2048), (256, 4096, 16384), (1, 8, 16)):
            selected = [x["speed_ratio_vs_bf16"] for x in pairs if tuple(x[k] for k in KEYS[:3]) == (p, o, c)]
            groups.append(
                {
                    "prompt": p,
                    "output": o,
                    "concurrency": c,
                    "median_ratio": statistics.median(selected),
                    "ratios": selected,
                }
            )
        comparisons[precision] = {"pairs": pairs, "workloads": groups}
    return {
        "comparisons": comparisons,
        "caveats": [
            "BF16 original45 context18432; recovered9 context18434. Other arms context18434. Mixed-context pairs are explicitly flagged.",
            "Two slots change allocator capacity; identical measured lengths do not establish identical allocation or graph behavior. No causal precision-only claim for mixed settings.",
            "Only three repeats per workload. Paired ratios are descriptive; GPU allocation/order and shape-specific compilation may affect them.",
            "Timing starts after explicit warmup/cache flush; inspect per-job logs for shape-specific compilation within timed intervals before accepting compile-excluded results.",
            "Synthetic outputs are performance probes, never accuracy samples.",
        ],
    }


def repeated_text(text):
    """Diagnostic: >=80% repeated whitespace 4-grams in at least 64 words."""
    words = text.split()
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    return len(words) >= 64 and 1 - len(set(grams)) / len(grams) >= 0.8


def aime_year(predictions, snapshot, metrics, year, expected_hash):
    if digest(snapshot) != expected_hash:
        raise ValueError("AIME snapshot hash mismatch")
    data = json.loads(Path(snapshot).read_text())
    gold = {str(row["id"]): str(row["answer"]).lstrip("0") or "0" for row in data if row["year"] == year}
    rows = read_rows(predictions)
    if len(gold) != 30 or len(rows) != 30 or {str(r["native_id"]) for r in rows} != set(gold):
        raise ValueError("AIME requires exact 30 source questions once per year")
    parsed, truncated, repeated, counts, finishes, per_question = [], [], [], [], Counter(), []
    for row in rows:
        if str(row["label"]).lstrip("0") != gold[str(row["native_id"])].lstrip("0"):
            raise ValueError("AIME label mismatch")
        outputs = row["model_output"]
        if row.get("scoring_error") or len(outputs) != 32:
            raise ValueError("Missing/failed AIME draws")
        correct = []
        for output in outputs:
            if output.get("scoring_errors"):
                raise ValueError("AIME sample scoring failure")
            score = output["sample_metrics"][SCORER][SCORER]
            if score not in (0, 1):
                raise ValueError("Invalid sample score")
            count, finish = output["completion_tokens"], output["finish_reason"]
            if type(count) is not int or not 0 < count <= 16384 or finish not in ("stop", "length"):
                raise ValueError("Invalid completion accounting or provider failure")
            correct.append(score)
            parsed.append(output.get("extracted_answer") not in (None, ""))
            truncated.append(finish == "length" or count == 16384)
            repeated.append(repeated_text(output.get("original_text", output["text"])))
            counts.append(count)
            finishes[finish] += 1
        per_question.append(
            {
                "id": row["native_id"],
                "correct": sum(correct),
                "draws": 32,
                "pass1": statistics.mean(correct),
                "pass32": int(any(correct)),
            }
        )
    entries = [t for t in json.loads(Path(metrics).read_text())["tasks"] if t["task"] == f"aime_{year}:pass_at_32"]
    if len(entries) != 1:
        raise ValueError("Missing/ambiguous year metrics entry")
    task = entries[0]
    if task["num_instances"] != 30 or task.get("instances_failed", 0) or task.get("error_summary"):
        raise ValueError("Evaluator task incomplete")
    timing = {
        k: task.get(k)
        for k in (
            "first_request_at",
            "last_completed_at",
            "duration_seconds",
            "attributed_inference_seconds",
            "completion_tokens_total",
        )
    }
    if timing["completion_tokens_total"] != sum(counts):
        raise ValueError("Evaluator token totals differ from predictions")
    summary = {}
    for metric, key in (("pass_at_1_estimate", "pass1"), ("observed_pass_at_32", "pass32")):
        values = [q[key] for q in per_question]
        summary[metric] = {"mean": statistics.mean(values), "question_bootstrap_95pct": question_interval(values)}
    for key in ("pass_at_1", "pass_at_32"):
        scorer_values = task["metrics"].get(key, {})
        target = summary["pass_at_1_estimate" if key == "pass_at_1" else "observed_pass_at_32"]["mean"]
        if SCORER not in scorer_values or not math.isclose(scorer_values[SCORER], target, abs_tol=1e-9):
            raise ValueError("Harness and sample scores disagree or are missing")
    duration = timing["duration_seconds"]
    inference = timing["attributed_inference_seconds"]
    return {
        "year": year,
        "questions": 30,
        "draws_per_question": 32,
        **summary,
        "parsing_failure_rate": 1 - statistics.mean(parsed),
        "truncation_rate": statistics.mean(truncated),
        "repetition_rate": statistics.mean(repeated),
        "repetition_definition": repeated_text.__doc__,
        "actual_output_tokens": {
            "total": sum(counts),
            "mean": statistics.mean(counts),
            "quantiles": quantiles(counts),
        },
        "finish_reasons": dict(finishes),
        "timing": timing,
        "tokens_per_task_second": sum(counts) / duration if duration and duration > 0 else None,
        "tokens_per_attributed_inference_second": sum(counts) / inference if inference and inference > 0 else None,
        "harness_metrics": task["metrics"],
        "per_question": per_question,
        "sources": {str(p): digest(p) for p in (predictions, snapshot, metrics)},
        "caveats": [
            "Only 30 questions per year. Question-bootstrap intervals may degenerate at all-zero/all-one outcomes; they do not prove a population bound.",
            "Pass@32 is observed any-correct among 32 draws. Pass@1 is the mean of per-question sample accuracy.",
            "Task duration includes evaluation processing; attributed inference time may sum overlapping requests and is not device walltime.",
            "Parsing uses the pinned extractor, not a new boxed-answer parser; repetition is a whitespace diagnostic.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    speed = commands.add_parser("speed")
    speed.add_argument("--measurements", action="append", required=True, help="precision=measurements.jsonl")
    speed.add_argument("--legacy-original", type=Path)
    speed.add_argument("--legacy-validation", type=Path)
    speed.add_argument(
        "--warmed", action="store_true", help="Require exact-workload warmup receipts for all four arms"
    )
    aime = commands.add_parser("aime")
    aime.add_argument("--predictions", type=Path, required=True)
    aime.add_argument("--snapshot", type=Path, required=True)
    aime.add_argument("--metrics", type=Path, required=True)
    aime.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    aime.add_argument("--snapshot-sha256", required=True)
    aime.add_argument("--precision", required=True, choices=("bf16", "bf16-control", "fp8", "nvfp4"))
    for child in (speed, aime):
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "aime":
        report = aime_year(args.predictions, args.snapshot, args.metrics, args.year, args.snapshot_sha256)
        report["precision"] = args.precision
    else:
        if args.warmed == bool(args.legacy_original or args.legacy_validation):
            parser.error("Choose --warmed or both legacy provenance paths")
        if not args.warmed and not (args.legacy_original and args.legacy_validation):
            parser.error("Legacy analysis requires both provenance paths")
        paths = dict(item.split("=", 1) for item in args.measurements)
        if set(paths) != {"bf16", "bf16-control", "fp8", "nvfp4"} or len(args.measurements) != 4:
            parser.error("Require exactly the four approved precision measurement paths")
        cases = {
            p: speed_cases(
                path,
                p,
                **(
                    {"legacy_original": args.legacy_original, "legacy_validation": args.legacy_validation}
                    if p == "bf16"
                    else {}
                ),
            )
            for p, path in paths.items()
        }
        warmup_receipts = {}
        if args.warmed:
            for precision, path in paths.items():
                cases[precision], warmup_receipts[precision] = warmed_speed_cases(path, precision)
        report = paired_speed(cases)
        if args.warmed:
            report["warmup_receipts"] = warmup_receipts
            report["caveats"][:2] = [
                "All four warmed grids use context18434; original mixed-context and short-warmup grids remain separate sensitivity results.",
                "Every measurement follows same-process exact-workload warmup and cache flush. No compiler-event trace was collected; warmup does not prove absence of all runtime compilation.",
            ]
        report["sources"] = {path: digest(path) for path in paths.values()}
        report["auxiliary_provenance"] = {
            str(p): digest(p)
            for path in paths.values()
            for pattern in ("complete.json", "server-info.json", "provenance-*.json")
            for p in Path(path).parent.glob(pattern)
        }
        report["cases"] = {p: list(arm.values()) for p, arm in cases.items()}
    report["analysis_source_sha256"] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
