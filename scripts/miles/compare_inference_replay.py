"""Compare frozen-token serving diagnostics with prompt-clustered intervals.

These are serving-relative differences, not errors against a trainer or evidence
of learning quality. Compare repeated controls to measure the diagnostic's floor.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def route_agreement(left_root, right_root, left, right, *, bootstrap=2000, seed=17):
    clusters = {}
    layer_changed = layer_total = None
    matched = 0
    for a, b in zip(left, right, strict=True):
        if not a.get("route_path") or not b.get("route_path"):
            continue
        matrices = []
        for root, record in ((left_root, a), (right_root, b)):
            with np.load(Path(root) / record["route_path"], allow_pickle=False) as archive:
                matrix = archive["routes"]
            if matrix.dtype != np.int32 or hashlib.sha256(matrix.tobytes()).hexdigest() != record["route_sha256"]:
                raise ValueError("Replay routing artifact disagrees with its recorded hash or dtype")
            matrices.append(matrix)
        x, y = matrices
        expected_length = a["prompt_tokens"] + a["response_tokens"] - 1
        if x.ndim != 3 or x.shape != y.shape or x.shape[0] != expected_length:
            raise ValueError("Replay routing matrices have inconsistent shapes")
        # The route for the last prompt token conditions the first response logit.
        changed = x[a["prompt_tokens"] - 1 :] != y[b["prompt_tokens"] - 1 :]
        row = np.array([changed.size, changed.sum(), len(changed), changed.any(axis=(1, 2)).sum()], dtype=np.float64)
        clusters[a["group_index"]] = clusters.get(a["group_index"], np.zeros(4)) + row
        per_layer = changed.sum(axis=(0, 2))
        total_per_layer = changed.shape[0] * changed.shape[2]
        layer_changed = per_layer if layer_changed is None else layer_changed + per_layer
        layer_total = total_per_layer if layer_total is None else layer_total + total_per_layer
        matched += 1
    result = {"matched_trajectories": matched, "coverage_fraction": matched / len(left)}
    if not clusters:
        return result
    grouped = np.stack(list(clusters.values()))
    total = grouped.sum(axis=0)
    rng = np.random.default_rng(seed)
    draws = grouped[rng.integers(0, len(grouped), size=(bootstrap, len(grouped)))].sum(axis=1)
    for name, numerator, denominator in (
        ("expert_slot_changed_fraction", 1, 0),
        ("conditioning_token_changed_fraction", 3, 2),
    ):
        result[name] = {
            "value": float(total[numerator] / total[denominator]),
            "prompt_cluster_95pct_interval": np.quantile(
                draws[:, numerator] / draws[:, denominator], [0.025, 0.975]
            ).tolist(),
        }
    result["per_layer_expert_slot_changed_fraction"] = (layer_changed / layer_total).tolist()
    result["scope"] = (
        "Ordered expert slots for inputs conditioning response logits; prompt clusters retain copies and repeated waves. Expert-set overlap is a different metric."
    )
    return result


def paired_metrics(left, right, *, bootstrap=2000, seed=17):
    if len(left) != len(right) or not left:
        raise ValueError("Replay comparisons require equally sized nonempty trajectories")
    clusters = {}
    all_deltas = []
    route_matches = []
    for a, b in zip(left, right, strict=True):
        if any(a[key] != b[key] for key in ("index", "group_index", "prompt_tokens", "response_tokens")):
            raise ValueError("Replay trajectory identities or lengths differ")
        if a.get("source_index", a["index"]) != b.get("source_index", b["index"]) or a.get("copy_index", 0) != b.get(
            "copy_index", 0
        ):
            raise ValueError("Replay trajectory copy identities differ")
        x, y = np.asarray(a["logprobs"], dtype=np.float64), np.asarray(b["logprobs"], dtype=np.float64)
        if (
            x.shape != y.shape
            or x.size != a["response_tokens"]
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("Replay logprob vectors are incomplete or nonfinite")
        delta = y - x
        all_deltas.append(delta)
        # exp(left - right) outside the usual 0.8..1.2 illustrative ratio band.
        outside = (delta < -np.log(1.2)) | (delta > -np.log(0.8))
        row = np.array([len(delta), np.abs(delta).sum(), delta.sum(), outside.sum(), np.count_nonzero(delta)])
        clusters[a["group_index"]] = clusters.get(a["group_index"], np.zeros(5)) + row
        if a.get("route_sha256") is not None and b.get("route_sha256") is not None:
            route_matches.append(a["route_sha256"] == b["route_sha256"])
    grouped = np.stack(list(clusters.values()))
    total = grouped.sum(axis=0)
    rng = np.random.default_rng(seed)
    draws = grouped[rng.integers(0, len(grouped), size=(bootstrap, len(grouped)))].sum(axis=1)
    names = (
        "mean_absolute_logprob_delta",
        "mean_signed_logprob_delta",
        "ratio_outside_0_8_to_1_2_fraction",
        "nonidentical_logprob_fraction",
    )
    metrics = {
        name: {
            "value": float(total[i] / total[0]),
            "prompt_cluster_95pct_interval": np.quantile(draws[:, i] / draws[:, 0], [0.025, 0.975]).tolist(),
        }
        for i, name in enumerate(names, 1)
    }
    absolute = np.abs(np.concatenate(all_deltas))
    return {
        "trajectories_including_repeats": len(left),
        "prompt_clusters": len(clusters),
        "tokens_including_repeats": int(total[0]),
        "metrics": metrics,
        "absolute_delta_quantiles": dict(
            zip(("p50", "p95", "p99", "maximum"), np.quantile(absolute, [0.5, 0.95, 0.99, 1]).tolist(), strict=True)
        ),
        "route_hash_changed_fraction": 1 - sum(route_matches) / len(route_matches) if route_matches else None,
        "interval_scope": "Paired resampling of prompt groups, keeping siblings and repeated waves together; not machine/day uncertainty or a trainer-agreement interval.",
    }


def core_comparison(root, reference_sha256, left, right, *, bootstrap=2000, seed=17):
    root = Path(root)
    summary = json.loads((root / "benchmark-summary.json").read_text())
    if (
        json.loads((root / "workflow.json").read_text())["status"] != "complete"
        or summary.get("reference_backend") != "core"
        or summary["reference_sha256"] != reference_sha256
    ):
        raise ValueError("Core scores must be complete and use the same captured reference")
    rows = [json.loads(line) for line in (root / "benchmark-core-reference.jsonl").read_text().splitlines()]
    scores = {row["index"]: row for row in rows}
    if len(scores) != len(rows) or len(rows) != summary["trajectories"]:
        raise ValueError("Core reference has missing or duplicate trajectories")
    reference, clusters = [], {}
    for a, b in zip(left, right, strict=True):
        row = scores[a.get("source_index", a["index"])]
        if any(row[key] != a[key] for key in ("group_index", "prompt_tokens", "response_tokens")):
            raise ValueError("Core reference trajectory identity differs")
        reference.append({**row, **{key: a[key] for key in ("index", "source_index", "copy_index") if key in a}})
        core = np.asarray(row["logprobs"], dtype=np.float64)
        before, after = np.asarray(a["logprobs"]) - core, np.asarray(b["logprobs"]) - core
        if len(core) != row["response_tokens"] or not np.isfinite(core).all():
            raise ValueError("Core reference has invalid logprobs")
        before_outside = (before < -np.log(1.2)) | (before > -np.log(0.8))
        after_outside = (after < -np.log(1.2)) | (after > -np.log(0.8))
        values = np.array(
            [len(core), (np.abs(after) - np.abs(before)).sum(), after_outside.sum() - before_outside.sum()]
        )
        clusters[a["group_index"]] = clusters.get(a["group_index"], np.zeros(3)) + values
    grouped = np.stack(list(clusters.values()))
    total = grouped.sum(axis=0)
    rng = np.random.default_rng(seed)
    draws = grouped[rng.integers(0, len(grouped), size=(bootstrap, len(grouped)))].sum(axis=1)
    change = {
        name: {
            "value": float(total[i] / total[0]),
            "prompt_cluster_95pct_interval": np.quantile(draws[:, i] / draws[:, 0], [0.025, 0.975]).tolist(),
        }
        for i, name in enumerate(("mean_absolute_error_change", "ratio_outlier_fraction_change"), 1)
    }
    return {
        "reference": str(root),
        "contract": summary,
        "left_error": paired_metrics(reference, left),
        "right_error": paired_metrics(reference, right),
        "paired_change": change,
    }


def compare(left, right, core=None):
    summaries, records = [], []
    for root in (Path(left), Path(right)):
        if json.loads((root / "workflow.json").read_text())["status"] != "complete":
            raise ValueError("Replay workflow must complete")
        summary = json.loads((root / "benchmark-summary.json").read_text())
        if not summary.get("diagnostic_only"):
            raise ValueError("Compare diagnostic replays, not sampled trajectories")
        summaries.append(summary)
        rows = []
        for repeat in range(summary["repeats"]):
            wave = [json.loads(line) for line in (root / f"benchmark-replay-{repeat}.jsonl").read_text().splitlines()]
            if len(wave) != summary["trajectories"] or len({r["index"] for r in wave}) != len(wave):
                raise ValueError("Replay wave has missing or duplicate trajectories")
            rows.extend(sorted(wave, key=lambda row: row["index"]))
        records.append(rows)
    for key in ("reference_sha256", "trajectories", "repeats", "response_tokens_per_repeat"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f"Replay reference mismatch: {key}")
    if summaries[0].get("copies", 1) != summaries[1].get("copies", 1):
        raise ValueError("Replay reference mismatch: copies")
    result = {
        "left": str(left),
        "right": str(right),
        "reference_sha256": summaries[0]["reference_sha256"],
        **paired_metrics(*records),
        "route_agreement": route_agreement(left, right, *records),
    }
    if core is not None:
        result["core_comparison"] = core_comparison(core, summaries[0]["reference_sha256"], *records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--core", type=Path, help="Completed forward-only Core reference run")
    args = parser.parse_args()
    report = json.dumps(compare(args.left, args.right, args.core), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
