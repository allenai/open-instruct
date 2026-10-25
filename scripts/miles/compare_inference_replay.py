"""Compare frozen-token serving diagnostics with prompt-clustered intervals.

These are serving-relative differences, not errors against a trainer or evidence
of learning quality. Compare repeated controls to measure the diagnostic's floor.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def paired_metrics(left, right, *, bootstrap=2000, seed=17):
    if len(left) != len(right) or not left:
        raise ValueError("Replay comparisons require equally sized nonempty trajectories")
    clusters = {}
    all_deltas = []
    route_matches = []
    for a, b in zip(left, right, strict=True):
        if any(a[key] != b[key] for key in ("index", "group_index", "prompt_tokens", "response_tokens")):
            raise ValueError("Replay trajectory identities or lengths differ")
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


def compare(left, right):
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
    return {
        "left": str(left),
        "right": str(right),
        "reference_sha256": summaries[0]["reference_sha256"],
        **paired_metrics(*records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(compare(args.left, args.right), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
