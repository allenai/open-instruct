"""Compare frozen-token serving diagnostics with prompt-clustered intervals.

Serving-relative differences and optional fixed Core errors are not evidence of
learning quality. Compare repeated controls to observe repeat variability.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.miles import throughput_occupancy


def route_agreement(left_root, right_root, left, right, *, bootstrap=2000, seed=17):
    clusters = {}
    layer_changed = layer_total = None
    layouts = []
    for root in (left_root, right_root):
        path = Path(root) / "benchmark-model-config.json"
        config = json.loads(path.read_text()) if path.is_file() else {}
        layouts.append(
            (
                config["num_hidden_layers"],
                config.get("dense_layers_indices") or [],
                config.get("n_routed_experts"),
                config.get("num_experts_per_tok"),
            )
            if config.get("model_type") == "olmo3moe"
            else None
        )
    if layouts[0] != layouts[1]:
        raise ValueError("Routing comparisons require matching layer layouts")
    routed_layers = None
    recorded_layers = None
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
        if layouts[0] is not None and x.shape[1] != layouts[0][0]:
            raise ValueError("Routing matrix layer dimension differs from the model configuration")
        recorded_layers = x.shape[1]
        dense = layouts[0][1] if layouts[0] is not None else []
        if any(not isinstance(layer, int) or not 0 <= layer < recorded_layers for layer in dense):
            raise ValueError("Model configuration contains an invalid dense-layer index")
        routed_layers = [layer for layer in range(recorded_layers) if layer not in dense]
        if not routed_layers:
            raise ValueError("Routing matrices contain no routed layers")
        if layouts[0] is not None:
            experts, top_k = layouts[0][2:]
            if any(value is not None and (type(value) is not int or value < 1) for value in (experts, top_k)):
                raise ValueError("Model configuration contains invalid expert counts")
            if top_k is not None and x.shape[2] != top_k:
                raise ValueError("Routing matrix top-k dimension differs from the model configuration")
            # Agreement alone can accept two identically corrupt payloads. Check
            # the complete prompt/response payload before slicing metric positions.
            # Dense-layer padding is deliberately excluded from these checks.
            for matrix in (x, y):
                values = matrix[:, routed_layers]
                if (values < 0).any() or (experts is not None and (values >= experts).any()):
                    raise ValueError("Routing payload contains an out-of-range expert ID")
                ordered = np.sort(values, axis=-1)
                if (ordered[..., 1:] == ordered[..., :-1]).any():
                    raise ValueError("Routing payload selects a duplicate expert for one token/layer")
        # The route for the last prompt token conditions the first response logit.
        x = x[a["prompt_tokens"] - 1 :, routed_layers]
        y = y[b["prompt_tokens"] - 1 :, routed_layers]
        changed = x != y
        absent_slots = changed_sets = 0
        # Chunk the top-k cross-comparison to bound memory for long trajectories.
        # Permuting the same selected experts changes ordered slots but not membership.
        for offset in range(0, len(x), 1024):
            absent = ~(x[offset : offset + 1024, ..., None] == y[offset : offset + 1024, ..., None, :]).any(axis=-1)
            absent_slots += absent.sum()
            changed_sets += absent.any(axis=(1, 2)).sum()
        row = np.array(
            [changed.size, changed.sum(), len(changed), changed.any(axis=(1, 2)).sum(), absent_slots, changed_sets],
            dtype=np.float64,
        )
        clusters[a["group_index"]] = clusters.get(a["group_index"], np.zeros(6)) + row
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
        ("expert_membership_changed_fraction", 4, 0),
        ("conditioning_token_expert_set_changed_fraction", 5, 2),
    ):
        result[name] = {
            "value": float(total[numerator] / total[denominator]),
            "prompt_cluster_95pct_interval": np.quantile(
                draws[:, numerator] / draws[:, denominator], [0.025, 0.975]
            ).tolist(),
        }
    assert layer_changed is not None and layer_total is not None and routed_layers is not None
    assert recorded_layers is not None
    per_layer: list[float | None] = [None] * recorded_layers
    for layer, rate in zip(routed_layers, layer_changed / layer_total, strict=True):
        per_layer[layer] = float(rate)
    result["per_layer_expert_slot_changed_fraction"] = per_layer
    result["routed_layer_indices"] = routed_layers
    result["layer_layout_known"] = layouts[0] is not None
    result["expert_id_contract_checked"] = layouts[0] is not None and all(
        value is not None for value in layouts[0][2:]
    )
    result["scope"] = (
        "Response-conditioning routes; known dense layers are excluded, otherwise all recorded layers are used. "
        "Membership counts selected experts absent from the other arm, independently of top-k order. "
        "Prompt clusters retain copies and repeated waves."
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


def response_position_slices(*arms):
    """Half-open response offsets; long-position rows include only reaching prompts."""
    for start, stop in ((0, 1024), (1024, 8192), (8192, 16384), (16384, 32768), (32768, None)):
        sliced = []
        for records in arms:
            sliced.append(
                [
                    {
                        **row,
                        "logprobs": row["logprobs"][start:stop],
                        "response_tokens": len(row["logprobs"][start:stop]),
                        "route_sha256": None,
                        "route_path": None,
                    }
                    for row in records
                    if row["response_tokens"] > start
                ]
            )
        if sliced[0]:
            yield start, stop, sliced


def core_error_change(reference, left, right, *, bootstrap=2000, seed=17):
    clusters = {}
    for row, a, b in zip(reference, left, right, strict=True):
        core = np.asarray(row["logprobs"], dtype=np.float64)
        before, after = np.asarray(a["logprobs"]) - core, np.asarray(b["logprobs"]) - core
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
    return {
        name: {
            "value": float(total[i] / total[0]),
            "prompt_cluster_95pct_interval": np.quantile(draws[:, i] / draws[:, 0], [0.025, 0.975]).tolist(),
        }
        for i, name in enumerate(("mean_absolute_error_change", "ratio_outlier_fraction_change"), 1)
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
    if summary.get("routing", "natural") != "natural":
        raise ValueError(
            "Comparing serving arms requires a natural-routing Core reference, not one arm's replay routes"
        )
    rows = [json.loads(line) for line in (root / "benchmark-core-reference.jsonl").read_text().splitlines()]
    scores = {row["index"]: row for row in rows}
    if len(scores) != len(rows) or len(rows) != summary["trajectories"]:
        raise ValueError("Core reference has missing or duplicate trajectories")
    reference = []
    for a in left:
        row = scores[a.get("source_index", a["index"])]
        if any(row[key] != a[key] for key in ("group_index", "prompt_tokens", "response_tokens")):
            raise ValueError("Core reference trajectory identity differs")
        reference.append({**row, **{key: a[key] for key in ("index", "source_index", "copy_index") if key in a}})
    return {
        "reference": str(root),
        "contract": summary,
        "left_error": paired_metrics(reference, left, bootstrap=bootstrap, seed=seed),
        "right_error": paired_metrics(reference, right, bootstrap=bootstrap, seed=seed),
        "paired_change": core_error_change(reference, left, right, bootstrap=bootstrap, seed=seed),
        "by_response_position": [
            {
                "start": start,
                "stop": stop,
                "left_error": paired_metrics(ref, a, bootstrap=bootstrap, seed=seed),
                "right_error": paired_metrics(ref, b, bootstrap=bootstrap, seed=seed),
                "paired_change": core_error_change(ref, a, b, bootstrap=bootstrap, seed=seed),
            }
            for start, stop, (ref, a, b) in response_position_slices(reference, left, right)
        ],
    }


def engine_attribution(summary, rows):
    urls = summary.get("worker_urls")
    if urls is None:
        return {"available": False, "reason": "Older replay did not retain serving-engine attribution"}
    if not urls or len(set(urls)) != len(urls):
        raise ValueError("Replay worker URLs must be nonempty and unique")
    if any(row.get("worker_url") not in urls for row in rows):
        raise ValueError("Replay has missing or unknown worker attribution")
    waves = []
    count = summary["trajectories"]
    for repeat in range(summary["repeats"]):
        wave = rows[repeat * count : (repeat + 1) * count]
        engines = {}
        for url in urls:
            subset = [row for row in wave if row["worker_url"] == url]
            engines[url] = {
                "trajectories": len(subset),
                "prompt_clusters": len({row["group_index"] for row in subset}),
                "response_tokens": sum(row["response_tokens"] for row in subset),
            }
        waves.append({"repeat": repeat, "engines": engines})
    return {
        "available": True,
        "waves": waves,
        "all_engines_sampled_each_wave": all(
            engine["trajectories"] > 0 for wave in waves for engine in wave["engines"].values()
        ),
    }


def compare_engine_subsets(roots, summaries, records, core):
    result = {}
    for side, (root, summary) in enumerate(zip(roots, summaries, strict=True)):
        engines = {}
        for url in summary.get("worker_urls", []):
            indices = [i for i, row in enumerate(records[side]) if row["worker_url"] == url]
            if not indices:
                continue
            arms = [[rows[i] for i in indices] for rows in records]
            engines[url] = {"paired_serving_difference": paired_metrics(*arms)}
            if core is not None:
                engines[url]["core_comparison"] = core_comparison(core, summary["reference_sha256"], *arms)
        result[str(root)] = engines
    return {
        "arms": result,
        "scope": "Matched comparisons conditioned on observed engine routing. Subsets differ across engines; "
        "intervals retain prompt clusters and are not adjusted for multiple comparisons.",
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
    if summaries[0].get("background", False) != summaries[1].get("background", False):
        raise ValueError("Replay reference mismatch: background load protocol")
    result = {
        "left": str(left),
        "right": str(right),
        "reference_sha256": summaries[0]["reference_sha256"],
        **paired_metrics(*records),
        "by_response_position": [
            {"start": start, "stop": stop, **paired_metrics(*arms)}
            for start, stop, arms in response_position_slices(*records)
        ],
        "route_agreement": route_agreement(left, right, *records),
        "engine_attribution": {
            str(root): engine_attribution(summary, rows)
            for root, summary, rows in zip((left, right), summaries, records, strict=True)
        },
        "diagnostic_occupancy": {
            str(root): [
                {
                    "repeat": wave["repeat"],
                    "engines": throughput_occupancy.analyze(
                        Path(root), window=(wave["started_unix"], wave["finished_unix"])
                    )["engines"],
                }
                for wave in summary.get("waves", [])
            ]
            for root, summary in zip((left, right), summaries, strict=True)
        },
    }
    if core is not None:
        result["core_comparison"] = core_comparison(core, summaries[0]["reference_sha256"], *records)
    result["by_serving_engine"] = compare_engine_subsets((left, right), summaries, records, core)
    result["per_wave"] = []
    count = summaries[0]["trajectories"]
    for repeat in range(summaries[0]["repeats"]):
        arms = [rows[repeat * count : (repeat + 1) * count] for rows in records]
        wave = {"repeat": repeat, **paired_metrics(*arms)}
        if core is not None:
            wave["core_comparison"] = core_comparison(core, summaries[0]["reference_sha256"], *arms)
        result["per_wave"].append(wave)
    result["strata_scope"] = (
        "Response positions are zero-based, half-open intervals conditional on reaching that position. "
        "Waves are matched by repeat index; intervals retain prompt clusters and are not adjusted for multiple comparisons."
    )
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
