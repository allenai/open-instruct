"""Analyze retained artifacts from the paired, bounded head comparison."""

import ast
import json
import math
import re
import statistics as stats
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
results = root / "comparison-results"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    return dict(mean=stats.mean(values), median=stats.median(values), min=min(values), max=max(values))


def arm_metrics(name):
    folder = results / name
    text = re.sub(r"\x1b\[[0-9;]*m", "", (folder / "run.log").read_text())
    metrics = {}
    for line in text.splitlines():
        match = re.search(r"Core optimizer step (\d+): (\{.*\})", line)
        if match:
            metrics[int(match[1])] = ast.literal_eval(match[2])
    assert sorted(metrics) == list(range(1, 33)), (name, sorted(metrics))
    checkpoints = folder / "run/checkpoints"
    contracts = []
    for rank in (0, 1):
        contract = rows(checkpoints / f"training_contract_rank{rank}.jsonl")
        contracts.extend(contract)
        assert [r["step"] for r in contract if r["event"] == "optimizer"] == list(range(1, 33))
        assert not any(r["event"] == "optimizer_rejected" for r in contract)
    publications = rows(checkpoints / "publication.jsonl")
    assert [r["version"] for r in publications if not r["repeated_version"]] == list(range(33))
    times = rows(checkpoints / "driver_timing.jsonl")
    flow = {r["rollout_id"]: r for r in rows(checkpoints / "rollout_flow.jsonl")}
    assert sorted(flow) == list(range(32))
    warm = list(range(9, 33))
    values = [metrics[i] for i in warm]
    stages = {
        stage: [r for r in times if r["stage"] == stage and r["rollout_id"] in range(8, 32)]
        for stage in ("generation_wait", "training", "publication")
    }
    assert all(len(v) == 24 and all(r["passed"] for r in v) for v in stages.values())
    start = min(r["started_unix"] for r in stages["generation_wait"])
    end = max(r["started_unix"] + r["seconds"] for r in stages["publication"])
    total_active = sum(v["train/active_response_tokens"] for v in values)
    response_tokens = sum(flow[i - 1]["response_tokens"] for i in warm)
    gradients = [v["train/grad_norm"] for v in metrics.values()]
    assert all(math.isfinite(g) and g >= 0 for g in gradients) and max(gradients) > 0
    refreshed = [r for r in contracts if r["event"] == "refresh_scores" and r["rollout_id"] in range(8, 32)]
    queue = [flow[i - 1]["queue_metrics"] for i in warm]
    ratio_groups = [r[k] for r in refreshed for k in ("historical_prefix", "latest_forward") if k in r]
    return {
        "updates": 32,
        "warm_mean_sample_staleness": stats.mean(q["rollout/fully_async/avg_staleness"] for q in queue),
        "warm_max_sample_staleness": max(q["rollout/fully_async/max_staleness"] for q in queue),
        "warm_stale_groups_filtered": sum(q["rollout/fully_async/stale_groups_filtered"] for q in queue),
        "warm_tis_clip_fraction_active": sum(r["tokens"] * r["tis_clip_fraction"] for r in ratio_groups)
        / sum(r["tokens"] for r in ratio_groups),
        "first_score_mean_abs": metrics[1]["train/collection_score_mean_abs"],
        "first_step_seconds": metrics[1]["train/step_seconds"],
        "first_training_call_seconds": next(
            r["seconds"] for r in times if r["stage"] == "training" and r["rollout_id"] == 0
        ),
        "zero_gradient_updates": [i for i, v in metrics.items() if v["train/grad_norm"] == 0],
        "zero_advantage_updates": [i for i, v in metrics.items() if v["advantages/max_abs"] == 0],
        "grad_norm": distribution(gradients),
        "grad_clip_updates": [i for i, v in metrics.items() if v["train/grad_norm"] > 1],
        "warm_window": [9, 32],
        "warm_cycle_seconds": (end - start) / 24,
        "warm_elapsed_seconds": end - start,
        "warm_response_tokens": response_tokens,
        "warm_response_tokens_per_second": response_tokens / (end - start),
        "warm_active_tokens_per_second": total_active / (end - start),
        "warm_model_tokens_per_trainer_second": sum(v["train/model_tokens"] for v in values)
        / sum(v["train/step_seconds"] for v in values),
        "warm_stages_seconds": {stage: distribution([r["seconds"] for r in data]) for stage, data in stages.items()},
        "warm_step_seconds": distribution([v["train/step_seconds"] for v in values]),
        "warm_score_mean_abs_token_weighted": sum(
            v["train/collection_score_mean_abs"] * v["train/collection_score_active_tokens"] for v in values
        )
        / sum(v["train/collection_score_active_tokens"] for v in values),
        "warm_peak_allocated_gib": max(v["train/packing/rank0_peak_allocated_bytes"] for v in values) / 2**30,
        "warm_current_version_token_fraction": sum(
            r["current_version_token_fraction"] * r["active_tokens"] for r in refreshed
        )
        / sum(r["active_tokens"] for r in refreshed),
        "per_step": [
            {
                key.removeprefix("train/"): value
                for key, value in metrics[i].items()
                if key
                in (
                    "train/completed_steps",
                    "train/step_seconds",
                    "train/grad_norm",
                    "train/collection_score_mean_abs",
                    "train/active_response_tokens",
                    "train/model_tokens",
                    "advantages/max_abs",
                )
            }
            for i in metrics
        ],
    }


def eval_summary(data):
    n = len(data)
    return dict(
        n=n,
        correct=sum(r["correct"] for r in data),
        correct_finished=sum(r["correct"] and r["finished"] for r in data),
        capped=sum(not r["finished"] for r in data),
        response_tokens=distribution([r["response_tokens"] for r in data]),
    )


def paired(a, b, key):
    aa = {r["id"]: r for r in a}
    bb = {r["id"]: r for r in b}
    assert aa.keys() == bb.keys()

    def score(row):
        return row["correct"] if key == "correct" else int(row["correct"] and row["finished"])

    diff = np.array([score(bb[k]) - score(aa[k]) for k in sorted(aa)])
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    discordant = wins + losses
    p = (
        min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
        if discordant
        else 1.0
    )
    rng = np.random.default_rng(17)
    boots = rng.choice(diff, size=(20000, len(diff)), replace=True).mean(axis=1)
    return dict(
        b_only=wins,
        a_only=losses,
        difference_pp=float(diff.mean() * 100),
        paired_bootstrap_95_pp=(np.quantile(boots, [0.025, 0.975]) * 100).tolist(),
        mcnemar_exact_p=p,
    )


def main():
    manifests = [
        json.loads((results / name / "run/prepared/data/manifest.json").read_text()) for name in ("bf16", "fp32")
    ]
    assert manifests[0]["outputs"] == manifests[1]["outputs"]
    assert manifests[0]["provenance"] == manifests[1]["provenance"]
    source = manifests[0]["provenance"]["sources"][0]
    assert len(source["train_rows"]) == 1024 and len(source["eval_rows"]) == 128
    assert set(source["train_rows"]).isdisjoint(source["eval_rows"])
    audit = json.loads((results / "eval-audit.json").read_text())

    report = {
        "data_sha256": manifests[0]["outputs"],
        "training": {name: arm_metrics(name) for name in ("bf16", "fp32")},
        "evaluation": {},
        "paired": {},
    }
    for name in ("bf16", "fp32"):
        report["evaluation"][name] = {step: eval_summary(data["rows"]) for step, data in audit[name].items()}
    for step in ("0", "32"):
        report["paired"][step] = {
            key: paired(audit["bf16"][step]["rows"], audit["fp32"][step]["rows"], key)
            for key in ("correct", "correct_finished")
        }
    for name in ("bf16", "fp32"):
        report["paired"][name + "_initial_to_final"] = {
            key: paired(audit[name]["0"]["rows"], audit[name]["32"]["rows"], key)
            for key in ("correct", "correct_finished")
        }
    (root / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "training"}, indent=2))
    for name, data in report["training"].items():
        print(name, json.dumps({k: v for k, v in data.items() if k != "per_step"}, indent=2))


if __name__ == "__main__":
    main()
