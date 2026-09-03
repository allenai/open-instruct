"""Summarize and compare thinking-trace lengths across models.

Reads the .jsonl files written by ``generate_traces.py`` and reports, per model,
the mean and variance of thinking-trace length in tokens, plus the things that
decide whether those two numbers can be trusted:

  * **Truncation rate.** A trace that hit ``--max-tokens`` is censored, so its
    true length is only a lower bound. The mean over all traces is therefore a
    lower bound too; we also report the mean over completed traces only.
  * **Variance decomposition.** With several samples per prompt, total variance
    splits into between-prompt (some questions just need more thought) and
    within-prompt (the same question answered at different lengths). A single
    pooled variance hides which one dominates.
  * **Clustered bootstrap CIs.** Samples from one prompt are correlated, so the
    bootstrap resamples *prompts*, not traces. Naive per-trace CIs would be too
    narrow by roughly the design effect.

Example:
    PYTHONPATH=. uv run python scripts/thinking_traces/analyze_traces.py \\
        --traces qwen3-8b=/results/qwen.jsonl \\
        --traces r1-distill-llama-8b=/results/deepseek.jsonl \\
        --json-output /results/summary.json
"""

import argparse
import collections
import json

import numpy as np

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# Mirrors generate_traces.KIND_CLOSED. Duplicated rather than imported so this
# module stays numpy-only: fetch_and_compare.sh runs it locally, without the
# generator's datasets/transformers/openai stack.
KIND_CLOSED = "closed"

BOOTSTRAP_ROUNDS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="labelled trace file; repeat once per model",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--min-per-source", type=int, default=10, help="min traces to break out a source")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_traces(spec: str) -> tuple[str, list[dict]]:
    label, _, path = spec.partition("=")
    if not path:
        raise ValueError(f"--traces expects LABEL=PATH, got {spec!r}")
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    errors = [r for r in records if "error" in r]
    if errors:
        logger.warning("%s: dropping %d/%d failed generations", label, len(errors), len(records))
    return label, [r for r in records if "error" not in r]


def decompose_variance(records: list[dict]) -> dict:
    """Split total variance into between-prompt and within-prompt components.

    One-way random-effects decomposition. Unbalanced groups use the standard
    effective group size so a few short prompts don't skew the split.
    """
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["prompt_index"]].append(record["thinking_tokens"])
    sizes = np.array([len(v) for v in groups.values()], dtype=float)
    usable = [np.array(v, dtype=float) for v in groups.values() if len(v) >= 1]
    n_groups = len(usable)
    if n_groups < 2 or sizes.sum() <= n_groups:
        return {"between_prompt_var": None, "within_prompt_var": None, "intraclass_correlation": None}

    grand_mean = np.concatenate(usable).mean()
    group_means = np.array([g.mean() for g in usable])
    ss_between = float((sizes * (group_means - grand_mean) ** 2).sum())
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in usable))
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (sizes.sum() - n_groups)
    # Effective group size: equals k exactly when the design is balanced.
    k_eff = (sizes.sum() - (sizes**2).sum() / sizes.sum()) / (n_groups - 1)
    between = max((ms_between - ms_within) / max(k_eff, 1e-9), 0.0)
    total = between + ms_within
    return {
        "between_prompt_var": between,
        "within_prompt_var": ms_within,
        "intraclass_correlation": (between / total) if total > 0 else None,
    }


def clustered_bootstrap(records: list[dict], rounds: int, seed: int) -> dict:
    """Bootstrap the mean and SD by resampling whole prompts, not individual traces."""
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["prompt_index"]].append(record["thinking_tokens"])
    keys = list(groups.keys())
    if len(keys) < 2:
        return {"mean_ci95": None, "sd_ci95": None}
    arrays = [np.array(groups[k], dtype=float) for k in keys]
    rng = np.random.default_rng(seed)
    means = np.empty(rounds)
    sds = np.empty(rounds)
    for i in range(rounds):
        picks = rng.integers(0, len(arrays), len(arrays))
        sample = np.concatenate([arrays[p] for p in picks])
        means[i] = sample.mean()
        sds[i] = sample.std(ddof=1)
    return {
        "mean_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "sd_ci95": [float(np.percentile(sds, 2.5)), float(np.percentile(sds, 97.5))],
    }


def summarize(label: str, records: list[dict], args: argparse.Namespace) -> dict:
    lengths = np.array([r["thinking_tokens"] for r in records], dtype=float)
    truncated = np.array([bool(r.get("truncated")) for r in records])
    no_block = np.array([r.get("kind") == "no_block" for r in records])
    # A completion can hit the token cap *after* closing its thinking block: the
    # trace length is then exact and only the final answer is cut off. That does
    # not censor this metric, but it is a signal the cap is close to binding.
    answer_cut = np.array([r.get("finish_reason") == "length" and r.get("kind") == KIND_CLOSED for r in records])
    completed = lengths[~truncated]

    summary = {
        "label": label,
        "model": records[0].get("model") if records else None,
        "n_traces": len(records),
        "n_prompts": len({r["prompt_index"] for r in records}),
        "n_truncated": int(truncated.sum()),
        "truncation_rate": float(truncated.mean()),
        "n_no_thinking_block": int(no_block.sum()),
        "n_answer_truncated_after_complete_trace": int(answer_cut.sum()),
        "mean": float(lengths.mean()),
        "variance": float(lengths.var(ddof=1)),
        "sd": float(lengths.std(ddof=1)),
        "cv": float(lengths.std(ddof=1) / lengths.mean()) if lengths.mean() > 0 else None,
        "min": float(lengths.min()),
        "max": float(lengths.max()),
        "quantiles": {f"p{q}": float(np.percentile(lengths, q)) for q in (5, 10, 25, 50, 75, 90, 95, 99)},
        "completed_only": {
            "n": int(len(completed)),
            "mean": float(completed.mean()) if len(completed) else None,
            "variance": float(completed.var(ddof=1)) if len(completed) > 1 else None,
            "sd": float(completed.std(ddof=1)) if len(completed) > 1 else None,
        },
        "mean_answer_tokens": float(np.mean([r["answer_tokens"] for r in records])),
    }
    summary.update(decompose_variance(records))
    summary.update(clustered_bootstrap(records, BOOTSTRAP_ROUNDS, args.seed))

    by_source = {}
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record.get("dataset_source") or "unknown"].append(record["thinking_tokens"])
    for source, values in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        if len(values) < args.min_per_source:
            continue
        arr = np.array(values, dtype=float)
        by_source[source] = {
            "n": len(values),
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if len(values) > 1 else None,
            "median": float(np.median(arr)),
        }
    summary["by_dataset_source"] = by_source
    return summary


def compare(label_a: str, recs_a: list[dict], label_b: str, recs_b: list[dict], seed: int) -> dict:
    """Bootstrap the paired difference in mean length over the shared prompt set."""
    by_prompt_a = collections.defaultdict(list)
    by_prompt_b = collections.defaultdict(list)
    for record in recs_a:
        by_prompt_a[record["prompt_sha"]].append(record["thinking_tokens"])
    for record in recs_b:
        by_prompt_b[record["prompt_sha"]].append(record["thinking_tokens"])
    shared = sorted(set(by_prompt_a) & set(by_prompt_b))

    result = {
        "model_a": label_a,
        "model_b": label_b,
        "n_shared_prompts": len(shared),
        "prompt_sets_identical": set(by_prompt_a) == set(by_prompt_b),
    }
    if len(shared) < 2:
        logger.warning("only %d shared prompts; skipping paired comparison", len(shared))
        return result

    means_a = np.array([np.mean(by_prompt_a[s]) for s in shared])
    means_b = np.array([np.mean(by_prompt_b[s]) for s in shared])
    diff = means_b - means_a

    rng = np.random.default_rng(seed)
    boot_diff = np.empty(BOOTSTRAP_ROUNDS)
    boot_ratio = np.empty(BOOTSTRAP_ROUNDS)
    for i in range(BOOTSTRAP_ROUNDS):
        picks = rng.integers(0, len(shared), len(shared))
        sa, sb = means_a[picks], means_b[picks]
        boot_diff[i] = sb.mean() - sa.mean()
        boot_ratio[i] = sb.var(ddof=1) / sa.var(ddof=1) if sa.var(ddof=1) > 0 else np.nan

    result.update(
        {
            "mean_a": float(means_a.mean()),
            "mean_b": float(means_b.mean()),
            "mean_difference_b_minus_a": float(diff.mean()),
            "mean_difference_ci95": [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))],
            "ratio_b_over_a": float(means_b.mean() / means_a.mean()) if means_a.mean() > 0 else None,
            "prompt_mean_variance_ratio_b_over_a": float(np.nanmean(boot_ratio)),
            "prompt_mean_variance_ratio_ci95": [
                float(np.nanpercentile(boot_ratio, 2.5)),
                float(np.nanpercentile(boot_ratio, 97.5)),
            ],
            "share_of_prompts_where_b_longer": float((diff > 0).mean()),
        }
    )
    return result


def render(summaries: list[dict], comparison: dict | None) -> str:
    lines = ["", "=" * 78, "THINKING TRACE LENGTH (tokens)", "=" * 78]
    for s in summaries:
        ci = s.get("mean_ci95")
        ci_text = f"  [95% CI {ci[0]:.0f}–{ci[1]:.0f}]" if ci else ""
        lines += [
            "",
            f"{s['label']}  ({s['n_traces']} traces over {s['n_prompts']} prompts)",
            "-" * 78,
            f"  mean                {s['mean']:10.1f}{ci_text}",
            f"  variance            {s['variance']:10.1f}",
            f"  std dev             {s['sd']:10.1f}"
            + (f"  [95% CI {s['sd_ci95'][0]:.0f}–{s['sd_ci95'][1]:.0f}]" if s.get("sd_ci95") else ""),
            f"  coeff of variation  {s['cv']:10.3f}" if s.get("cv") else "",
            f"  median / p90 / p99  {s['quantiles']['p50']:9.0f} /"
            f" {s['quantiles']['p90']:.0f} / {s['quantiles']['p99']:.0f}",
            f"  min / max           {s['min']:10.0f} / {s['max']:.0f}",
            f"  truncated at cap    {s['n_truncated']:10d}  ({s['truncation_rate'] * 100:.1f}%)",
            f"  no thinking block   {s['n_no_thinking_block']:10d}",
            f"  answer cut, trace ok{s['n_answer_truncated_after_complete_trace']:10d}"
            "  (trace length still exact; cap is close to binding)",
        ]
        if s.get("between_prompt_var") is not None:
            lines += [
                f"  between-prompt var  {s['between_prompt_var']:10.1f}  (SD {s['between_prompt_var'] ** 0.5:.0f})",
                f"  within-prompt var   {s['within_prompt_var']:10.1f}  (SD {s['within_prompt_var'] ** 0.5:.0f})",
                f"  ICC                 {s['intraclass_correlation']:10.3f}"
                "  (share of variance explained by which prompt)",
            ]
        if s["completed_only"]["mean"] is not None and s["n_truncated"]:
            lines.append(
                f"  completed only      mean {s['completed_only']['mean']:.1f},"
                f" sd {s['completed_only']['sd']:.1f} (n={s['completed_only']['n']})"
            )

    if comparison and "mean_difference_b_minus_a" in comparison:
        lo, hi = comparison["mean_difference_ci95"]
        vlo, vhi = comparison["prompt_mean_variance_ratio_ci95"]
        lines += [
            "",
            "=" * 78,
            f"COMPARISON  ({comparison['model_b']} minus {comparison['model_a']})",
            "=" * 78,
            f"  shared prompts               {comparison['n_shared_prompts']}"
            f"  (identical sets: {comparison['prompt_sets_identical']})",
            f"  mean difference              {comparison['mean_difference_b_minus_a']:+.1f} tokens"
            f"  [95% CI {lo:+.0f}, {hi:+.0f}]",
            f"  ratio of means               {comparison['ratio_b_over_a']:.2f}x",
            f"  variance ratio (prompt means){comparison['prompt_mean_variance_ratio_b_over_a']:9.2f}x"
            f"  [95% CI {vlo:.2f}, {vhi:.2f}]",
            f"  prompts where B is longer    {comparison['share_of_prompts_where_b_longer'] * 100:.0f}%",
            "",
            "  The CI excludes zero, so the difference is real."
            if lo > 0 or hi < 0
            else "  The CI spans zero, so the difference is not resolved at this sample size.",
        ]
    lines.append("")
    return "\n".join(line for line in lines if line != "")


def main() -> None:
    args = parse_args()
    loaded = [load_traces(spec) for spec in args.traces]
    summaries = [summarize(label, records, args) for label, records in loaded if records]

    comparison = None
    if len(loaded) == 2:
        (label_a, recs_a), (label_b, recs_b) = loaded
        comparison = compare(label_a, recs_a, label_b, recs_b, args.seed)

    report = render(summaries, comparison)
    print(report)

    if args.json_output:
        with open(args.json_output, "w") as handle:
            json.dump({"models": summaries, "comparison": comparison}, handle, indent=2)
        logger.info("wrote %s", args.json_output)


if __name__ == "__main__":
    main()
