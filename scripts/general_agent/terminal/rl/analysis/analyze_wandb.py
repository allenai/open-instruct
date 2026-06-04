#!/usr/bin/env python3
"""Summarize and sanity-check a GRPO / Terminal-RL run from its wandb logs.

This encodes the metric-reading heuristics documented in
``docs/algorithms/monitoring_and_debugging_runs.md``: it pulls the keys this
repo logs, prints three grouped tables (learning signal / stability vital-signs
/ efficiency & infra), and emits automated FLAGS that call out the common
failure modes (dead reward key, degenerate advantages, KL runaway, high
truncation, protocol failures, staleness/preemption, tail regression, etc.).

It is intentionally dependency-light (wandb + numpy only) so it runs on a laptop
without importing the training stack.

Usage:
    uv run python analyze_wandb.py https://wandb.ai/ai2-llm/oe-general-agents/runs/9ou3i1in
    uv run python analyze_wandb.py ai2-llm/oe-general-agents/9ou3i1in

Returns the parsed wandb run config (dict) from ``run()`` so an orchestrator can
reuse ``exp_name`` / ``rollouts_save_path`` / ``response_length`` to drive the
trajectory analysis.
"""

import argparse
import re

import numpy as np
import wandb

# ---------------------------------------------------------------------------
# Metric groups. These are the exact keys open_instruct/grpo_fast.py logs.
# ---------------------------------------------------------------------------
LEARNING_KEYS = [
    "scores",
    "val/avg_group_performance_pre_filter",
    "val/avg_group_performance_post_filter",
    "objective/verifiable_reward",
    "objective/verifiable_correct_rate",
]
STABILITY_KEYS = [
    "objective/kl2_avg",  # preferred (stable k3) KL-to-reference estimator
    "objective/kl1_avg",
    "optim/grad_norm",
    "policy/clipfrac_avg",
    "val/advantages_min",
    "val/advantages_max",
    "val/ratio",
    "val/ratio_var",
    "loss/policy_avg",
    "lr",
]
BEHAVIOR_KEYS = [
    "val/stop_rate",
    "val/non_submitting_completion_fraction",
    "val/truncated_completion_fraction",
    "val/sequence_lengths",
    "val/sequence_lengths_solved",
    "val/sequence_lengths_unsolved",
    "tools/aggregate/avg_calls_per_rollout",
    "tools/aggregate/failure_rate",
    "tools/bash/failure_rate",
]
INFRA_KEYS = [
    "stale_results_dropped",
    "real_batch_size_ratio",
    "unsolved_batch_size_ratio",
    "model_step_mean",
    "training_step",
    "learner_mfu",
    "learner_tokens_per_second_step",
    "time/trainer_idle_waiting_for_inference",
    "time/total",
]
ALL_KEYS = LEARNING_KEYS + STABILITY_KEYS + BEHAVIOR_KEYS + INFRA_KEYS

CONFIG_KEYS = [
    "exp_name",
    "run_name",
    "response_length",
    "async_steps",
    "total_episodes",
    "num_unique_prompts_rollout",
    "num_samples_per_prompt_rollout",
    "beta",
    "learning_rate",
    "rollouts_save_path",
    "save_traces",
]


def parse_run_path(s: str) -> str:
    """Accept a full wandb URL or an ``entity/project/run_id`` string."""
    s = s.strip()
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    parts = [p for p in s.split("/") if p]
    if len(parts) == 3:
        return "/".join(parts)
    raise ValueError(f"Could not parse wandb run from {s!r}. Pass a wandb URL or 'entity/project/run_id'.")


def _stats(series: np.ndarray):
    """first, early-mean (first 10%), late-mean (last 10%), min, max, slope-over-run."""
    s = series[~np.isnan(series)]
    if s.size == 0:
        return None
    n = s.size
    k = max(1, n // 10)
    slope = float(np.polyfit(np.arange(n), s, 1)[0]) if n >= 2 else 0.0
    return {
        "first": float(s[0]),
        "early": float(s[:k].mean()),
        "late": float(s[-k:].mean()),
        "min": float(s.min()),
        "max": float(s.max()),
        "slope": slope,
        "slope_total": slope * n,
        "n": n,
    }


def _print_table(title: str, keys, history: dict):
    print(f"\n=== {title} ===")
    print(f"{'metric':<46}{'first':>10}{'early':>10}{'late':>10}{'min':>10}{'max':>10}{'Δ/run':>10}")
    for k in keys:
        st = _stats(history[k]) if k in history else None
        if st is None:
            continue
        print(
            f"{k:<46}{st['first']:>10.4g}{st['early']:>10.4g}{st['late']:>10.4g}"
            f"{st['min']:>10.4g}{st['max']:>10.4g}{st['slope_total']:>10.4g}"
        )


def _get(history, key):
    return _stats(history[key]) if key in history else None


def compute_flags(history: dict, config: dict) -> list:
    """Encode the diagnostic heuristics. Returns list of (level, message)."""
    flags = []
    resp_len = config.get("response_length")
    async_steps = config.get("async_steps")
    beta = config.get("beta")

    # --- which reward key is live? ---
    sc = _get(history, "scores")
    vr = _get(history, "objective/verifiable_reward")
    if sc and sc["late"] > 1e-6 and vr is not None and abs(vr["late"]) < 1e-9 and abs(vr["max"]) < 1e-9:
        flags.append(
            (
                "INFO",
                "Reward flows through the ENVIRONMENT path: `objective/verifiable_reward` is flat 0.0 "
                "while `scores` is non-zero. This is expected for sandbox/agent envs — read `scores` and "
                "`val/avg_group_performance_*`, ignore the objective/* reward keys.",
            )
        )

    # --- is it learning? prefer avg_group_performance, fall back to scores ---
    perf = _get(history, "val/avg_group_performance_pre_filter") or sc
    perf_name = (
        "val/avg_group_performance_pre_filter" if _get(history, "val/avg_group_performance_pre_filter") else "scores"
    )
    if perf:
        rel = perf["slope_total"] / (abs(perf["early"]) + 1e-9)
        if perf["slope_total"] > 0.02 and rel > 0.05:
            flags.append(
                (
                    "GOOD",
                    f"Learning: {perf_name} rising {perf['early']:.3f} -> {perf['late']:.3f} (Δ={perf['slope_total']:+.3f} over run).",
                )
            )
        elif abs(perf["slope_total"]) <= 0.02:
            flags.append(
                (
                    "WARN",
                    f"{perf_name} is ~flat ({perf['early']:.3f} -> {perf['late']:.3f}). Possibly not learning — check advantage spread, LR, reward signal.",
                )
            )
        else:
            flags.append(
                (
                    "WARN",
                    f"{perf_name} is DECLINING ({perf['early']:.3f} -> {perf['late']:.3f}, Δ={perf['slope_total']:+.3f}).",
                )
            )

    # --- advantage signal degenerate? ---
    amin, amax = _get(history, "val/advantages_min"), _get(history, "val/advantages_max")
    if amin and amax and max(abs(amax["late"]), abs(amin["late"])) < 1e-4:
        flags.append(
            (
                "BAD",
                "Advantages collapsed to ~0 (min~=max~=0): NO gradient signal. Every group is all-solved or all-failed, or filtering removed all signal.",
            )
        )

    # --- KL ---
    kl = _get(history, "objective/kl2_avg") or _get(history, "objective/kl1_avg")
    if kl:
        if beta in (0, 0.0):
            note = " (beta=0 -> KL is monitor-only; nothing in the loss reins it in, watch it yourself)"
        else:
            note = ""
        if kl["max"] > 0 and kl["late"] > 5 * (kl["early"] + 1e-9) and kl["late"] > 0.5:
            flags.append(
                (
                    "WARN",
                    f"KL climbing fast (late {kl['late']:.3f} vs early {kl['early']:.3f}){note}. Watch for divergence; consider lower LR / raise beta / tighter clip.",
                )
            )
        else:
            flags.append(("INFO", f"KL-to-ref steady ({kl['early']:.3f} -> {kl['late']:.3f}){note}."))

    # --- clip fraction ---
    cf = _get(history, "policy/clipfrac_avg")
    if cf and cf["late"] > 0.2:
        flags.append(
            (
                "WARN",
                f"High clip fraction ({cf['late']:.2%}): updates want to exceed the trust region. LR too high or data too stale.",
            )
        )

    # --- grad norm spikes ---
    gn = _get(history, "optim/grad_norm")
    if gn and gn["max"] > 8 * (np.median([gn["early"], gn["late"]]) + 1e-9) and gn["max"] > 1.0:
        flags.append(
            (
                "WARN",
                f"Grad-norm spikes (max {gn['max']:.3g} vs typical ~{gn['late']:.3g}): instability risk. Inspect the step(s); consider stronger clipping / lower LR.",
            )
        )

    # --- truncation (budget) ---
    tr = _get(history, "val/truncated_completion_fraction")
    su = _get(history, "val/sequence_lengths_unsolved")
    if tr and tr["late"] > 0.15:
        msg = f"High truncation ({tr['late']:.1%} of rollouts): a big chunk never finished."
        if su and resp_len and su["late"] > 0.8 * resp_len:
            msg += f" Unsolved length ({su['late']:.0f}) is pressing the response_length cap ({resp_len}) — BUDGET-BOUND failures."
        flags.append(("WARN", msg + " Levers: raise response_length, or encourage conciseness / curriculum."))

    # --- non-submitting ---
    ns = _get(history, "val/non_submitting_completion_fraction")
    if ns and ns["late"] > 0.2:
        flags.append(
            (
                "WARN",
                f"{ns['late']:.1%} of rollouts never submit. Check parser/format, system prompt clarity, and token budget.",
            )
        )

    # --- staleness / preemption ---
    sd = _get(history, "stale_results_dropped")
    if sd and sd["max"] > 5 and sd["max"] > 4 * (sd["late"] + 1e-9):
        flags.append(
            (
                "INFO",
                f"`stale_results_dropped` has bursts (max {sd['max']:.0f}, typical ~{sd['late']:.0f}) — usually preemption/restart events; benign unless constant.",
            )
        )
    ms = _get(history, "model_step_mean")
    ts = _get(history, "training_step")
    if ms and ts and async_steps:
        gap = ts["late"] - ms["late"]
        if gap > 2 * async_steps:
            flags.append(
                (
                    "WARN",
                    f"Staleness gap (training_step - model_step ~= {gap:.1f}) exceeds 2x async_steps ({async_steps}): generation falling behind -> off-policy data.",
                )
            )

    # --- tail regression detector (catches the "last few steps fell apart" case) ---
    if perf_name in history:
        s = history[perf_name]
        s = s[~np.isnan(s)]
        if s.size >= 8:
            tail = s[-3:].mean()
            body = s[-10:-3].mean() if s.size >= 10 else s[:-3].mean()
            if body > 0 and tail < 0.8 * body:
                flags.append(
                    (
                        "WARN",
                        f"TAIL REGRESSION: last 3 steps of {perf_name} ({tail:.3f}) dropped >20% below the preceding window ({body:.3f}). Could be a post-preemption stale batch or onset of instability — inspect recent trajectories.",
                    )
                )

    # --- LR display sanity ---
    lr = _get(history, "lr")
    if lr and lr["late"] > 0 and lr["late"] < 1e-4:
        flags.append(
            (
                "INFO",
                f"LR is small ({lr['late']:.2g}); if grad-norm is tiny+stable and learning is slow, there is headroom to raise it.",
            )
        )

    return flags


def estimate_eta(history: dict, config: dict):
    total_ep = config.get("total_episodes")
    n_prompts = config.get("num_unique_prompts_rollout")
    ts = _get(history, "training_step")
    tt = _get(history, "time/total")
    if not (total_ep and n_prompts and ts):
        return None
    planned = total_ep / n_prompts
    cur = ts["late"]
    step_s = tt["late"] if tt else None
    line = f"~{planned:.0f} planned steps; at step ~{cur:.0f} ({100 * cur / planned:.0f}%)."
    if step_s:
        remain_h = (planned - cur) * step_s / 3600
        line += f" ~{step_s:.0f}s/step -> ~{remain_h:.0f}h ({remain_h / 24:.1f}d) of wall-clock remaining."
    return line


def run(run_path: str, samples: int = 100000) -> dict:
    """Fetch, print report, return the run config dict."""
    path = parse_run_path(run_path)
    api = wandb.Api()
    run_obj = api.run(path)
    config = {k: run_obj.config.get(k) for k in CONFIG_KEYS}

    print("=" * 92)
    print(f"WANDB RUN: {path}")
    print(f"  name={run_obj.name}  state={run_obj.state}  created={run_obj.created_at}")
    print(f"  exp_name={config.get('exp_name')}")
    print(f"  run_name={config.get('run_name')}")
    print(
        f"  beta={config.get('beta')}  lr={config.get('learning_rate')}  response_length={config.get('response_length')}  "
        f"async_steps={config.get('async_steps')}"
    )
    print(f"  rollouts_save_path={config.get('rollouts_save_path')}  save_traces={config.get('save_traces')}")
    runtime = run_obj.summary.get("_runtime")
    if runtime:
        print(f"  runtime={runtime / 3600:.1f}h  last_step={run_obj.summary.get('_step')}")

    df = run_obj.history(keys=ALL_KEYS, samples=samples, pandas=True)
    history = {k: df[k].to_numpy(dtype=float) for k in ALL_KEYS if k in df.columns}

    _print_table("LEARNING SIGNAL", LEARNING_KEYS, history)
    _print_table("STABILITY / VITAL SIGNS", STABILITY_KEYS, history)
    _print_table("AGENT / BEHAVIOR", BEHAVIOR_KEYS, history)
    _print_table("EFFICIENCY / INFRA", INFRA_KEYS, history)

    eta = estimate_eta(history, config)
    if eta:
        print(f"\n=== WALL-CLOCK ===\n{eta}")

    print("\n" + "=" * 92)
    print("FLAGS  (heuristics from docs/algorithms/monitoring_and_debugging_runs.md)")
    print("=" * 92)
    order = {"BAD": 0, "WARN": 1, "GOOD": 2, "INFO": 3}
    flags = sorted(compute_flags(history, config), key=lambda f: order.get(f[0], 9))
    if not flags:
        print("  (no flags fired)")
    for level, msg in flags:
        print(f"  [{level:4}] {msg}")
    print()
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="wandb run URL or 'entity/project/run_id'")
    ap.add_argument("--samples", type=int, default=100000, help="max history points to fetch")
    args = ap.parse_args()
    run(args.run, samples=args.samples)


if __name__ == "__main__":
    main()
