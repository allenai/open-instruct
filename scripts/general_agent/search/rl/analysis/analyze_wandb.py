#!/usr/bin/env python3
"""Summarize and sanity-check a Deep-Research (DR-Tulu) RL run from its wandb logs.

DR-Tulu RL is GRPO with **search/browse tools** and an **evolving-rubric reward**
(a GPT-4.1 judge scores each answer against a per-query rubric set on a 0..
``max_possible_score`` scale). That makes it differ from the binary-reward
Terminal-RL setup in three ways this script is built around:

  1. **Reward is continuous, not pass/fail.** ``scores`` / ``objective/rubric_reward``
     / ``objective/verifiable_reward`` are the *same* number (the rubric verifier is
     routed onto the verifiable-reward path via ``--remap_verifier general_rubric=rubric``)
     on a 0..``max_possible_score`` scale (default 10).
     ``val/avg_group_performance_*`` is that score divided by ``max_possible_score``.
  2. **The reward target is non-stationary.** Rubrics *evolve*: every step the judge
     can mint new criteria (``evolving_rubrics/*``), so the bar the model is graded
     against rises over training. Raw ``scores`` can fall even as the model improves,
     because it is chasing a moving target — read it together with the rubric counts.
  3. **``*_correct_rate`` = fraction of rollouts with reward > 0** (any rubric credit),
     NOT fraction at max. It is the cleanest "is total failure shrinking?" signal.

It is intentionally dependency-light (wandb + numpy only) so it runs on a laptop
without importing the training stack.

Usage:
    uv run python analyze_wandb.py https://wandb.ai/ai2-llm/oe-general-agents/runs/m8xd8yr4
    uv run python analyze_wandb.py ai2-llm/oe-general-agents/m8xd8yr4

Returns the parsed wandb run config (dict) from ``run()`` so an orchestrator can
reuse ``exp_name`` / ``rollouts_save_path`` / ``response_length`` / ``max_possible_score``
to drive the trajectory analysis.
"""

import argparse
import re

import numpy as np
import wandb

# ---------------------------------------------------------------------------
# Metric groups. These are the exact keys open_instruct/grpo_fast.py logs for a
# DR-Tulu (evolving-rubric + search-tool) GRPO run.
# ---------------------------------------------------------------------------
LEARNING_KEYS = [
    "scores",  # == rubric_reward == verifiable_reward (continuous, 0..max_possible_score)
    "objective/rubric_reward",
    "objective/verifiable_reward",
    "objective/rubric_correct_rate",  # fraction of rollouts with reward > 0
    "objective/verifiable_correct_rate",
    "val/avg_group_performance_pre_filter",  # == scores / max_possible_score
    "val/avg_group_performance_post_filter",
]
# Evolving-rubric machinery: the moving target the reward is measured against.
RUBRIC_KEYS = [
    "evolving_rubrics/num_active_rubrics",  # avg active rubrics per query this step
    "evolving_rubrics/avg_gt_rubrics",  # avg ground-truth (persistent) rubrics per query
    "evolving_rubrics/num_new_rubrics",  # avg new rubrics minted this step
    "evolving_rubrics/valid_rate",  # fraction of judge rubric-generations that parsed OK
    "evolving_rubrics/num_overrides",  # ground-truth overrides pushed to vLLM
    "evolving_rubrics/skipped",  # queries whose rubric update was skipped
    "evolving_rubrics/total_active",  # buffer-wide active rubric count (grows)
    "evolving_rubrics/total_persistent",  # buffer-wide persistent (seed) rubric count
    "evolving_rubrics/total_rubrics_in_use",  # active + persistent
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
# Agent behavior: how long, how many tool calls, and which tools.
BEHAVIOR_KEYS = [
    "val/stop_rate",
    "val/truncated_completion_fraction",
    "val/truncated_completion_correct_count",  # truncated rollouts that still scored > 0
    "val/non_submitting_completion_fraction",  # CONSTANT 1.0 for this env (see flags) — ignore
    "val/sequence_lengths",
    "val/sequence_lengths_solved",  # length of rollouts scoring max
    "val/sequence_lengths_unsolved",
    "tools/aggregate/avg_calls_per_rollout",
    "tools/google_search/avg_calls_per_rollout",
    "tools/browse_webpage/avg_calls_per_rollout",
    "tools/snippet_search/avg_calls_per_rollout",
    "tools/aggregate/failure_rate",
    "tools/google_search/failure_rate",
    "tools/browse_webpage/failure_rate",
    "tools/snippet_search/failure_rate",
    "tools/aggregate/avg_runtime",
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
ALL_KEYS = LEARNING_KEYS + RUBRIC_KEYS + STABILITY_KEYS + BEHAVIOR_KEYS + INFRA_KEYS

CONFIG_KEYS = [
    "exp_name",
    "run_name",
    "response_length",
    "max_possible_score",
    "async_steps",
    "total_episodes",
    "num_training_steps",
    "num_unique_prompts_rollout",
    "num_samples_per_prompt_rollout",
    "max_active_rubrics",
    "beta",
    "learning_rate",
    "loss_fn",
    "rollouts_save_path",
    "save_traces",
    "tools",
    "tool_call_names",
    "apply_evolving_rubric_reward",
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
    rows = [(k, _stats(history[k])) for k in keys if k in history]
    rows = [(k, st) for k, st in rows if st is not None]
    if not rows:
        return
    print(f"\n=== {title} ===")
    print(f"{'metric':<46}{'first':>10}{'early':>10}{'late':>10}{'min':>10}{'max':>10}{'Δ/run':>10}")
    for k, st in rows:
        print(
            f"{k:<46}{st['first']:>10.4g}{st['early']:>10.4g}{st['late']:>10.4g}"
            f"{st['min']:>10.4g}{st['max']:>10.4g}{st['slope_total']:>10.4g}"
        )


def _get(history, key):
    return _stats(history[key]) if key in history else None


def compute_flags(history: dict, config: dict) -> list:
    """Encode the DR-Tulu diagnostic heuristics. Returns list of (level, message)."""
    flags = []
    resp_len = config.get("response_length")
    max_score = config.get("max_possible_score") or 10
    async_steps = config.get("async_steps")
    beta = config.get("beta")

    # --- reward-key identity note (the DR analogue of the Terminal "reward-key trap") ---
    sc = _get(history, "scores")
    rr = _get(history, "objective/rubric_reward")
    vr = _get(history, "objective/verifiable_reward")
    if sc and rr and vr and abs(rr["late"] - sc["late"]) < 1e-6 and abs(vr["late"] - sc["late"]) < 1e-6:
        flags.append(
            (
                "INFO",
                f"`scores` == `objective/rubric_reward` == `objective/verifiable_reward` (all ~{sc['late']:.2f}/"
                f"{max_score:g}): the evolving-rubric judge IS the verifier (--remap_verifier general_rubric=rubric), "
                "so these three are one signal, not three. `val/avg_group_performance_*` is that score / max_possible_score.",
            )
        )

    # --- is it learning? Two complementary signals against a MOVING bar. ---
    corr = _get(history, "objective/rubric_correct_rate") or _get(history, "objective/verifiable_correct_rate")
    perf = _get(history, "val/avg_group_performance_pre_filter")
    # rubric count growth = the bar rising
    ta = _get(history, "evolving_rubrics/total_active")
    na = _get(history, "evolving_rubrics/num_active_rubrics")
    bar_rising = (ta and ta["slope_total"] > 0 and ta["late"] > 1.5 * (ta["early"] + 1e-9)) or (
        na and na["slope_total"] > 0.3
    )

    if corr:
        if corr["slope_total"] > 0.02:
            flags.append(
                (
                    "GOOD",
                    f"Learning (failure shrinking): `*_correct_rate` (fraction of rollouts scoring > 0) rising "
                    f"{corr['early']:.3f} -> {corr['late']:.3f}. Fewer rollouts earn zero rubric credit over training.",
                )
            )
        elif corr["slope_total"] < -0.02:
            flags.append(
                (
                    "WARN",
                    f"`*_correct_rate` DECLINING ({corr['early']:.3f} -> {corr['late']:.3f}): more rollouts are "
                    "scoring zero. Could be a harder rubric bar (see rubric growth) or genuine regression.",
                )
            )

    if perf:
        if perf["slope_total"] < -0.02 and bar_rising:
            flags.append(
                (
                    "INFO",
                    f"`avg_group_performance` (= scores/{max_score:g}) is FALLING ({perf['early']:.3f} -> "
                    f"{perf['late']:.3f}) WHILE the rubric bar is rising (active rubrics growing). A falling raw "
                    "score against a stricter rubric is NOT necessarily regression — cross-check `*_correct_rate` "
                    "and the trajectory mean reward. The reward target is non-stationary by design.",
                )
            )
        elif perf["slope_total"] > 0.02:
            flags.append(
                (
                    "GOOD",
                    f"`avg_group_performance` rising {perf['early']:.3f} -> {perf['late']:.3f} — mean rubric score "
                    "improving even as rubrics evolve.",
                )
            )
        elif abs(perf["slope_total"]) <= 0.02 and not (corr and abs(corr["slope_total"]) > 0.02):
            flags.append(
                (
                    "WARN",
                    f"`avg_group_performance` ~flat ({perf['early']:.3f} -> {perf['late']:.3f}) and correct-rate flat. "
                    "Possibly not learning — check advantage spread, LR, and rubric valid_rate.",
                )
            )

    # --- evolving-rubric health ---
    vrate = _get(history, "evolving_rubrics/valid_rate")
    if vrate and vrate["late"] < 0.9:
        flags.append(
            (
                "WARN",
                f"Rubric valid_rate is {vrate['late']:.2f} (<0.9): the judge/generator (GPT-4.1) is failing to "
                "produce parseable rubrics for a chunk of queries -> noisy/under-specified reward. Check the "
                "RUBRIC_GENERATION_MODEL / API errors.",
            )
        )
    skipped = _get(history, "evolving_rubrics/skipped")
    if skipped and skipped["late"] > 0.5:
        flags.append(
            (
                "WARN",
                f"evolving_rubrics/skipped averaging {skipped['late']:.1f} per step: rubric updates are being "
                "skipped for some queries (often judge errors or empty responses). Reward for those queries is stale.",
            )
        )
    if na:
        cap = config.get("max_active_rubrics")
        note = f" (cap = max_active_rubrics={cap})" if cap else ""
        flags.append(
            (
                "INFO",
                f"Active rubrics/query: {na['early']:.1f} -> {na['late']:.1f}{note}; buffer total_active "
                f"{(ta['early'] if ta else float('nan')):.0f} -> {(ta['late'] if ta else float('nan')):.0f}. "
                "This is the bar the reward is measured against — rising = stricter grading over time.",
            )
        )

    # --- advantage signal degenerate? (continuous reward => should be a wide spread) ---
    amin, amax = _get(history, "val/advantages_min"), _get(history, "val/advantages_max")
    if amin and amax and max(abs(amax["late"]), abs(amin["late"])) < 1e-3:
        flags.append(
            (
                "BAD",
                "Advantages collapsed to ~0: NO gradient signal. Every group's rollouts got near-identical rubric "
                "scores (zero within-group spread), or filtering removed all signal.",
            )
        )

    # --- KL ---
    kl = _get(history, "objective/kl2_avg") or _get(history, "objective/kl1_avg")
    if kl:
        note = " (beta=0 -> KL is monitor-only)" if beta in (0, 0.0) else f" (beta={beta} -> KL is in the loss)"
        if kl["max"] > 0 and kl["late"] > 5 * (kl["early"] + 1e-9) and kl["late"] > 0.5:
            flags.append(
                (
                    "WARN",
                    f"KL climbing fast (late {kl['late']:.3f} vs early {kl['early']:.3f}){note}. Watch for divergence; "
                    "consider lower LR / raise beta / tighter clip.",
                )
            )
        else:
            flags.append(("INFO", f"KL-to-ref steady ({kl['early']:.4f} -> {kl['late']:.4f}){note}."))

    # --- clip fraction (DAPO loss uses asymmetric clip_lower/clip_higher) ---
    cf = _get(history, "policy/clipfrac_avg")
    if cf and cf["late"] > 0.2:
        flags.append(
            (
                "WARN",
                f"High clip fraction ({cf['late']:.2%}): updates want to exceed the trust region. LR too high or "
                "data too stale.",
            )
        )

    # --- grad norm spikes ---
    gn = _get(history, "optim/grad_norm")
    if gn and gn["max"] > 8 * (np.median([gn["early"], gn["late"]]) + 1e-9) and gn["max"] > 1.0:
        flags.append(
            (
                "WARN",
                f"Grad-norm spikes (max {gn['max']:.3g} vs typical ~{gn['late']:.3g}): instability risk. Inspect the "
                "step(s); consider stronger clipping / lower LR.",
            )
        )

    # --- truncation / budget. DR is heavily budget-bound, but truncation != failure here. ---
    tr = _get(history, "val/truncated_completion_fraction")
    su = _get(history, "val/sequence_lengths_unsolved")
    if tr and tr["late"] > 0.4:
        msg = (
            f"VERY high truncation ({tr['late']:.0%} of rollouts hit the length budget). For DR this is the norm: "
            "long multi-turn search/browse transcripts press the response_length cap."
        )
        if su and resp_len and su["late"] > 0.8 * resp_len:
            msg += f" Unsolved length ({su['late']:.0f}) is pressing the response_length cap ({resp_len})."
        msg += (
            " Unlike binary envs, truncated rollouts still earn partial rubric reward (cross-check "
            "`val/truncated_completion_correct_count` and the trajectory truncated-vs-completed mean reward), "
            "so raising response_length buys answer quality, not just a pass/fail flip."
        )
        flags.append(("WARN", msg))

    # --- non_submitting is a constant artifact for this env ---
    ns = _get(history, "val/non_submitting_completion_fraction")
    if ns and ns["min"] > 0.99:
        flags.append(
            (
                "INFO",
                "`val/non_submitting_completion_fraction` is pinned at 1.0 — for this search env it is defined as "
                "`not rollout_state['done']`, and the env never sets done=True, so EVERY rollout counts as "
                "non-submitting. It is a constant artifact here, not a protocol failure; ignore it (and note "
                "mask_non_submitting_completions=False, so it does not affect training).",
            )
        )

    # --- tool failures ---
    tf = _get(history, "tools/aggregate/failure_rate")
    if tf and tf["late"] > 0.15:
        flags.append(
            (
                "WARN",
                f"Tool failure rate {tf['late']:.1%} (search/browse): the Serper/Jina/S2 backends are erroring or "
                "timing out on a meaningful fraction of calls -> noisy observations and wasted budget. Check the "
                "per-tool failure_rate rows and the API keys/quotas.",
            )
        )

    # --- staleness / preemption ---
    sd = _get(history, "stale_results_dropped")
    if sd and sd["max"] > 5 and sd["max"] > 4 * (sd["late"] + 1e-9):
        flags.append(
            (
                "INFO",
                f"`stale_results_dropped` has bursts (max {sd['max']:.0f}, typical ~{sd['late']:.0f}) — usually "
                "preemption/restart events; benign unless constant.",
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
                    f"Staleness gap (training_step - model_step ~= {gap:.1f}) exceeds 2x async_steps ({async_steps}): "
                    "generation falling behind -> off-policy data.",
                )
            )

    # --- tail regression on the cleanest learning signal (correct_rate) ---
    tail_key = (
        "objective/rubric_correct_rate"
        if "objective/rubric_correct_rate" in history
        else "val/avg_group_performance_pre_filter"
    )
    if tail_key in history:
        s = history[tail_key]
        s = s[~np.isnan(s)]
        if s.size >= 8:
            tail = s[-3:].mean()
            body = s[-10:-3].mean() if s.size >= 10 else s[:-3].mean()
            if body > 0 and tail < 0.8 * body:
                flags.append(
                    (
                        "WARN",
                        f"TAIL REGRESSION: last 3 steps of {tail_key} ({tail:.3f}) dropped >20% below the preceding "
                        f"window ({body:.3f}). Could be a post-preemption stale batch or onset of instability — "
                        "inspect recent trajectories.",
                    )
                )

    return flags


def estimate_eta(history: dict, config: dict):
    total_ep = config.get("total_episodes")
    n_prompts = config.get("num_unique_prompts_rollout")
    planned_steps = config.get("num_training_steps")
    ts = _get(history, "training_step")
    tt = _get(history, "time/total")
    if not ts:
        return None
    if not planned_steps and total_ep and n_prompts:
        planned_steps = total_ep / n_prompts
    if not planned_steps:
        return None
    cur = ts["late"]
    step_s = tt["late"] if tt else None
    line = f"~{planned_steps:.0f} planned steps; at step ~{cur:.0f} ({100 * cur / planned_steps:.0f}%)."
    if step_s:
        remain_h = (planned_steps - cur) * step_s / 3600
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
        f"  beta={config.get('beta')}  lr={config.get('learning_rate')}  loss_fn={config.get('loss_fn')}  "
        f"response_length={config.get('response_length')}  max_possible_score={config.get('max_possible_score')}  "
        f"async_steps={config.get('async_steps')}"
    )
    print(
        f"  max_active_rubrics={config.get('max_active_rubrics')}  "
        f"evolving_rubric_reward={config.get('apply_evolving_rubric_reward')}  "
        f"num_training_steps={config.get('num_training_steps')}"
    )
    print(f"  rollouts_save_path={config.get('rollouts_save_path')}  save_traces={config.get('save_traces')}")
    runtime = run_obj.summary.get("_runtime")
    if runtime:
        print(f"  runtime={runtime / 3600:.1f}h  last_step={run_obj.summary.get('_step')}")

    df = run_obj.history(keys=ALL_KEYS, samples=samples, pandas=True)
    history = {k: df[k].to_numpy(dtype=float) for k in ALL_KEYS if k in df.columns}

    _print_table(
        "LEARNING SIGNAL (reward 0..max_possible_score; correct_rate = frac reward>0)", LEARNING_KEYS, history
    )
    _print_table("EVOLVING RUBRICS (the moving reward target)", RUBRIC_KEYS, history)
    _print_table("STABILITY / VITAL SIGNS", STABILITY_KEYS, history)
    _print_table("AGENT / TOOLS / BEHAVIOR", BEHAVIOR_KEYS, history)
    _print_table("EFFICIENCY / INFRA", INFRA_KEYS, history)

    eta = estimate_eta(history, config)
    if eta:
        print(f"\n=== WALL-CLOCK ===\n{eta}")

    print("\n" + "=" * 92)
    print("FLAGS  (DR-Tulu heuristics; see docs/algorithms/dr_research_rl_trajectory_analysis.md)")
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
