#!/usr/bin/env python3
"""Classify Terminal-RL rollout failures from saved traces: truncation vs genuinely-wrong.

Reads the JSONL trace shards written by ``--save_traces`` (RolloutRecord, see
``open_instruct/rl_utils.py``) and answers the question "are my failures the
model running out of token budget, or is it finishing and being wrong?".

It uses the SAME truncation definition as the training code
(``open_instruct/data_loader.py``):
    truncated  <=>  finish_reason != "stop"  OR  len(response_tokens) >= response_length

For every training step it computes solved / zero-reward, and within the
zero-reward failures the split between truncated (budget) and stopped (wrong).
It then reports the TREND OVER TRAINING (is the truncation share of failures
rising or falling as the model learns?), aggregating across all run-instances
(restarts produce new timestamped files that continue the same step counter).

Optionally decodes the tails of a few example trajectories per bucket so you can
eyeball that the buckets mean what they claim (requires `transformers`).

Files are multi-GB; everything is streamed line-by-line. A cheap step-prefix
parse lets ``--per-step-cap`` skip the expensive json.loads on capped steps, so
a quick low-cap pass is fast.

Usage:
    uv run python analyze_trajectories.py --exp-name swerl_qwen35_4b_base_tmax_15k_verified_grpo_no_prev_reasoning
    uv run python analyze_trajectories.py --exp-name <name> --per-step-cap 64        # fast sample
    uv run python analyze_trajectories.py --exp-name <name> --decode-examples 1 --step 150
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

DEFAULT_ROLLOUTS_DIR = "/weka/oe-adapt-default/allennlp/deletable_rollouts/"
DEFAULT_RESPONSE_LENGTH = 32768
STEP_PREFIX_RE = re.compile(r'"step":\s*(\d+)')
INSTANCE_RE = re.compile(r"__(\d+)_rollouts_\d+\.jsonl$")


def find_shards(rollouts_dir: str, exp_name: str):
    """All rollout data shards for an exp, grouped by run-instance timestamp, in time order."""
    pattern = os.path.join(rollouts_dir, f"{exp_name}*_rollouts_*.jsonl")
    files = [f for f in glob.glob(pattern) if "trainer_logprobs" not in f and "_metadata" not in f]
    by_instance = defaultdict(list)
    for f in files:
        m = INSTANCE_RE.search(f)
        inst = m.group(1) if m else "unknown"
        by_instance[inst].append(f)
    for inst in by_instance:
        by_instance[inst].sort()  # shard order
    return dict(sorted(by_instance.items()))


def _blank():
    return dict(
        n=0,
        solved=0,
        zero=0,
        partial=0,
        trunc=0,
        trunc_zero=0,
        trunc_solved=0,
        trunc_partial=0,
        stop_zero=0,
        stop_partial=0,
        rlen_sum=0,
        rlen_unsolved_sum=0,
        n_unsolved=0,
        calls_sum=0,
    )


def classify_shards(shards_by_instance, response_length: int, per_step_cap: int = 0):
    """Stream all shards; return per-step aggregated stats, instance ranges, and per-group outcomes.

    ``group_outcomes[step][prompt_idx] = [n_solved_in_group, n_total_in_group]`` lets us measure
    per-prompt consistency (all-solved / all-failed / mixed), which reconciles a flat per-sample
    solve rate with a rising ``val/avg_group_performance`` (the model sharpening, not necessarily
    raising its overall success rate).
    """
    steps = defaultdict(_blank)
    group_outcomes = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    instance_ranges = {}
    overlap_steps = set()
    seen_step_instance = {}

    for inst, files in shards_by_instance.items():
        inst_min, inst_max = None, None
        for fp in files:
            with open(fp) as f:
                for line in f:
                    if not line:
                        continue
                    m = STEP_PREFIX_RE.search(line[:64])
                    if not m:
                        continue
                    st = int(m.group(1))
                    inst_min = st if inst_min is None else min(inst_min, st)
                    inst_max = st if inst_max is None else max(inst_max, st)
                    # overlap bookkeeping (a step produced by >1 instance = restart redo)
                    if st in seen_step_instance and seen_step_instance[st] != inst:
                        overlap_steps.add(st)
                    seen_step_instance[st] = inst
                    if per_step_cap and steps[st]["n"] >= per_step_cap:
                        continue

                    r = json.loads(line)
                    rew = r["reward"]
                    fr = r["finish_reason"]
                    rlen = len(r["response_tokens"])
                    calls = r.get("request_info", {}).get("num_calls", 0) or 0
                    g = group_outcomes[st][r.get("prompt_idx", -1)]
                    g[1] += 1
                    if rew >= 1.0 - 1e-8:
                        g[0] += 1

                    d = steps[st]
                    d["n"] += 1
                    d["rlen_sum"] += rlen
                    d["calls_sum"] += calls
                    trunc = (fr != "stop") or (rlen >= response_length)
                    solved = rew >= 1.0 - 1e-8
                    zero = rew <= 1e-8
                    if solved:
                        d["solved"] += 1
                    elif zero:
                        d["zero"] += 1
                    else:
                        d["partial"] += 1
                    if not solved:
                        d["n_unsolved"] += 1
                        d["rlen_unsolved_sum"] += rlen
                    if trunc:
                        d["trunc"] += 1
                        d["trunc_solved" if solved else ("trunc_zero" if zero else "trunc_partial")] += 1
                    else:
                        if zero:
                            d["stop_zero"] += 1
                        elif not solved:
                            d["stop_partial"] += 1
        if inst_min is not None:
            instance_ranges[inst] = (inst_min, inst_max)
    return steps, instance_ranges, overlap_steps, group_outcomes


def _slope(xs, ys):
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), 1)[0])


def report_group_consistency(group_outcomes, ordered_steps, n_bins=10):
    """Per-prompt group outcomes over training: all-solved / all-failed / mixed.

    IMPORTANT caveat: saved traces are the batch that reaches the trainer, which with
    `--active_sampling` (+ filter_zero_std_samples) is pre-selected to be INFORMATIVE — i.e. only
    mixed (non-zero-std) groups. So with active sampling this table is ~100% mixed by construction and
    does NOT reveal generation-time consistency; it confirms active sampling is working and that the
    trained-batch solve rate is HELD near the informative point by selection (not the model's eval
    accuracy). The true sharpening signal lives on wandb: a rising `val/avg_group_performance_pre_filter`
    against a ~flat trace/`scores` solve rate means more prompts are becoming fully solved at GENERATION
    time (then filtered out). Without active sampling, all-solved/all-failed shares are meaningful and a
    growing all-solved+all-failed share = the model sharpening per-prompt.
    """
    if not group_outcomes:
        return
    lo, hi = ordered_steps[0], ordered_steps[-1]
    width = max(1, (hi - lo + 1) / n_bins)
    bins = defaultdict(lambda: dict(allsolved=0, allfailed=0, mixed=0, groups=0, lo=None, hi=None, frac_sum=0.0))
    tot_groups = tot_mixed = 0
    for st in ordered_steps:
        b = int((st - lo) // width)
        d = bins[b]
        d["lo"] = st if d["lo"] is None else min(d["lo"], st)
        d["hi"] = st if d["hi"] is None else max(d["hi"], st)
        for _pidx, (nsolved, ntot) in group_outcomes[st].items():
            if ntot < 2:  # need a real group to talk about consistency
                continue
            d["groups"] += 1
            d["frac_sum"] += nsolved / ntot
            tot_groups += 1
            if nsolved == ntot:
                d["allsolved"] += 1
            elif nsolved == 0:
                d["allfailed"] += 1
            else:
                d["mixed"] += 1
                tot_mixed += 1
    print("\n=== PER-PROMPT GROUP CONSISTENCY over training (samples-per-prompt groups in saved traces) ===")
    print(f"{'step-range':>15}{'groups':>8}{'allSolved%':>11}{'allFailed%':>11}{'mixed%':>8}{'meanSolve%':>11}")
    for b in sorted(bins):
        d = bins[b]
        g = d["groups"]
        if not g:
            continue
        rng = f"{d['lo']}..{d['hi']}"
        print(
            f"{rng:>15}{g:>8}{100 * d['allsolved'] / g:>11.1f}{100 * d['allfailed'] / g:>11.1f}"
            f"{100 * d['mixed'] / g:>8.1f}{100 * d['frac_sum'] / g:>11.1f}"
        )
    mixed_share = tot_mixed / tot_groups if tot_groups else 0
    if mixed_share > 0.95:
        print("  -> ~100% mixed: active sampling is selecting only informative (non-zero-std) groups for")
        print("     training, so this is NOT the model's eval accuracy and is held ~flat by selection.")
        print("     For true task progress read `val/avg_group_performance_pre_filter` on wandb (it rising")
        print("     against a flat trace solve rate = more prompts solved at generation time, then filtered).")
    else:
        print("  -> all-solved + all-failed share growing over training = the model sharpening per-prompt.")


def report(steps, instance_ranges, overlap_steps, response_length: int, group_outcomes=None, n_bins: int = 10):
    if not steps:
        print("No rollout records found.")
        return

    print("\n=== RUN-INSTANCES (restarts share one step counter) ===")
    for inst, (lo, hi) in instance_ranges.items():
        print(f"  instance {inst}: steps {lo}..{hi}")
    if overlap_steps:
        print(
            f"  NOTE: {len(overlap_steps)} step(s) produced by >1 instance (restart re-did them); counts include both."
        )

    ordered = sorted(steps)

    # per-step derived series
    def fail_trunc_share(d):
        return d["trunc_zero"] / d["zero"] if d["zero"] else None

    # ---- binned trend over training ----
    print("\n=== TREND OVER TRAINING (binned) ===")
    print(
        f"{'step-range':>15}{'rollouts':>9}{'solve%':>8}{'fail%':>7}{'trunc%':>8}"
        f"{'fail=trunc%':>12}{'fail=stop%':>11}{'avgLen':>8}{'unslvLen':>9}{'calls':>7}"
    )
    lo, hi = ordered[0], ordered[-1]
    width = max(1, (hi - lo + 1) / n_bins)
    bins = defaultdict(_blank)
    for st in ordered:
        b = int((st - lo) // width)
        agg = bins[b]
        for k in steps[st]:
            agg[k] += steps[st][k]
        agg.setdefault("_lo", st)
        agg["_lo"] = min(agg.get("_lo", st), st)
        agg["_hi"] = max(agg.get("_hi", st), st)
    bin_centers, bin_truncshare = [], []
    for b in sorted(bins):
        d = bins[b]
        n = d["n"]
        if not n:
            continue
        failz = d["zero"]
        fts = 100 * d["trunc_zero"] / failz if failz else 0
        fss = 100 * d["stop_zero"] / failz if failz else 0
        unslv = d["rlen_unsolved_sum"] / d["n_unsolved"] if d["n_unsolved"] else 0
        rng = f"{d['_lo']}..{d['_hi']}"
        print(
            f"{rng:>15}"
            f"{n:>9}{100 * d['solved'] / n:>8.1f}{100 * (d['zero'] + d['partial']) / n:>7.1f}"
            f"{100 * d['trunc'] / n:>8.1f}{fts:>12.1f}{fss:>11.1f}"
            f"{d['rlen_sum'] / n:>8.0f}{unslv:>9.0f}{d['calls_sum'] / n:>7.1f}"
        )
        if failz:
            bin_centers.append((d["_lo"] + d["_hi"]) / 2)
            bin_truncshare.append(fts)

    # ---- aggregate ----
    A = _blank()
    for st in ordered:
        for k in steps[st]:
            A[k] += steps[st][k]
    n = A["n"]
    failz = A["zero"]
    print("\n=== AGGREGATE (all steps, all instances) ===")
    print(f"  rollouts={n}  over steps {lo}..{hi}")
    print(
        f"  solved={A['solved']} ({100 * A['solved'] / n:.1f}%)  zero-reward={A['zero']} ({100 * A['zero'] / n:.1f}%)"
        + (
            f"  partial={A['partial']} ({100 * A['partial'] / n:.1f}%)"
            if A["partial"]
            else "  (reward is BINARY: no partial credit)"
        )
    )
    print(
        f"  truncated overall={A['trunc']} ({100 * A['trunc'] / n:.1f}%); of those, solved={A['trunc_solved']} "
        f"({100 * A['trunc_solved'] / A['trunc']:.1f}% of truncated) -> truncation ~= guaranteed failure"
    )
    if failz:
        print(f"\n  Decomposition of ZERO-reward failures ({failz}):")
        print(f"    truncated (ran out of budget):  {A['trunc_zero']:>6} ({100 * A['trunc_zero'] / failz:.1f}%)")
        print(f"    stopped normally but wrong:     {A['stop_zero']:>6} ({100 * A['stop_zero'] / failz:.1f}%)")

    # ---- the headline trend answer ----
    if len(bin_centers) >= 2:
        sl = _slope(bin_centers, bin_truncshare)
        direction = "RISING" if sl > 0.02 else ("FALLING" if sl < -0.02 else "FLAT")
        # show fitted-line endpoints (not raw noisy first/last bins) so they match the arrow
        intercept = sum(bin_truncshare) / len(bin_truncshare) - sl * (sum(bin_centers) / len(bin_centers))
        fit_lo = sl * bin_centers[0] + intercept
        fit_hi = sl * bin_centers[-1] + intercept
        print(
            f"\n  TRUNCATION-SHARE-OF-FAILURES trend over training: {direction} "
            f"({fit_lo:.0f}% -> {fit_hi:.0f}% fitted, slope {sl:+.3f} %/step)."
        )
        if direction == "RISING":
            print(
                "    -> failures are increasingly budget-bound as training proceeds (model generates longer). "
                "Raising response_length and/or rewarding conciseness becomes more valuable over time."
            )
        elif direction == "FALLING":
            print("    -> the model is learning to finish within budget; remaining failures are increasingly genuine.")

    if group_outcomes:
        report_group_consistency(group_outcomes, ordered, n_bins)


def decode_examples(shards_by_instance, response_length, step, k, tokenizer_name, tail_tokens=260):
    from transformers import AutoTokenizer  # noqa: PLC0415  (optional heavy dep; only needed for --decode-examples)

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    buckets = {"trunc_zero": [], "stop_zero": [], "solved": []}
    done = False
    for _inst, files in shards_by_instance.items():
        if done:
            break
        for fp in files:
            with open(fp) as f:
                for line in f:
                    m = STEP_PREFIX_RE.search(line[:64])
                    if not m or int(m.group(1)) != step:
                        continue
                    r = json.loads(line)
                    rew, fr = r["reward"], r["finish_reason"]
                    rlen = len(r["response_tokens"])
                    trunc = (fr != "stop") or (rlen >= response_length)
                    cat = "solved" if rew >= 1 - 1e-8 else ("trunc_zero" if trunc else "stop_zero")
                    if len(buckets[cat]) < k:
                        buckets[cat].append(r)
                    if all(len(v) >= k for v in buckets.values()):
                        done = True
                        break
            if done:
                break
    print(f"\n=== DECODED EXAMPLES (step {step}, last ~{tail_tokens} tokens each) ===")
    for cat, recs in buckets.items():
        for r in recs:
            ri = r.get("request_info", {})
            print("\n" + "-" * 92)
            print(
                f"### {cat} | reward={r['reward']} finish_reason={r['finish_reason']} "
                f"resp_tokens={len(r['response_tokens'])} num_calls={ri.get('num_calls')} timeouts={ri.get('timeouts')}"
            )
            print(tok.decode(r["response_tokens"][-tail_tokens:]))


def run(
    exp_name,
    rollouts_dir=DEFAULT_ROLLOUTS_DIR,
    response_length=DEFAULT_RESPONSE_LENGTH,
    per_step_cap=0,
    decode=0,
    step=None,
    tokenizer="hamishivi/Qwen3.5-4B",
):
    shards = find_shards(rollouts_dir, exp_name)
    if not shards:
        print(f"No rollout shards found for exp_name={exp_name!r} in {rollouts_dir}")
        print("(Did the run use --save_traces, and is the path right?)")
        return
    nfiles = sum(len(v) for v in shards.values())
    print(
        f"Found {nfiles} shard(s) across {len(shards)} run-instance(s) for {exp_name!r} "
        f"(response_length cap = {response_length}; per_step_cap = {per_step_cap or 'all'})."
    )
    steps, ranges, overlap, group_outcomes = classify_shards(shards, response_length, per_step_cap)
    report(steps, ranges, overlap, response_length, group_outcomes)
    if decode:
        tgt = step if step is not None else max(steps)
        decode_examples(shards, response_length, tgt, decode, tokenizer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-name", required=True, help="exp_name prefix of the run (matches all restart instances)")
    ap.add_argument("--rollouts-dir", default=DEFAULT_ROLLOUTS_DIR)
    ap.add_argument(
        "--response-length",
        type=int,
        default=DEFAULT_RESPONSE_LENGTH,
        help="must match the run's --response_length for correct truncation labeling",
    )
    ap.add_argument(
        "--per-step-cap",
        type=int,
        default=0,
        help="sample at most N rollouts/step for a fast pass (0 = all). 64-128 is plenty for fractions.",
    )
    ap.add_argument("--decode-examples", type=int, default=0, help="decode K example tails per bucket")
    ap.add_argument("--step", type=int, default=None, help="step to pull decode examples from (default: last)")
    ap.add_argument("--tokenizer", default="hamishivi/Qwen3.5-4B")
    args = ap.parse_args()
    run(
        args.exp_name,
        args.rollouts_dir,
        args.response_length,
        args.per_step_cap,
        args.decode_examples,
        args.step,
        args.tokenizer,
    )


if __name__ == "__main__":
    main()
