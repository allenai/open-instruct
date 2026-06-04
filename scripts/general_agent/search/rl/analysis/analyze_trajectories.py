#!/usr/bin/env python3
"""Classify Deep-Research (DR-Tulu) RL rollouts from saved traces: reward, budget, tools.

Reads the JSONL trace shards written by ``--save_traces`` (RolloutRecord, see
``open_instruct/rl_utils.py``) and answers, for a search/browse agent graded by a
**continuous evolving-rubric reward** (0..``max_possible_score``):

  * Where does reward sit? — the zero / partial / full(==max) split, and the mean.
  * Is hitting the token budget costing reward? — mean reward of **truncated** vs
    **completed** rollouts. (Unlike a binary env, DR truncation is the NORM and
    only mildly penalized, because the rubric judges the partial answer.)
  * How is the agent using its tools? — per-tool call counts and failure rates,
    pulled from ``request_info.tool_call_stats`` (a list of
    ``{tool_name, success, runtime}`` per call), plus turns (``step_count``).
  * Within-group reward spread — GRPO learns from *disagreement inside a group*.
    For continuous reward the signal is the within-group std, not all-solved /
    all-failed. Groups with ~zero std contribute ~zero gradient.

Truncation uses the SAME definition as the training code
(``open_instruct/data_loader.py``):
    truncated  <=>  finish_reason != "stop"  OR  len(response_tokens) >= response_length

Files are multi-GB; everything is streamed line-by-line. A cheap step-prefix
parse lets ``--per-step-cap`` skip the expensive json.loads on capped steps.

Usage:
    uv run python analyze_trajectories.py --exp-name drtulu_rl_qwen35_4b
    uv run python analyze_trajectories.py --exp-name <name> --per-step-cap 64        # fast sample
    uv run python analyze_trajectories.py --exp-name <name> --decode-examples 1 --step 10
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

DEFAULT_ROLLOUTS_DIR = "/weka/oe-adapt-default/allennlp/deletable_rollouts/"
DEFAULT_RESPONSE_LENGTH = 16384
DEFAULT_MAX_SCORE = 10.0
STEP_PREFIX_RE = re.compile(r'"step":\s*(\d+)')
INSTANCE_RE = re.compile(r"__(\d+)_rollouts_\d+\.jsonl$")
TOOLS = ("google_search", "browse_webpage", "snippet_search")


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
    d = dict(
        n=0,
        zero=0,
        partial=0,
        full=0,
        rew_sum=0.0,
        trunc=0,
        trunc_rew_sum=0.0,
        trunc_zero=0,
        comp=0,
        comp_rew_sum=0.0,
        rlen_sum=0,
        rlen_unsolved_sum=0,
        n_unsolved=0,
        calls_sum=0,
        turns_sum=0,
        tool_err_sum=0,
    )
    for t in TOOLS:
        d[f"calls_{t}"] = 0
        d[f"fail_{t}"] = 0
    return d


def _parse_tool_stats(ri):
    """Return (per_tool_calls dict, per_tool_fail dict, total_calls, total_fail) from request_info."""
    calls = defaultdict(int)
    fails = defaultdict(int)
    stats = ri.get("tool_call_stats")
    if isinstance(stats, str):
        try:
            stats = json.loads(stats)
        except (json.JSONDecodeError, ValueError):
            stats = None
    if isinstance(stats, list):
        for s in stats:
            if not isinstance(s, dict):
                continue
            name = s.get("tool_name", "unknown")
            calls[name] += 1
            if not s.get("success", True):
                fails[name] += 1
    return calls, fails, sum(calls.values()), sum(fails.values())


def classify_shards(shards_by_instance, response_length: int, max_score: float, per_step_cap: int = 0):
    """Stream all shards; return per-step aggregated stats, instance ranges, and per-group reward lists."""
    steps = defaultdict(_blank)
    group_rewards = defaultdict(lambda: defaultdict(list))  # step -> prompt_idx -> [rewards]
    instance_ranges = {}
    overlap_steps = set()
    seen_step_instance = {}
    full_thresh = max_score - 1e-8

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
                    if st in seen_step_instance and seen_step_instance[st] != inst:
                        overlap_steps.add(st)
                    seen_step_instance[st] = inst
                    if per_step_cap and steps[st]["n"] >= per_step_cap:
                        continue

                    r = json.loads(line)
                    rew = float(r["reward"])
                    fr = r["finish_reason"]
                    rlen = len(r["response_tokens"])
                    ri = r.get("request_info", {})
                    calls = ri.get("num_calls", 0) or 0
                    rs = ri.get("rollout_state") or {}
                    turns = rs.get("step_count", calls) or 0
                    pcalls, pfails, tcalls, tfails = _parse_tool_stats(ri)

                    group_rewards[st][r.get("prompt_idx", -1)].append(rew)

                    d = steps[st]
                    d["n"] += 1
                    d["rew_sum"] += rew
                    d["rlen_sum"] += rlen
                    d["calls_sum"] += calls
                    d["turns_sum"] += turns
                    d["tool_err_sum"] += tfails
                    for t in TOOLS:
                        d[f"calls_{t}"] += pcalls.get(t, 0)
                        d[f"fail_{t}"] += pfails.get(t, 0)

                    trunc = (fr != "stop") or (rlen >= response_length)
                    zero = rew <= 1e-8
                    full = rew >= full_thresh
                    if zero:
                        d["zero"] += 1
                    elif full:
                        d["full"] += 1
                    else:
                        d["partial"] += 1
                    if not full:
                        d["n_unsolved"] += 1
                        d["rlen_unsolved_sum"] += rlen
                    if trunc:
                        d["trunc"] += 1
                        d["trunc_rew_sum"] += rew
                        if zero:
                            d["trunc_zero"] += 1
                    else:
                        d["comp"] += 1
                        d["comp_rew_sum"] += rew
        if inst_min is not None:
            instance_ranges[inst] = (inst_min, inst_max)
    return steps, instance_ranges, overlap_steps, group_rewards


def _slope(xs, ys):
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), 1)[0])


def report_group_spread(group_rewards, ordered_steps, max_score, n_bins=10):
    """Within-group reward spread over training — the quantity GRPO's advantage uses.

    For continuous reward, a group contributes gradient in proportion to its reward
    std (advantage = reward - group mean). ~zero-std groups (all rollouts scored the
    same) give ~zero gradient. With --active_sampling + filter_zero_std_samples the
    saved (trained-on) batch is pre-selected to be informative, so the zero-std share
    here should be small by construction; a rising mean group-std would mean the model
    is finding more contrastable prompts, a shrinking one means it is homogenizing.
    """
    if not group_rewards:
        return
    lo, hi = ordered_steps[0], ordered_steps[-1]
    width = max(1, (hi - lo + 1) / n_bins)
    bins = defaultdict(lambda: dict(groups=0, std_sum=0.0, mean_sum=0.0, zerostd=0, lo=None, hi=None))
    tot_groups = tot_zerostd = 0
    for st in ordered_steps:
        b = int((st - lo) // width)
        d = bins[b]
        d["lo"] = st if d["lo"] is None else min(d["lo"], st)
        d["hi"] = st if d["hi"] is None else max(d["hi"], st)
        for _pidx, rews in group_rewards[st].items():
            if len(rews) < 2:
                continue
            arr = np.array(rews, dtype=float)
            d["groups"] += 1
            d["std_sum"] += float(arr.std())
            d["mean_sum"] += float(arr.mean())
            tot_groups += 1
            if arr.std() < 1e-6:
                d["zerostd"] += 1
                tot_zerostd += 1
    print("\n=== PER-PROMPT GROUP REWARD SPREAD over training (samples-per-prompt groups in saved traces) ===")
    print(f"{'step-range':>15}{'groups':>8}{'meanRew':>9}{'meanStd':>9}{'zeroStd%':>10}")
    for b in sorted(bins):
        d = bins[b]
        g = d["groups"]
        if not g:
            continue
        rng = f"{d['lo']}..{d['hi']}"
        print(f"{rng:>15}{g:>8}{d['mean_sum'] / g:>9.2f}{d['std_sum'] / g:>9.2f}{100 * d['zerostd'] / g:>10.1f}")
    zshare = tot_zerostd / tot_groups if tot_groups else 0
    print(
        f"  -> mean within-group reward std is the gradient magnitude (reward scale 0..{max_score:g}). "
        f"zero-std groups = {100 * zshare:.1f}% (give ~no gradient)."
    )
    if zshare < 0.05:
        print(
            "     ~0% zero-std: active sampling + filter_zero_std_samples is selecting informative groups, so this "
            "is the trained-on batch, not generation-time accuracy (read avg_group_performance_pre_filter on wandb)."
        )


def report(steps, instance_ranges, overlap_steps, response_length, max_score, group_rewards=None, n_bins=10):
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

    # ---- binned trend over training ----
    print(f"\n=== TREND OVER TRAINING (binned; reward scale 0..{max_score:g}) ===")
    print(
        f"{'step-range':>13}{'rollouts':>9}{'meanRew':>8}{'zero%':>7}{'full%':>7}"
        f"{'trunc%':>7}{'truncRew':>9}{'compRew':>8}{'avgLen':>8}{'turns':>6}{'calls':>6}{'toolErr%':>9}"
    )
    lo, hi = ordered[0], ordered[-1]
    width = max(1, (hi - lo + 1) / n_bins)
    bins = defaultdict(_blank)
    for st in ordered:
        b = int((st - lo) // width)
        agg = bins[b]
        for k in steps[st]:
            agg[k] += steps[st][k]
        agg["_lo"] = min(agg.get("_lo", st), st)
        agg["_hi"] = max(agg.get("_hi", st), st)
    bin_centers, bin_meanrew = [], []
    for b in sorted(bins):
        d = bins[b]
        n = d["n"]
        if not n:
            continue
        trunc_rew = d["trunc_rew_sum"] / d["trunc"] if d["trunc"] else 0
        comp_rew = d["comp_rew_sum"] / d["comp"] if d["comp"] else 0
        tool_err = 100 * d["tool_err_sum"] / d["calls_sum"] if d["calls_sum"] else 0
        rng = f"{d['_lo']}..{d['_hi']}"
        print(
            f"{rng:>13}{n:>9}{d['rew_sum'] / n:>8.2f}{100 * d['zero'] / n:>7.1f}{100 * d['full'] / n:>7.1f}"
            f"{100 * d['trunc'] / n:>7.1f}{trunc_rew:>9.2f}{comp_rew:>8.2f}"
            f"{d['rlen_sum'] / n:>8.0f}{d['turns_sum'] / n:>6.1f}{d['calls_sum'] / n:>6.1f}{tool_err:>9.1f}"
        )
        bin_centers.append((d["_lo"] + d["_hi"]) / 2)
        bin_meanrew.append(d["rew_sum"] / n)

    # ---- per-tool usage ----
    A = _blank()
    for st in ordered:
        for k in steps[st]:
            A[k] += steps[st][k]
    n = A["n"]
    print("\n=== TOOL USAGE (aggregate, all steps) ===")
    print(f"{'tool':>16}{'calls/rollout':>15}{'failure%':>10}{'share%':>9}")
    tot_tool_calls = sum(A[f"calls_{t}"] for t in TOOLS) or 1
    for t in TOOLS:
        c = A[f"calls_{t}"]
        fr = 100 * A[f"fail_{t}"] / c if c else 0
        print(f"{t:>16}{c / n:>15.2f}{fr:>10.1f}{100 * c / tot_tool_calls:>9.1f}")
    print(f"{'ALL':>16}{tot_tool_calls / n:>15.2f}{100 * A['tool_err_sum'] / tot_tool_calls:>10.1f}{100.0:>9.1f}")

    # ---- aggregate reward ----
    print("\n=== AGGREGATE (all steps, all instances) ===")
    print(f"  rollouts={n}  over steps {lo}..{hi}  (reward scale 0..{max_score:g})")
    print(
        f"  mean reward={A['rew_sum'] / n:.3f} ({100 * A['rew_sum'] / n / max_score:.1f}% of max)  "
        f"| zero={A['zero']} ({100 * A['zero'] / n:.1f}%)  "
        f"partial={A['partial']} ({100 * A['partial'] / n:.1f}%)  "
        f"full={A['full']} ({100 * A['full'] / n:.1f}%)"
    )
    if A["trunc"]:
        print(
            f"  truncated={A['trunc']} ({100 * A['trunc'] / n:.1f}%): mean reward {A['trunc_rew_sum'] / A['trunc']:.2f}"
            f" vs completed {A['comp_rew_sum'] / A['comp'] if A['comp'] else 0:.2f}"
            f"  (truncated-zero share {100 * A['trunc_zero'] / A['trunc']:.1f}%)"
        )
        delta = (A["comp_rew_sum"] / A["comp"] if A["comp"] else 0) - A["trunc_rew_sum"] / A["trunc"]
        if A["trunc"] / n > 0.4 and delta < 0.15 * max_score:
            print(
                "    -> truncation is the NORM and only mildly penalized: hitting the budget does NOT mean failure "
                "(the rubric scores the partial answer). Raising response_length buys answer quality, not pass/fail."
            )
        elif delta >= 0.15 * max_score:
            print(
                f"    -> completed rollouts score notably higher (+{delta:.2f}): budget IS costing reward; "
                "raising response_length / encouraging conciseness should help."
            )
    print(
        f"  avg tool calls/rollout={A['calls_sum'] / n:.2f}  avg turns={A['turns_sum'] / n:.2f}  "
        f"avg response_len={A['rlen_sum'] / n:.0f}"
    )

    # ---- headline trend ----
    if len(bin_centers) >= 2:
        sl = _slope(bin_centers, bin_meanrew)
        direction = "RISING" if sl > 0.01 else ("FALLING" if sl < -0.01 else "FLAT")
        # show fitted-line endpoints (not raw noisy first/last bins) so they match the arrow
        intercept = sum(bin_meanrew) / len(bin_meanrew) - sl * (sum(bin_centers) / len(bin_centers))
        fit_lo = sl * bin_centers[0] + intercept
        fit_hi = sl * bin_centers[-1] + intercept
        print(
            f"\n  MEAN-REWARD trend over training (against the EVOLVING rubric bar): {direction} "
            f"({fit_lo:.2f} -> {fit_hi:.2f} fitted, slope {sl:+.3f}/step)."
        )
        print(
            "    Reward here is graded against rubrics that get stricter over training, so a flat/falling raw mean "
            "can still be improvement. Cross-check the wandb correct_rate (frac reward>0) and rubric-count growth."
        )

    if group_rewards:
        report_group_spread(group_rewards, ordered, max_score, n_bins)


def decode_examples(shards_by_instance, response_length, max_score, step, k, tokenizer_name, tail_tokens=300):
    from transformers import AutoTokenizer  # noqa: PLC0415  (optional heavy dep; only for --decode-examples)

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    buckets = {"zero": [], "partial": [], "full": []}
    full_thresh = max_score - 1e-8
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
                    rew = float(r["reward"])
                    cat = "full" if rew >= full_thresh else ("zero" if rew <= 1e-8 else "partial")
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
            pcalls, pfails, tcalls, tfails = _parse_tool_stats(ri)
            rlen = len(r["response_tokens"])
            trunc = (r["finish_reason"] != "stop") or (rlen >= response_length)
            tool_summary = ", ".join(f"{t}={pcalls.get(t, 0)}" for t in TOOLS if pcalls.get(t, 0))
            gt = r.get("ground_truth")
            try:
                gtp = json.loads(gt[0] if isinstance(gt, list) else gt)
                query = (gtp.get("query") or gtp.get("Question") or "")[:120]
                nrub = len(gtp.get("rubrics", []))
            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                query, nrub = "", 0
            print("\n" + "-" * 92)
            print(
                f"### {cat} | reward={r['reward']:.2f}/{max_score:g} finish_reason={r['finish_reason']} "
                f"resp_tokens={rlen} truncated={trunc} | tools[{tool_summary}] fails={tfails} rubrics={nrub}"
            )
            if query:
                print(f"query: {query}")
            print(tok.decode(r["response_tokens"][-tail_tokens:]))


def run(
    exp_name,
    rollouts_dir=DEFAULT_ROLLOUTS_DIR,
    response_length=DEFAULT_RESPONSE_LENGTH,
    max_score=DEFAULT_MAX_SCORE,
    per_step_cap=0,
    decode=0,
    step=None,
    tokenizer="Qwen/Qwen3.5-4B",
):
    shards = find_shards(rollouts_dir, exp_name)
    if not shards:
        print(f"No rollout shards found for exp_name={exp_name!r} in {rollouts_dir}")
        print("(Did the run use --save_traces, and is the path right?)")
        return
    nfiles = sum(len(v) for v in shards.values())
    print(
        f"Found {nfiles} shard(s) across {len(shards)} run-instance(s) for {exp_name!r} "
        f"(response_length cap = {response_length}; max_score = {max_score:g}; per_step_cap = {per_step_cap or 'all'})."
    )
    steps, ranges, overlap, group_rewards = classify_shards(shards, response_length, max_score, per_step_cap)
    report(steps, ranges, overlap, response_length, max_score, group_rewards)
    if decode:
        tgt = step if step is not None else max(steps)
        decode_examples(shards, response_length, max_score, tgt, decode, tokenizer)


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
        "--max-score",
        type=float,
        default=DEFAULT_MAX_SCORE,
        help="must match the run's --max_possible_score (reward scale; default 10)",
    )
    ap.add_argument(
        "--per-step-cap",
        type=int,
        default=0,
        help="sample at most N rollouts/step for a fast pass (0 = all). 64-128 is plenty for fractions.",
    )
    ap.add_argument("--decode-examples", type=int, default=0, help="decode K example tails per reward bucket")
    ap.add_argument("--step", type=int, default=None, help="step to pull decode examples from (default: last)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()
    run(
        args.exp_name,
        args.rollouts_dir,
        args.response_length,
        args.max_score,
        args.per_step_cap,
        args.decode_examples,
        args.step,
        args.tokenizer,
    )


if __name__ == "__main__":
    main()
