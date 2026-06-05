#!/usr/bin/env python3
"""Inspect Deep-Research (DR-Tulu) RL rollouts qualitatively: full transcript + rubric analysis.

Companion to ``analyze_trajectories.py`` (which gives the *aggregate* reward / budget /
tool / group-spread stats). This script is for the two things that one cannot show:

  1. ``--view`` — a **full multi-turn trajectory viewer**. Reconstructs one rollout's
     search/browse transcript turn-by-turn from ``response_tokens``: every reasoning
     block, every tool call (parsed ``<function=...>`` name + ``<parameter=...>`` args),
     every tool response, and the trailing answer (if the rollout produced one) — under a
     header tying it to what is graded (reward, the rubric list with weights+types,
     truncation, per-tool call counts). This is the "read what the agent actually did"
     tool that the 300-token tail dump in ``analyze_trajectories.py --decode-examples``
     can't give you.

  2. ``--rubrics`` — a **per-rubric reward analysis**. IMPORTANT: the rubric judge
     (``open_instruct/ground_truth_utils.py::RubricVerifier``) only persists the final
     *weighted-average* score per rollout — the per-rubric pass/fail verdicts are NOT
     saved in the traces. So this is necessarily a **structural + correlational** view:
     it inventories the rubric sets (persistent vs evolving, positive vs negative-weight
     "penalty" rubrics), tracks how the evolving share grows over training (the rising
     bar), and correlates achieved reward with rubric composition (count, #evolving,
     #penalty, total weight). To get true per-rubric pass rates you would have to re-run
     the GPT-4.1 judge on the saved answers (out of scope here — noted in the output).

Transcript markers parsed (Qwen3.5 / ``<|im_start|>`` chat format):
    <|im_start|>assistant / <|im_start|>user  -> turn boundaries
    <think>...</think>                         -> reasoning
    <tool_call><function=NAME><parameter=P>v</parameter></function></tool_call>  -> tool call
    <tool_response>...</tool_response>         -> tool result

Files are multi-GB; everything is streamed line-by-line and stops as soon as the
requested rollouts are found (``--view``) or the per-step cap is hit (``--rubrics``).

Usage:
    # view one full-reward + one zero-reward trajectory at the last step
    uv run python inspect_trajectories.py --exp-name drtulu_rl_qwen35_4b --view

    # view 2 trajectories from a specific reward bucket / step / prompt
    uv run python inspect_trajectories.py --exp-name <name> --view \
        --reward-bucket partial --step 10 --num 2 --result-chars 800

    # per-rubric structural + correlational analysis across the run
    uv run python inspect_trajectories.py --exp-name <name> --rubrics --per-step-cap 96
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_trajectories as at  # noqa: E402  (shared: find_shards, STEP_PREFIX_RE, _parse_tool_stats, defaults)

ROLE_SPLIT_RE = re.compile(r"<\|im_start\|>(assistant|user)\b")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
TOOLCALL_RE = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)
TOOLRESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
CLEAN_RE = re.compile(r"<\|im_end\|>|<\|im_start\|>(assistant|user)?|<think>|</think>")


# --------------------------------------------------------------------------------------
# ground-truth / rubric parsing
# --------------------------------------------------------------------------------------
def parse_ground_truth(r):
    """Return dict(query, rubrics=[{title,description,weight,type}], or None)."""
    gt = r.get("ground_truth")
    g = gt[0] if isinstance(gt, list) and gt else gt
    if isinstance(g, str):
        try:
            g = json.loads(g)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(g, dict):
        return None
    rubrics = g.get("rubrics", []) or []
    types = g.get("rubrics_types", []) or []
    out = []
    for i, rb in enumerate(rubrics):
        if not isinstance(rb, dict):
            continue
        out.append(
            {
                "title": rb.get("title") or rb.get("rubric_item") or "",
                "description": rb.get("description") or rb.get("rubric_item") or rb.get("Ingredient", ""),
                "weight": float(rb.get("weight", 1.0)),
                "type": types[i] if i < len(types) else "?",
            }
        )
    return {"query": g.get("query") or g.get("Question") or "", "rubrics": out}


def rubric_features(gt):
    """Composition features used for the reward-correlation table."""
    rubs = gt["rubrics"]
    return {
        "n": len(rubs),
        "n_evolving": sum(1 for r in rubs if r["type"] == "evolving"),
        "n_persistent": sum(1 for r in rubs if r["type"] == "persistent"),
        "n_penalty": sum(1 for r in rubs if r["weight"] < 0),
        "total_pos_weight": sum(r["weight"] for r in rubs if r["weight"] > 0),
    }


# --------------------------------------------------------------------------------------
# transcript segmentation (for --view)
# --------------------------------------------------------------------------------------
def split_turns(text):
    """Split a decoded response into [(role, body), ...]; leading text is the first assistant turn."""
    parts = ROLE_SPLIT_RE.split(text)
    turns = []
    if parts[0].strip():
        turns.append(("assistant", parts[0]))
    for i in range(1, len(parts), 2):
        role = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        turns.append((role, body))
    return turns


def parse_tool_call(block):
    """(name, [(param, value)]) from one <function=...>...</function> body."""
    params = [(p, v.strip()) for p, v in PARAM_RE.findall(block)]
    return params


def render_turn(role, body, reasoning_chars, result_chars):
    """Yield printable lines for one turn."""
    lines = []
    if role == "user":
        for resp in TOOLRESP_RE.findall(body):
            resp = resp.strip()
            shown = resp if len(resp) <= result_chars else resp[:result_chars] + f"\n    … (+{len(resp) - result_chars} chars)"
            lines.append("  ↙ TOOL RESULT:")
            lines.extend("    " + ln for ln in shown.splitlines())
        return lines

    # assistant turn: reasoning, then tool calls, then any trailing prose (the answer)
    think = "\n".join(THINK_RE.findall(body))
    # the first turn often has reasoning before a lone </think> with no opening <think>
    if not think and "</think>" in body:
        think = body.split("</think>")[0]
    if think.strip():
        t = think.strip()
        shown = t if len(t) <= reasoning_chars else t[:reasoning_chars] + f" … (+{len(t) - reasoning_chars} chars)"
        lines.append("  \U0001f4ad reasoning:")
        lines.extend("    " + ln for ln in shown.splitlines())

    for name, call_body in TOOLCALL_RE.findall(body):
        params = parse_tool_call(call_body)
        arg_str = "  ".join(f"{p}={v[:120]!r}" for p, v in params)
        lines.append(f"  \U0001f527 TOOL CALL: {name.strip()}({arg_str})")

    # trailing prose = body with think + tool_calls + markers stripped
    prose = TOOLCALL_RE.sub("", body)
    prose = THINK_RE.sub("", prose)
    if "</think>" in prose:
        prose = prose.split("</think>", 1)[1]
    prose = CLEAN_RE.sub("", prose).strip()
    if prose:
        shown = prose if len(prose) <= result_chars else prose[:result_chars] + f"\n    … (+{len(prose) - result_chars} chars)"
        lines.append("  \U0001f4dd answer / prose:")
        lines.extend("    " + ln for ln in shown.splitlines())
    return lines


def view_trajectory(r, response_length, max_score, reasoning_chars, result_chars):
    gt = parse_ground_truth(r) or {"query": "", "rubrics": []}
    ri = r.get("request_info", {})
    pcalls, pfails, tcalls, tfails = at._parse_tool_stats(ri)
    rlen = len(r["response_tokens"])
    trunc = (r["finish_reason"] != "stop") or (rlen >= response_length)
    text = TOKENIZER.decode(r["response_tokens"])
    turns = split_turns(text)
    has_answer = bool(turns) and turns[-1][0] == "assistant" and not turns[-1][1].rstrip().endswith("</tool_response>")

    print("\n" + "=" * 96)
    print(
        f"step={r['step']} sample={r.get('sample_idx')} prompt_idx={r.get('prompt_idx')} | "
        f"reward={r['reward']:.2f}/{max_score:g}  finish={r['finish_reason']}  truncated={trunc}  "
        f"resp_tokens={rlen}  adv={r.get('advantage'):+.2f}"
    )
    tool_summary = ", ".join(f"{t}={pcalls.get(t, 0)}" for t in at.TOOLS if pcalls.get(t, 0)) or "none"
    print(f"tools: [{tool_summary}]  tool_failures={tfails}  ends_with_answer={has_answer}")
    print(f"QUERY: {gt['query'][:300]}")
    if gt["rubrics"]:
        print(f"RUBRICS being graded ({len(gt['rubrics'])}):")
        for rb in gt["rubrics"]:
            tag = "PENALTY" if rb["weight"] < 0 else rb["type"]
            print(f"   [{tag:>10}] w={rb['weight']:+.1f}  {rb['title']}: {rb['description'][:90]}")
    print("-" * 96)
    for ti, (role, body) in enumerate(turns):
        label = "ASSISTANT" if role == "assistant" else "TOOL (user)"
        out = render_turn(role, body, reasoning_chars, result_chars)
        if out:
            print(f"\n[turn {ti}] {label}")
            print("\n".join(out))
    if not has_answer:
        print(
            "\n  ⚠ no trailing assistant answer turn — rollout ended inside a tool response "
            "(budget cut-off / non-submitting). The judge still grades the accumulated response."
        )


def run_view(shards, response_length, max_score, bucket, step, num, reasoning_chars, result_chars):
    full_thresh = max_score - 1e-8

    def in_bucket(rew):
        if bucket == "any":
            return True
        if bucket == "zero":
            return rew <= 1e-8
        if bucket == "full":
            return rew >= full_thresh
        if bucket == "partial":
            return 1e-8 < rew < full_thresh
        return True

    picked = []
    target_step = step
    for _inst, files in shards.items():
        for fp in files:
            with open(fp) as f:
                for line in f:
                    m = at.STEP_PREFIX_RE.search(line[:64])
                    if not m:
                        continue
                    st = int(m.group(1))
                    if target_step is not None and st != target_step:
                        continue
                    r = json.loads(line)
                    if in_bucket(float(r["reward"])):
                        picked.append(r)
                        if len(picked) >= num:
                            break
            if len(picked) >= num:
                break
        if len(picked) >= num:
            break

    if not picked:
        print(f"No rollouts matched bucket={bucket!r} step={target_step}. Try --reward-bucket any or a different --step.")
        return
    print(
        f"\n### FULL TRAJECTORY VIEW — bucket={bucket}, step={target_step if target_step is not None else 'first-found'}, "
        f"showing {len(picked)} rollout(s)"
    )
    for r in picked:
        view_trajectory(r, response_length, max_score, reasoning_chars, result_chars)


# --------------------------------------------------------------------------------------
# per-rubric structural + correlational analysis (for --rubrics)
# --------------------------------------------------------------------------------------
def run_rubrics(shards, max_score, per_step_cap, n_bins=10):
    rows = []  # (step, reward, features)
    type_counter = Counter()
    weight_pos = weight_neg = 0
    seen_per_step = defaultdict(int)
    title_reward = defaultdict(lambda: [0.0, 0])  # persistent-rubric title -> [reward_sum, n]

    for _inst, files in shards.items():
        for fp in files:
            with open(fp) as f:
                for line in f:
                    m = at.STEP_PREFIX_RE.search(line[:64])
                    if not m:
                        continue
                    st = int(m.group(1))
                    if per_step_cap and seen_per_step[st] >= per_step_cap:
                        continue
                    r = json.loads(line)
                    gt = parse_ground_truth(r)
                    if not gt or not gt["rubrics"]:
                        continue
                    seen_per_step[st] += 1
                    rew = float(r["reward"])
                    feat = rubric_features(gt)
                    rows.append((st, rew, feat))
                    for rb in gt["rubrics"]:
                        type_counter[rb["type"]] += 1
                        if rb["weight"] < 0:
                            weight_neg += 1
                        else:
                            weight_pos += 1
                        if rb["type"] == "persistent" and rb["title"]:
                            tr = title_reward[rb["title"]]
                            tr[0] += rew
                            tr[1] += 1

    if not rows:
        print("No rollouts with parseable rubrics found.")
        return

    n = len(rows)
    print("\n" + "#" * 96)
    print("# PER-RUBRIC REWARD ANALYSIS  (structural + correlational)")
    print("#" * 96)
    print(
        "NOTE: the judge persists only the per-rollout *weighted-average* score, not per-rubric\n"
        "      verdicts (RubricVerifier returns one VerificationResult.score). So this is rubric\n"
        "      STRUCTURE vs achieved reward — not literal per-rubric pass rates. (Those would need\n"
        "      re-running the GPT-4.1 judge on the saved answers.)"
    )

    # ---- inventory ----
    total_rubrics = sum(type_counter.values())
    print(f"\n=== RUBRIC INVENTORY ({n} rollouts, {total_rubrics} rubric-instances) ===")
    for ty, c in type_counter.most_common():
        print(f"  {ty:>12}: {c:>6} ({100 * c / total_rubrics:.1f}%)")
    print(f"  {'positive-w':>12}: {weight_pos:>6} ({100 * weight_pos / total_rubrics:.1f}%)")
    print(
        f"  {'penalty(-w)':>12}: {weight_neg:>6} ({100 * weight_neg / total_rubrics:.1f}%)"
        "   <- satisfying these LOWERS reward"
    )
    feats = [f for _, _, f in rows]
    for key, label in [("n", "rubrics/query"), ("n_evolving", "evolving/query"), ("n_penalty", "penalty/query")]:
        vals = [f[key] for f in feats]
        print(f"  mean {label:>16}: {np.mean(vals):.2f}  (min {min(vals)}, max {max(vals)})")
    pen_share = sum(1 for f in feats if f["n_penalty"] > 0) / n
    print(f"  rollouts with >=1 penalty rubric: {100 * pen_share:.1f}%")

    # ---- reward vs composition (binned) ----
    rewards = np.array([rw for _, rw, _ in rows], dtype=float)
    print(f"\n=== REWARD vs RUBRIC COMPOSITION (reward scale 0..{max_score:g}) ===")
    for key, label in [
        ("n", "# rubrics"),
        ("n_evolving", "# evolving rubrics"),
        ("n_penalty", "# penalty rubrics"),
    ]:
        print(f"  by {label}:")
        groups = defaultdict(list)
        for (_, rw, f) in rows:
            groups[f[key]].append(rw)
        for k in sorted(groups):
            v = groups[k]
            print(f"      {k:>2}: meanRew {np.mean(v):>6.2f}   (n={len(v)}, zero%={100 * np.mean(np.array(v) <= 1e-8):.0f})")

    # ---- correlations ----
    print("\n=== CORRELATION of achieved reward with rubric composition (Pearson r) ===")
    for key, label in [
        ("n", "# rubrics"),
        ("n_evolving", "# evolving"),
        ("n_persistent", "# persistent"),
        ("n_penalty", "# penalty"),
        ("total_pos_weight", "total +weight"),
    ]:
        x = np.array([f[key] for f in feats], dtype=float)
        r = float(np.corrcoef(x, rewards)[0, 1]) if x.std() > 1e-9 else float("nan")
        print(f"  reward vs {label:>16}: r={r:+.3f}")
    print("  (negative r on '# evolving' = the evolving bar is where reward is lost — the stricter target.)")

    # ---- evolving share over training (the rising bar) ----
    steps = sorted({s for s, _, _ in rows})
    if len(steps) >= 2:
        lo, hi = steps[0], steps[-1]
        width = max(1, (hi - lo + 1) / n_bins)
        bins = defaultdict(lambda: {"evo": 0, "tot": 0, "rew": 0.0, "n": 0, "lo": None, "hi": None})
        for st, rw, f in rows:
            b = int((st - lo) // width)
            d = bins[b]
            d["evo"] += f["n_evolving"]
            d["tot"] += f["n"]
            d["rew"] += rw
            d["n"] += 1
            d["lo"] = st if d["lo"] is None else min(d["lo"], st)
            d["hi"] = st if d["hi"] is None else max(d["hi"], st)
        print("\n=== EVOLVING-RUBRIC SHARE & REWARD over training ===")
        print(f"{'step-range':>14}{'rollouts':>9}{'rubrics/q':>11}{'evolving%':>11}{'meanRew':>9}")
        for b in sorted(bins):
            d = bins[b]
            if not d["n"]:
                continue
            rng = f"{d['lo']}..{d['hi']}"
            print(
                f"{rng:>14}{d['n']:>9}{d['tot'] / d['n']:>11.2f}"
                f"{100 * d['evo'] / d['tot'] if d['tot'] else 0:>11.1f}{d['rew'] / d['n']:>9.2f}"
            )
        print("  -> rising evolving% with flat/falling meanRew = the bar is getting stricter, not the model worse.")

    # ---- most common persistent rubric themes (these recur; evolving ones are mostly unique) ----
    common = sorted(title_reward.items(), key=lambda kv: kv[1][1], reverse=True)[:12]
    if common and common[0][1][1] > 1:
        print("\n=== MOST FREQUENT PERSISTENT RUBRIC TITLES (mean reward of rollouts graded on them) ===")
        for title, (rsum, cnt) in common:
            if cnt < 2:
                continue
            print(f"  n={cnt:>4}  meanRew={rsum / cnt:>6.2f}  {title[:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-name", required=True, help="exp_name prefix of the run (matches all restart instances)")
    ap.add_argument("--rollouts-dir", default=at.DEFAULT_ROLLOUTS_DIR)
    ap.add_argument("--response-length", type=int, default=at.DEFAULT_RESPONSE_LENGTH)
    ap.add_argument("--max-score", type=float, default=at.DEFAULT_MAX_SCORE)
    ap.add_argument("--view", action="store_true", help="render full multi-turn trajectory transcript(s)")
    ap.add_argument("--rubrics", action="store_true", help="per-rubric structural + correlational reward analysis")
    # --view options
    ap.add_argument("--reward-bucket", choices=["any", "zero", "partial", "full"], default="any")
    ap.add_argument("--step", type=int, default=None, help="step to pull --view rollouts from (default: first found)")
    ap.add_argument("--num", type=int, default=1, help="how many trajectories to render in --view")
    ap.add_argument("--reasoning-chars", type=int, default=700, help="truncate each reasoning block to N chars")
    ap.add_argument("--result-chars", type=int, default=600, help="truncate each tool result / answer to N chars")
    # --rubrics options
    ap.add_argument("--per-step-cap", type=int, default=96, help="sample N rollouts/step in --rubrics (0=all)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()

    if not args.view and not args.rubrics:
        args.view = args.rubrics = True  # default: do both

    shards = at.find_shards(args.rollouts_dir, args.exp_name)
    if not shards:
        print(f"No rollout shards found for exp_name={args.exp_name!r} in {args.rollouts_dir}")
        print("(Did the run use --save_traces, and is the path right?)")
        return
    nfiles = sum(len(v) for v in shards.values())
    print(f"Found {nfiles} shard(s) across {len(shards)} run-instance(s) for {args.exp_name!r}.")

    if args.rubrics:
        run_rubrics(shards, args.max_score, args.per_step_cap)

    if args.view:
        global TOKENIZER
        from transformers import AutoTokenizer  # noqa: PLC0415  (heavy dep; only needed for --view)

        TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer)
        run_view(
            shards,
            args.response_length,
            args.max_score,
            args.reward_bucket,
            args.step,
            args.num,
            args.reasoning_chars,
            args.result_chars,
        )


if __name__ == "__main__":
    main()
