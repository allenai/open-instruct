"""Bin reward by response length over the basket arms' retained training rollouts."""
import collections
import glob
import json
import os
import re

import torch

R = "/weka/oe-training-default/robertb/open-instruct/runs"
ARMS = {
    "dense": [f"{R}/olmo3-think-sft-basket-200-32k-single-node-20260915", f"{R}/olmo3-think-sft-basket-200-32k-g16-20260915",
              f"{R}/olmo3-think-sft-basket-200-32k-kv-20260915", f"{R}/olmo3-think-sft-basket-200-32k-c24-20260916",
              f"{R}/olmo3-think-sft-basket-200-32k-c24b-20260916", f"{R}/olmo3-think-sft-basket-200-32k-c24c-20260916"],
    "moe": [f"{R}/full-sft-basket-200-32k-robust-20260915", f"{R}/full-sft-basket-200-32k-c16-20260915", f"{R}/full-sft-basket-200-32k-c16b-20260916",
            f"{R}/full-sft-basket-200-32k-c16c-20260916", f"{R}/full-sft-basket-200-32k-c16d-20260916", f"{R}/full-sft-basket-200-32k-c16e-20260916",
            f"{R}/full-sft-basket-200-32k-c16f-20260916"],
}
CAP = 32768
BINS = [(0, 2048), (2048, 4096), (4096, 8192), (8192, 12288), (12288, 16384), (16384, 24576), (24576, 32768), (32768, 10**9)]
OUT = f"{R}/length-reward-20260916"
os.makedirs("/output", exist_ok=True); os.makedirs(OUT, exist_ok=True)

def domain_of(md):
    v = (md or {}).get("verifiers") or []
    name = v[0]["name"] if v else str((md or {}).get("source_dataset", "?"))
    n = name.lower()
    if "math" in n or "gsm" in n: return "math"
    if "ifeval" in n or n.startswith("if"): return "if"
    if "code" in n or "stdio" in n or "func" in n: return "code"
    if "general" in n or "quality" in n or "judge" in n: return "general"
    return name

report = {}
for arm, roots in ARMS.items():
    samples = {}  # rollout_id -> list
    files_seen = 0; names = collections.Counter()
    for root in roots:
        for f in glob.glob(f"{root}/rollouts/*.pt"):
            if os.path.basename(f).startswith("eval_"): continue
            rid = int(re.search(r"(\d+)\.pt$", f).group(1))
            try: d = torch.load(f, weights_only=False)
            except Exception as e: print("skip", f, e); continue
            files_seen += 1
            if files_seen % 10 == 0: print(f"  loaded {files_seen} files ({arm})", flush=True)
            rows = []
            for s in d["samples"]:
                md = s.get("metadata") or {}
                r = s.get("reward"); r = r.get("reward", r.get("score")) if isinstance(r, dict) else r
                if r is None: continue
                L = int(s.get("response_length") or 0)
                trunc = L >= CAP or "TRUNC" in str(s.get("status", "")).upper()
                dom = domain_of(md); names[((md.get("verifiers") or [{}])[0].get("name", "?"))] += 1
                rows.append((dom, float(r), L, trunc))
            samples[rid] = rows  # later roots override earlier attempts of the same update
    allrows = [x for rid in sorted(samples) for x in samples[rid]]
    rids = sorted(samples)
    arm_rep = {"files": files_seen, "updates": len(rids), "update_range": [rids[0], rids[-1]] if rids else None, "samples": len(allrows), "verifier_names": dict(names), "domains": {}}
    for dom in sorted({x[0] for x in allrows}):
        rows = [x for x in allrows if x[0] == dom]
        total_r = sum(x[1] for x in rows)
        bins = []
        for lo, hi in BINS:
            b = [x for x in rows if lo <= x[2] < hi]
            n = len(b); sr = sum(x[1] for x in b)
            bins.append({"bin": f"{lo//1024}K-{min(hi,CAP)//1024}K" if hi <= CAP else "cap", "n": n, "frac_samples": n/len(rows) if rows else 0,
                         "mean_reward": sr/n if n else None, "reward_share": sr/total_r if total_r else None,
                         "positive_rate": sum(1 for x in b if x[1] > 0.5)/n if n else None})
        pos = [x for x in rows if x[1] > 0.5]
        arm_rep["domains"][dom] = {"n": len(rows), "mean_reward": total_r/len(rows), "mean_tokens": sum(x[2] for x in rows)/len(rows),
            "truncated_frac": sum(1 for x in rows if x[3])/len(rows),
            "reward_share_over_10k": sum(x[1] for x in rows if x[2] > 10240)/total_r if total_r else None,
            "reward_share_over_16k": sum(x[1] for x in rows if x[2] > 16384)/total_r if total_r else None,
            "positive_over_10k_frac_of_positive": (sum(1 for x in pos if x[2] > 10240)/len(pos)) if pos else None,
            "positive_over_16k_frac_of_positive": (sum(1 for x in pos if x[2] > 16384)/len(pos)) if pos else None,
            "truncated_positive_rate": (sum(1 for x in rows if x[3] and x[1] > 0.5)/max(1, sum(1 for x in rows if x[3]))),
            "bins": bins}
    report[arm] = arm_rep
    print(f"\n=== {arm}: {files_seen} files, updates {arm_rep['update_range']}, {len(allrows)} samples; verifiers {dict(names)}")
    for dom, dr in arm_rep["domains"].items():
        print(f"  {dom}: n={dr['n']} mean_reward={dr['mean_reward']:.3f} mean_tokens={dr['mean_tokens']:.0f} truncated={dr['truncated_frac']:.2%} "
              f"reward_share>10K={dr['reward_share_over_10k']:.2%} >16K={dr['reward_share_over_16k']:.2%} "
              f"positive>10K/positive={dr['positive_over_10k_frac_of_positive']} trunc_positive_rate={dr['truncated_positive_rate']:.3f}")
        for b in dr["bins"]:
            if b["n"]: print(f"     {b['bin']:>8}: n={b['n']:>5} ({b['frac_samples']:.1%})  mean_reward={b['mean_reward']:.3f}  positive_rate={b['positive_rate']:.3f}  reward_share={b['reward_share']:.1%}")
for path in ("/output/length-reward.json", f"{OUT}/length-reward.json"):
    json.dump(report, open(path, "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for i, arm in enumerate(report):
        for j, dom in enumerate(["math", "if", "code", "general"]):
            ax = axes[i][j]; dr = report[arm]["domains"].get(dom)
            if not dr: ax.set_visible(False); continue
            xs = [b["bin"] for b in dr["bins"]]; ax.bar(xs, [b["mean_reward"] or 0 for b in dr["bins"]], color="#2a7f9e")
            ax2 = ax.twinx(); ax2.plot(xs, [b["frac_samples"] for b in dr["bins"]], color="#d9822b", marker="o"); ax2.set_ylim(0, 1)
            ax.set_title(f"{arm} / {dom}: mean reward (bars) and sample share (line) by length"); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig("/output/length-reward.png", dpi=110); fig.savefig(f"{OUT}/length-reward.png", dpi=110); print("plot written")
except Exception as e:
    print("no plot:", e)
