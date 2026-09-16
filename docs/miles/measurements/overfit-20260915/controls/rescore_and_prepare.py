"""Re-score the Core overfit rollouts, compute per-prompt pass rates, and write control prompt files."""
import collections
import glob
import json
import os
import re
import shutil

import torch

from open_instruct.ground_truth_utils import GSM8KVerifier

ROOT = "/weka/oe-training-default/robertb/open-instruct/runs/overfit-gsm8k-core-20260915"
OUT = "/weka/oe-training-default/robertb/open-instruct/runs/overfit-controls-20260916"
CAP = 4096
verifier = GSM8KVerifier()

files = sorted(
    (f for f in glob.glob(f"{ROOT}/rollouts/*.pt") if not os.path.basename(f).startswith("eval_")),
    key=lambda f: int(re.search(r"(\d+)\.pt$", f).group(1)),
)
report = {"files": len(files), "samples": 0, "reward_mismatches": 0, "mismatch_examples": [], "truncated_but_rewarded": 0}
per_prompt = collections.defaultdict(list)
per_rollout = {}
key_seen = None
for f in files:
    d = torch.load(f, weights_only=False)
    rid = d["rollout_id"]
    samples = d["samples"]
    if key_seen is None:
        key_seen = sorted(samples[0].keys())
        report["sample_keys"] = key_seen
    rewards = []
    for s in samples:
        md = s.get("metadata") or {}
        pid = md.get("prepared_sample_id") or str(md.get("source_row"))
        label = s.get("label")
        if label is None and md.get("verifiers"):
            label = md["verifiers"][0]["target"]
        reward = s.get("reward")
        if isinstance(reward, dict):
            reward = reward.get("reward", reward.get("score"))
        reward = float(reward)
        resp = s.get("response") or ""
        rescore = verifier([], resp, str(label)).score
        report["samples"] += 1
        if abs(rescore - reward) > 1e-6:
            report["reward_mismatches"] += 1
            if len(report["mismatch_examples"]) < 5:
                report["mismatch_examples"].append({"rollout": rid, "prompt": pid, "reward": reward, "rescore": rescore, "tail": resp[-200:]})
        length = int(s.get("response_length") or 0)
        truncated = length >= CAP or "TRUNC" in str(s.get("status", "")).upper()
        if truncated and reward > 0.5:
            report["truncated_but_rewarded"] += 1
        per_prompt[pid].append((rid, reward, length, truncated))
        rewards.append(reward)
    groups = collections.defaultdict(list)
    for s, r in zip(samples, rewards):
        groups[s.get("group_index", s.get("index"))].append(r)
    per_rollout[rid] = {
        "mean_reward": sum(rewards) / len(rewards),
        "groups": len(groups),
        "mixed_groups": sum(1 for g in groups.values() if 0 < sum(g) < len(g)),
    }

def rate(items, lo, hi):
    sel = [r for rid, r, _, _ in items if lo <= rid <= hi]
    return (sum(sel) / len(sel), len(sel)) if sel else (None, 0)

prompts = {}
for pid, items in per_prompt.items():
    early, ne = rate(items, 0, 19)
    mid, nm = rate(items, 20, 39)
    late, nl = rate(items, 40, 59)
    trunc_early = sum(1 for rid, _, _, t in items if rid <= 19 and t) / max(1, ne)
    trunc_late = sum(1 for rid, _, _, t in items if rid >= 40 and t) / max(1, nl)
    prompts[pid] = {"n": len(items), "early": early, "mid": mid, "late": late, "trunc_early": trunc_early, "trunc_late": trunc_late}
report["prompts"] = dict(sorted(prompts.items(), key=lambda kv: (kv[1]["early"] if kv[1]["early"] is not None else 2)))
report["per_rollout"] = per_rollout
report["mixed_group_fraction_by_block"] = {
    f"{a}-{b}": sum(per_rollout[i]["mixed_groups"] for i in range(a, b + 1) if i in per_rollout)
    / max(1, sum(per_rollout[i]["groups"] for i in range(a, b + 1) if i in per_rollout))
    for a, b in ((0, 12), (13, 24), (25, 36), (37, 48), (49, 59))
}

# Eye-check material: for the prompts that moved most, one early-wrong and one late-right response.
examples = []
for pid, p in prompts.items():
    if p["early"] is not None and p["late"] is not None and p["late"] - p["early"] >= 0.25:
        items = per_prompt[pid]
        wrong = next((x for x in items if x[0] <= 19 and x[1] < 0.5), None)
        right = next((x for x in reversed(items) if x[0] >= 40 and x[1] > 0.5), None)
        examples.append({"prompt": pid, "early": p["early"], "late": p["late"], "early_wrong": wrong, "late_right": right})
report["moved_prompts"] = examples

# Write control prompt files.
prep = f"{ROOT}/prepared/data"
os.makedirs(f"{OUT}/signflip", exist_ok=True)
os.makedirs(f"{OUT}/hard", exist_ok=True)
rows = [json.loads(l) for l in open(f"{prep}/train.jsonl")]
with open(f"{OUT}/signflip/train.jsonl", "w") as fh:
    for r in rows:
        r = json.loads(json.dumps(r))
        for v in r["metadata"]["verifiers"]:
            v["weight"] = -1.0
        fh.write(json.dumps(r) + "\n")
hard_ids = [pid for pid, p in prompts.items() if p["early"] is not None and 0.0 < p["early"] <= 0.75]
with open(f"{OUT}/hard/train.jsonl", "w") as fh:
    for r in rows:
        if r["metadata"]["prepared_sample_id"] in hard_ids:
            fh.write(json.dumps(r) + "\n")
report["hard_ids"] = hard_ids
for sub in ("signflip", "hard"):
    shutil.copy(f"{prep}/eval.jsonl", f"{OUT}/{sub}/eval.jsonl")
    shutil.copy(f"{prep}/verifiers.json", f"{OUT}/{sub}/verifiers.json")
os.makedirs("/output", exist_ok=True)
for path in ("/output/report.json", f"{OUT}/report.json"):
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
print(json.dumps({k: report[k] for k in ("files", "samples", "reward_mismatches", "truncated_but_rewarded", "mixed_group_fraction_by_block", "hard_ids", "sample_keys")}, indent=1))
for pid, p in report["prompts"].items():
    print(pid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()})
