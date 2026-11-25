"""Collect the Olmicro RL-zero results from run roots, receipts and eval summaries into CSVs for the results branch.

Usage: python gather_results.py OUT_DIR
"""

import csv
import glob
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

O = Path("/weka/oe-adapt-default/jacobm/olmicro")
RECEIPTS = Path("/root/.cache/open-instruct/miles/launches")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

# (run name, size, base checkpoint, description, data description)
RUNS = [
    ("olmicro-2.7b-math-rl-v0-r1-20261118", "2.7B", "midtrain", "v0: barrier, 4k cap, unfiltered", "dolci-math-base-r1-v1 (13,251)"),
    ("olmicro-2.7b-math-rl-v1a-full-20261118", "2.7B", "midtrain", "v1a: barrier, 7,680 cap, unfiltered (stopped)", "dolci-math-base-r1-v1 (13,251)"),
    ("olmicro-2.7b-math-rl-v1b-passband-20261118", "2.7B", "midtrain", "v1b: barrier, 7,680 cap, base pass band (stopped)", "passband-base16 (4,029)"),
    ("olmicro-2.7b-lc-math-rl-v2-refresh-32k-20261118", "2.7B", "long-context", "32k cap, own band, 4+4 (u0-u99; u100 eval timed out)", "passband-lc2.7b-32k (4,900)"),
    ("olmicro-2.7b-lc-math-rl-v2-refresh-32k-4x12-20261118", "2.7B", "long-context", "32k cap, own band, 4+12 resume (u100-u399)", "passband-lc2.7b-32k (4,900)"),
    ("olmicro-2.7b-lc-math-rl-v2-refresh-8k-20261118", "2.7B", "long-context", "7,680 cap, own band (ablation)", "passband-lc2.7b-8k (4,353)"),
    ("olmicro-2.7b-lc-math-rl-v2-refresh-16k-20261118", "2.7B", "long-context", "16k cap (canceled at u4 to free a node)", "passband-lc2.7b-16k (4,845)"),
    ("olmicro-1.4b-lc-math-rl-refresh-32k-20261118", "1.4B", "long-context", "32k cap, own band", "passband-lc1.4b-32k (2,281)"),
    ("olmicro-1.4b-lc-math-rl-refresh-32k-combined-20261119", "1.4B", "long-context", "32k cap, own band + 2.7B band", "combined-lc1.4b-32k (4,969)"),
    ("olmicro-810m-lc-math-rl-refresh-32k-20261119", "810M", "long-context", "32k cap, own band", "passband-lc810m-32k-full (2,043)"),
    ("olmicro-810m-lc-math-rl-refresh-32k-combined-20261119", "810M", "long-context", "32k cap, own band + 2.7B band", "combined-lc810m-32k (5,416)"),
    ("olmicro-810m-lc-math-rl-refresh-32k-combined-zstd-20261124", "810M", "long-context", "as combined + filter_zero_std_groups", "combined-lc810m-32k (5,416)"),
    ("olmicro-450m-lc-math-rl-refresh-32k-combined-20261119", "450M", "long-context", "32k cap, own band + 2.7B band", "combined-lc450m-32k (5,283)"),
    ("olmicro-450m-lc-math-rl-refresh-32k-combined-zstd-20261120", "450M", "long-context", "as combined + filter_zero_std_groups", "combined-lc450m-32k (5,283)"),
    ("olmicro-275m-lc-math-rl-refresh-32k-combined-20261119", "275M", "long-context", "32k cap, own band + 2.7B band", "combined-lc275m-32k (5,209)"),
    ("olmicro-275m-lc-math-rl-refresh-32k-combined-zstd-20261120", "275M", "long-context", "as combined + filter_zero_std_groups", "combined-lc275m-32k (5,209)"),
]


def experiment_of(name):
    files = sorted(RECEIPTS.glob(f"{name}-*.json"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if re.fullmatch(rf"{re.escape(name)}-[0-9a-f]{{16}}\.json", f.name)] or files
    return json.loads(files[-1].read_text())["experiment_id"] if files else ""


def wandb_evals(root):
    best = []
    for f in glob.glob(f"{root}/wandb/wandb/run-*/files/output_*.log"):
        text = open(f, errors="ignore").read()
        ev = re.findall(r"'eval/(math500|gsm8k_platinum)/episode_raw_reward': ([-0-9.e]+)", text)
        if len(ev) > len(best):
            best = ev
    math = [round(100 * float(v), 2) for n, v in best if n == "math500"]
    gsm = [round(100 * float(v), 2) for n, v in best if n == "gsm8k_platinum"]
    return math, gsm


def timing(root):
    path = Path(root) / "checkpoints/driver_timing.jsonl"
    if not path.is_file():
        return [], None, None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    evals = [r["rollout_id"] for r in rows if r["stage"] == "evaluation" and r.get("passed")]
    span = (rows[-1]["started_unix"] + rows[-1]["seconds"] - rows[0]["started_unix"]) / 3600
    updates = sum(r["stage"] == "training" for r in rows)
    return evals, span, updates


def spec_bits(root):
    p = Path(root) / "run-spec.json"
    if not p.is_file():
        return {}
    s = json.loads(p.read_text())
    tr, inf = s.get("trainer", {}), s.get("inference", {})
    miles, core, training = s.get("miles", {}), s.get("core", {}), s.get("training", {})
    return {
        "trainer_gpus": tr.get("gpus"), "inference_gpus": inf.get("gpus"), "max_response_length": inf.get("max_response_length"),
        "publication_mode": core.get("publication_mode"), "filter_zero_std_groups": training.get("filter_zero_std_groups"),
        "num_rollouts": training.get("num_rollouts"), "in_flight": miles.get("async_max_concurrent_samples"),
    }


inloop, inventory = [], []
for name, size, base, desc, data in RUNS:
    root = O / "runs" / name
    evals, span, updates = timing(root)
    math, gsm = wandb_evals(root)
    status = json.loads((root / "workflow.json").read_text()).get("status") if (root / "workflow.json").is_file() else ""
    if len(evals) != len(math):
        print(f"WARN {name}: {len(evals)} eval stages vs {len(math)} wandb evals; using wandb order with 50-update spacing")
        steps = [0] + [50 * i for i in range(1, len(math))]
        if name.endswith("4x12-20261118"):
            steps = [150 + 50 * i for i in range(len(math))]
    else:
        steps = [0 if r == 0 else r + 1 for r in evals]
    for st, m, g in zip(steps, math, gsm):
        inloop.append({"run": name, "size": size, "update": st, "math500": m, "gsm8k_platinum": g, "source": "in-loop"})
    bits = spec_bits(root)
    inventory.append({
        "run": name, "size": size, "base_checkpoint": base, "description": desc, "train_data": data, **bits,
        "updates_trained": updates, "wall_hours": round(span, 2) if span else "", "status": status,
        "math500_first": math[0] if math else "", "math500_final": math[-1] if math else "", "math500_peak": max(math) if math else "",
        "gsm8k_first": gsm[0] if gsm else "", "gsm8k_final": gsm[-1] if gsm else "",
        "experiment": f"https://beaker.org/ex/{experiment_of(name)}" if experiment_of(name) else "", "run_root": str(root),
    })
# The 32k run's update-100 eval was re-run offline from checkpoint 99 (same protocol).
inloop.append({"run": "olmicro-2.7b-lc-math-rl-v2-refresh-32k-20261118", "size": "2.7B", "update": 100, "math500": 70.9, "gsm8k_platinum": 88.3,
               "source": "offline (u100 export + eval-only run, same protocol)"})
inloop.sort(key=lambda r: (r["run"].replace("-4x12", ""), r["update"]))

with (OUT / "in_loop_evals.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(inloop[0])); w.writeheader(); w.writerows(inloop)
with (OUT / "runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(inventory[0])); w.writeheader(); w.writerows(inventory)

# Minerva MATH + AIME 2026 offline evals (olmo-eval grader "oe", MathVerifier "mv").
rows = []
for f in sorted(glob.glob(str(O / "monitor/extra-evals-*-20261118/summary.json"))):
    tag = Path(f).parent.name.removeprefix("extra-evals-").removesuffix("-20261118")
    s = json.loads(Path(f).read_text()); m, a = s["minerva_math"], s["aime2026"]
    rows.append({"checkpoint": tag, "minerva_macro_oe": m["macro_oe"], "minerva_micro_oe": m["acc_oe"], "minerva_macro_mv": m["macro_mv"], "minerva_micro_mv": m["acc_mv"],
                 **{f"minerva_{k}": v for k, v in m.get("subjects_oe", {}).items()},
                 "aime_pass1_oe": a["pass@1_oe"], "aime_pass4_oe": a.get("pass@4_oe"), "aime_pass8_oe": a.get("pass@8_oe"), "aime_pass16_oe": a.get("pass@16_oe"),
                 "aime_pass32_oe": a.get("pass@32_oe"), "aime_pass1_oe_16k": a["pass@1_oe_16k"], "aime_pass1_mv": a["pass@1_mv"],
                 "minerva_mean_tokens": m["mean_tokens"], "aime_mean_tokens": a["mean_tokens"], "minerva_truncated_pct": m["truncated_pct"], "aime_truncated_pct": a["truncated_pct"]})
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with (OUT / "minerva_aime_evals.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def group_records(name):
    """Completed (non-aborted) group records of a run, from its MILES inference records."""
    for path in glob.glob(str(O / f"inference-records/**/{name}-*/records-*.jsonl"), recursive=True):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "responses" in r and r.get("filter_decision") != "aborted":
                yield r


def version(r):
    return max(int(p) for x in r["responses"] for p in (x.get("policy_versions") or ["0"]))


# Signal fraction (groups with mixed rewards) from inference records, by policy-version band.
sig = []
for name, size, *_ in RUNS:
    bins = {}
    for r in group_records(name):
        rs = [float(x["reward"]) for x in r["responses"]]
        v = version(r)
        b = "0-49" if v < 50 else ("50-199" if v < 200 else ("200-349" if v < 350 else "350-399"))
        c = sum(x >= 1 for x in rs)
        k = "signal" if 0 < c < len(rs) else ("all_wrong" if c == 0 else "all_right")
        bins.setdefault(b, {"signal": 0, "all_wrong": 0, "all_right": 0})[k] += 1
    for b, d in sorted(bins.items(), key=lambda kv: int(kv[0].split("-")[0])):
        n = sum(d.values())
        sig.append({"run": name, "size": size, "policy_versions": b, "groups": n, **{f"{k}_pct": round(100 * v / n, 1) for k, v in d.items()}})
with (OUT / "signal_fraction.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sig[0])); w.writeheader(); w.writerows(sig)

# Response length per 50-update policy-version band from inference records. "all" is every generated group;
# "trained" (zero-std runs only) drops the groups filter_zero_std_groups removed before training. Groups finished under
# the post-update-399 weights were drained at shutdown and never trained on (and skew long), so they are left out.
lens = []
for name, size, *_ in RUNS:
    bins, any_filtered = {}, False
    for r in group_records(name):
        v = version(r)
        if v >= 400:
            continue
        d = bins.setdefault(50 * (v // 50), {"all": [], "trained": []})
        rows = [(int(x["response_tokens"]), str(x.get("truncated")) == "True", float(x["reward"]) >= 1) for x in r["responses"]]
        d["all"] += rows
        if r.get("filter_decision") == "filtered":
            any_filtered = True
        else:
            d["trained"] += rows
    for b, d in sorted(bins.items()):
        for subset in ["all", "trained"] if any_filtered else ["all"]:
            rows = d[subset]
            if not rows:
                continue
            t = sorted(n for n, _, _ in rows)
            lens.append({"run": name, "size": size, "policy_versions": f"{b}-{b + 49}", "subset": subset, "responses": len(rows),
                         "mean_tokens": round(statistics.mean(t)), "median_tokens": round(statistics.median(t)), "p90_tokens": t[int(0.9 * (len(t) - 1))],
                         "truncated_pct": round(100 * statistics.mean(tr for _, tr, _ in rows), 2), "correct_pct": round(100 * statistics.mean(c for _, _, c in rows), 1)})
with (OUT / "response_lengths.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(lens[0])); w.writeheader(); w.writerows(lens)

# Generation throughput over updates 100-399 (runs that stop before update 200, such as the 4+4 2.7B 32k run, from their
# first version + 25). t(v) is when the first group carrying policy version v was recorded. End-to-end rates divide all
# generation in [t(lo), t(hi)) (zero-std-filtered groups included) by that wall time, so they include in-loop evals and
# weight syncs; median_s_per_update (the median t(v+1) - t(v)) leaves the eval pauses out. Decode rates are SGLang's
# per-engine "gen throughput" log lines in the same window (TP1, so per GPU), with the requests that engine was decoding.
DECODE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([\d.]+)")
gpus = {r["run"]: r["inference_gpus"] for r in inventory}
tput = []
for name, size, *_ in RUNS:
    recs = sorted((r["recorded_unix"], version(r), len(r["responses"]), sum(int(x["response_tokens"]) for x in r["responses"]),
                   r.get("filter_decision") != "filtered") for r in group_records(name))
    if not recs:
        continue
    top = max(v for _, v, *_ in recs if v < 400)
    lo, hi = (100, top + 1) if top >= 200 else (min(v for _, v, *_ in recs) + 25, top + 1)
    if hi - lo < 50:
        continue
    first, m = {}, -1
    for t, v, *_ in recs:
        while m < v:
            m += 1
            first[m] = t
    t_lo, t_hi = first[lo], first.get(hi, recs[-1][0])
    win = [r for r in recs if t_lo <= r[0] < t_hi]
    sec = t_hi - t_lo
    dec = []
    for path in glob.glob(str(O / f"runs/{name}/wandb/wandb/run-*/files/output_*.log")):
        for line in open(path, errors="replace"):
            if "gen throughput" in line and (d := DECODE.search(line)):
                ts = datetime.strptime(d[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                if t_lo <= ts < t_hi:
                    dec.append((int(d[2]), float(d[3])))
    tput.append({
        "run": name, "size": size, "inference_gpus": gpus[name], "updates": f"{lo}-{hi - 1}", "wall_hours": round(sec / 3600, 2),
        "mean_s_per_update": round(sec / (hi - lo), 1), "median_s_per_update": round(statistics.median(first[v + 1] - first[v] for v in range(lo, hi - 1)), 1),
        "responses_per_s": round(sum(n for _, _, n, _, _ in win) / sec, 2),
        "trained_responses_per_s": round(sum(n for _, _, n, _, k in win if k) / sec, 2),
        "gen_tokens_per_s": round(sum(tok for _, _, _, tok, _ in win) / sec),
        "gen_tokens_per_s_per_gpu": round(sum(tok for _, _, _, tok, _ in win) / sec / gpus[name]),
        "decode_tokens_per_s_per_gpu_median": round(statistics.median(x for _, x in dec)) if dec else "",
        "decode_running_requests_median": statistics.median(n for n, _ in dec) if dec else "",
    })
with (OUT / "throughput.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tput[0])); w.writeheader(); w.writerows(tput)

# Difficulty-filtering summary per size from dataset manifests.
def kept(d):
    p = O / "data" / d / "manifest.json"
    if not p.is_file():
        return ""
    m = json.loads(p.read_text())
    return m.get("kept", m.get("rows", ""))
filt = []
for size, tag in [("2.7B", "2.7b"), ("1.4B", "1.4b"), ("810M", "810m"), ("450M", "450m"), ("275M", "275m")]:
    filt.append({
        "size": size,
        "band_32k_first_pass": kept(f"dolci-math-base-r1-v1-passband-lc{tag}-32k"),
        "band_16k_first_pass": kept(f"dolci-math-base-r1-v1-passband-lc{tag}-16k"),
        "band_8k": kept(f"dolci-math-base-r1-v1-passband-lc{tag}-8k"),
        "band_32k_all_13251": kept(f"dolci-math-base-r1-v1-passband-lc{tag}-32k-full") or (kept(f"dolci-math-base-r1-v1-passband-lc{tag}-32k") if tag == "2.7b" else ""),
        "combined_with_2.7b_band": kept(f"dolci-math-base-r1-v1-combined-lc{tag}-32k"),
    })
with (OUT / "difficulty_filtering.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filt[0])); w.writeheader(); w.writerows(filt)
print("wrote", [p.name for p in OUT.iterdir()])
