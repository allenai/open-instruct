"""Collect the Olmicro RL-zero results from run roots, receipts and eval summaries into CSVs for the results branch.

Usage: python gather_results.py OUT_DIR
"""

import csv
import glob
import json
import re
import statistics
import sys
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

# Signal fraction (groups with mixed rewards) from inference records, by policy-version band.
sig = []
for name, size, *_ in RUNS:
    files = glob.glob(str(O / f"inference-records/**/{name}-*/records-*.jsonl"), recursive=True)
    if not files:
        continue
    bins = {}
    for path in files:
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "responses" not in r or r.get("filter_decision") == "aborted":
                continue
            rs = [float(x["reward"]) for x in r["responses"]]
            v = max(int(p) for x in r["responses"] for p in (x.get("policy_versions") or ["0"]))
            b = "0-49" if v < 50 else ("50-199" if v < 200 else ("200-349" if v < 350 else "350-399"))
            c = sum(x >= 1 for x in rs)
            k = "signal" if 0 < c < len(rs) else ("all_wrong" if c == 0 else "all_right")
            bins.setdefault(b, {"signal": 0, "all_wrong": 0, "all_right": 0})[k] += 1
    for b, d in sorted(bins.items(), key=lambda kv: int(kv[0].split("-")[0])):
        n = sum(d.values())
        sig.append({"run": name, "size": size, "policy_versions": b, "groups": n, **{f"{k}_pct": round(100 * v / n, 1) for k, v in d.items()}})
with (OUT / "signal_fraction.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sig[0])); w.writeheader(); w.writerows(sig)

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
