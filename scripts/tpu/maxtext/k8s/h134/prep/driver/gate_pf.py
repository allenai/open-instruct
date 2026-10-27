"""H134 preflight gates for one arm and seed (pf-<arm>-c11-s<seed>, 20 steps of the production overlay; H123's gate_h123.py ep8).
  python3 gate_pf.py <arm> <seed> <HF refce json>
  P2 gate (1): step-1 CE vs HF within 1e-3 (the arm's HF step-1 reference on the same batch).
  P3: required ragged_buffer_factor (REQUIRED_RBF triples, 15 MoE layers) over steps 1-6 at or below the overlay's 2.5, and
      the dropped/routed assignments per step 1-6 (#550: no step above 1%). A factor above 2.5 is a FAIL that needs a dated
      note raising the factor and a repeat, not a production launch.
  P4 (data): BATCH-HASH n=1..20 of TPU process 4h equal H122's CPU replay watch/tpu/h122/p5/s<seed>/h<h>.npz, every host.
  Banner: the 16,304-step schedule at LR 5e-5.
Exit 0 PASS, 1 FAIL, 2 incomplete. Writes pf/gate_pf_<arm>_s<seed>.json and an events line. Counts, hashes and losses only."""
import glob, json, re, sys, time
import numpy as np
arm, seed, REF = sys.argv[1], sys.argv[2], sys.argv[3]
W = "/weka/oe-eval-default/abhishekr/watch/tpu/h134"; D = f"{W}/pf"; js = f"h134-pf-{arm}-c11-s{seed}"
REP = f"/weka/oe-eval-default/abhishekr/watch/tpu/h122/p5/s{seed}"
def ev(m):
    open(f"{W}/events.log", "a").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + m + "\n"); print(m)
ce, tok, bh, rbf, banner = {}, {}, {}, {}, ""
for f in glob.glob(f"{D}/logs/{js}/*.log"):
    t = open(f, errors="replace").read()
    for m in re.finditer(r"LOSS-PARTS step=(\d+) ce=([\d.]+) .*?tokens=(\d+)", t):
        ce.setdefault(int(m[1]), set()).add(float(m[2])); tok.setdefault(int(m[1]), set()).add(int(m[3]))
    for m in re.finditer(r"BATCH-HASH n=(\d+) host=(\d+) sha256=([0-9a-f]{64})", t):
        bh.setdefault((int(m[2]), int(m[1])), set()).add(m[3])
    for m in re.finditer(r"REQUIRED_RBF step=(\d+) \S+ per_layer=\[([^\]]*)\]", t):
        v = [float(x) for x in m[2].split(",")]
        if len(v) == 45:
            per = [v[i:i + 3] for i in range(0, 45, 3)]
            rbf[int(m[1])] = (max(x[0] for x in per), sum(x[1] for x in per), sum(x[2] for x in per))
    if "schedule, LR" in t: banner = re.search(r"[^\n]*schedule, LR[^\n]*", t)[0]
if not all(s in ce for s in range(1, 7)) or not all(s in rbf for s in range(1, 7)) or len({n for _, n in bh}) < 20:
    print(f"incomplete: ce steps {sorted(ce)[:8]}, rbf steps {sorted(rbf)[:8]}, hash steps {len({n for _, n in bh})}"); sys.exit(2)
HF = json.load(open(REF))["ce"]
bad = []
c1 = sorted(ce[1]); d1 = max(abs(c - HF) for c in c1)
if d1 > 1e-3: bad.append(f"P2 step-1 ce {c1} vs HF {HF:.6f} (d {d1:.1e})")
maxf = max(rbf[s][0] for s in range(1, 7)); worst = max(rbf[s][1] / rbf[s][2] for s in range(1, 7) if rbf[s][2])
if maxf > 2.5: bad.append(f"P3 required ragged_buffer_factor {maxf:.3f} > 2.5 (dated note + repeat needed)")
if worst > 0.01: bad.append(f"P3 dropped/routed {worst:.3e} > 1% at a step in 1-6")
nh, nbad = 0, 0
for h in range(8):
    ref = np.load(f"{REP}/h{h}.npz")["batch_sha"]
    for n in range(1, 21):
        nh += 1
        if bh.get((4 * h, n)) != {str(ref[n - 1])}:
            nbad += 1; bad.append(f"P4 hash host{h}/process{4 * h} n={n} {'missing' if (4 * h, n) not in bh else 'MISMATCH'}")
if not re.search(r"16304-step schedule, LR 5e-0?5", banner): bad.append(f"banner '{banner}'")
out = {"arm": arm, "seed": seed, "jobset": js, "hf_ref": REF, "step1_ce": c1, "hf": HF, "d1": d1, "max_required_rbf_1_6": maxf, "worst_drop_frac_1_6": worst,
       "drops_1_6": {s: rbf[s][1:] for s in range(1, 7)}, "rbf_1_6": {s: rbf[s][0] for s in range(1, 7)},
       "ce_1_6": {s: sorted(ce[s]) for s in range(1, 7)}, "tokens_1_6": {s: sorted(tok[s]) for s in range(1, 7)},
       "hashes_checked": nh, "hashes_bad": nbad, "banner": banner, "bad": bad, "pass": not bad}
json.dump(out, open(f"{D}/gate_pf_{arm}_s{seed}.json", "w"), indent=1)
ev(("H134 PF GATE PASS " if not bad else "H134 PF GATE FAIL ") + f"{js}: step1 {c1} vs HF {HF:.6f} (d {d1:.1e}); max rbf 1-6 {maxf:.3f}; "
   f"worst drop 1-6 {worst:.2e}; hashes {nh - nbad}/{nh}" + ("" if not bad else " :: " + "; ".join(bad[:5])))
sys.exit(0 if not bad else 1)
