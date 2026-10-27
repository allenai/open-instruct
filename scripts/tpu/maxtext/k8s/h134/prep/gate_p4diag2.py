"""H134 P4 x16 diagnostic for one arm and seed (ledger-manager 2026-10-26: not a launch gate; gate_p4diag.py was B0-only).
  python3 gate_p4diag2.py <arm> <seed>
pfd-x16-<arm>-s<seed> (8x8x16, fsdp16 x ep64, factor 5.0, r2 emulation) vs pfd-ep8-<arm>-s<seed> (4x4x8 candidate, the
production mesh), both load_balance_loss_weight=0 (#620), same init and batches: LOSS-PARTS CE steps 1-6 gap against 1e-3,
tokens equal, each run's step-1 CE vs HF (refce/<arm>/refce_c11-s<seed>.json), drops per step (REQUIRED_RBF triples) and the
router-bias setting as logged. Logs in pfdiag/logs/<js>/. Exit 0 within, 1 outside, 2 incomplete. Writes
pfdiag/diag_<arm>_s<seed>.json and an events line. Limitation (AGENTS.md): two CP meshes can share an error this comparison
cannot see."""
import glob, json, re, sys, time
arm, seed = sys.argv[1], sys.argv[2]
W = "/weka/oe-eval-default/abhishekr/watch/tpu/h134"; D = f"{W}/pfdiag"


def scan(js):
    ce, tok, rbf, bias = {}, {}, {}, set()
    for f in glob.glob(f"{D}/logs/{js}/*.log"):
        t = open(f, errors="replace").read()
        for m in re.finditer(r"LOSS-PARTS step=(\d+) ce=([\d.]+) .*?tokens=(\d+)", t):
            ce.setdefault(int(m[1]), set()).add(float(m[2])); tok.setdefault(int(m[1]), set()).add(int(m[3]))
        for m in re.finditer(r"REQUIRED_RBF step=(\d+) \S+ per_layer=\[([^\]]*)\]", t):
            v = [float(x) for x in m[2].split(",")]
            if len(v) == 45:
                per = [v[i:i + 3] for i in range(0, 45, 3)]
                rbf[int(m[1])] = (max(x[0] for x in per), sum(x[1] for x in per), sum(x[2] for x in per))
        bias |= set(re.findall(r"routed_bias\W{0,4}(\w+)", t))
    return ce, tok, rbf, sorted(bias)


a, b = f"h134-pfd-ep8-{arm}-s{seed}", f"h134-pfd-x16-{arm}-s{seed}"
ce, tok, rbf, bias = scan(a); xce, xtok, xrbf, xbias = scan(b)
if not all(s in ce and s in xce and s in rbf and s in xrbf for s in range(1, 7)):
    print(f"incomplete: ep8 ce {sorted(ce)[:7]} rbf {sorted(rbf)[:7]}; x16 ce {sorted(xce)[:7]} rbf {sorted(xrbf)[:7]}"); sys.exit(2)
HF = json.load(open(f"{W}/refce/{arm}/refce_c11-s{seed}.json"))["ce"]
d = {s: max(abs(p - q) for p in ce[s] for q in xce[s]) for s in range(1, 7)}
dd = max(d.values()); teq = all(tok[s] == xtok[s] for s in range(1, 7))
drop = lambda r: max(r[s][1] / r[s][2] for s in range(1, 7) if r[s][2])
out = {"arm": arm, "seed": seed, "max_d_1_6": dd, "d_1_6": d, "tokens_equal": teq,
       "ep8_step1_vs_hf": max(abs(c - HF) for c in ce[1]), "x16_step1_vs_hf": max(abs(c - HF) for c in xce[1]), "hf": HF,
       "ce_ep8": {s: sorted(ce[s]) for s in range(1, 7)}, "ce_x16": {s: sorted(xce[s]) for s in range(1, 7)},
       "rbf_ep8": {s: rbf[s][0] for s in range(1, 7)}, "rbf_x16": {s: xrbf[s][0] for s in range(1, 7)},
       "worst_drop_frac_ep8": drop(rbf), "worst_drop_frac_x16": drop(xrbf), "routed_bias_ep8": bias, "routed_bias_x16": xbias,
       "within": dd <= 1e-3 and teq}
json.dump(out, open(f"{D}/diag_{arm}_s{seed}.json", "w"), indent=1)
msg = (f"P4DIAG {arm}-s{seed}: steps 1-6 max d {dd:.2e}, tokens equal {teq}, step-1 vs HF ep8 {out['ep8_step1_vs_hf']:.1e} "
       f"x16 {out['x16_step1_vs_hf']:.1e}, worst drop ep8 {out['worst_drop_frac_ep8']:.1e} x16 {out['worst_drop_frac_x16']:.1e}")
open(f"{W}/events.log", "a").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg + "\n")
print(msg); sys.exit(0 if out["within"] else 1)
