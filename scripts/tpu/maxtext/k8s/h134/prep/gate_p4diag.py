"""H134 P4 x16 diagnostic (ledger-manager 2026-10-26: not a launch gate). pfd-x16-b0-s<seed> (8x8x16, fsdp16 x ep64, factor
5.0, r2 emulation) vs pfd-ep8-b0-s<seed> (4x4x8 candidate), both load_balance_loss_weight=0 (#620), same init and batches:
LOSS-PARTS CE steps 1-6 within 1e-3 and tokens equal; also each run's step-1 CE vs HF (refce/refce_c11-s<seed>.json).
Logs in pfdiag/logs/<js>/. Exit 0 within, 1 outside, 2 incomplete. Writes pfdiag/diag_s<seed>.json. Limitation (AGENTS.md):
two CP meshes can share an error this comparison cannot see."""
import glob, json, re, sys, time
seed = sys.argv[1]; W = "/weka/oe-eval-default/abhishekr/watch/tpu/h134"; D = f"{W}/pfdiag"
def scan(js):
    ce, tok = {}, {}
    for f in glob.glob(f"{D}/logs/{js}/*.log"):
        for m in re.finditer(r"LOSS-PARTS step=(\d+) ce=([\d.]+) .*?tokens=(\d+)", open(f, errors="replace").read()):
            ce.setdefault(int(m[1]), set()).add(float(m[2])); tok.setdefault(int(m[1]), set()).add(int(m[3]))
    return ce, tok
a, b = f"h134-pfd-ep8-b0-s{seed}", f"h134-pfd-x16-b0-s{seed}"
ce, tok = scan(a); xce, xtok = scan(b)
if not all(s in ce and s in xce for s in range(1, 7)): print("incomplete"); sys.exit(2)
HF = json.load(open(f"{W}/refce/refce_c11-s{seed}.json"))["ce"]
dd = max(abs(p - q) for s in range(1, 7) for p in ce[s] for q in xce[s]); teq = all(tok[s] == xtok[s] for s in range(1, 7))
out = {"seed": seed, "max_d_1_6": dd, "tokens_equal": teq, "ep8_step1_vs_hf": max(abs(c - HF) for c in ce[1]), "x16_step1_vs_hf": max(abs(c - HF) for c in xce[1]),
       "ce_ep8": {s: sorted(ce[s]) for s in range(1, 7)}, "ce_x16": {s: sorted(xce[s]) for s in range(1, 7)}, "within": dd <= 1e-3 and teq}
json.dump(out, open(f"{D}/diag_s{seed}.json", "w"), indent=1)
open(f"{W}/events.log", "a").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + f" H134 P4 DIAG s{seed}: steps 1-6 max d {dd:.2e}, tokens equal {teq}, x16 step-1 vs HF {out['x16_step1_vs_hf']:.1e}\n")
print(json.dumps({k: out[k] for k in ("max_d_1_6", "tokens_equal", "ep8_step1_vs_hf", "x16_step1_vs_hf", "within")})); sys.exit(0 if out["within"] else 1)
