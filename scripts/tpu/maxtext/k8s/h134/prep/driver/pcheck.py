"""H134 production check for one arm and seed (<arm>-c11-s<seed>): BATCH-HASH of TPU process 4h vs H122's full CPU replay
(watch/tpu/h122/p5/s<seed>/h<h>.npz, all 16,304 steps), step-1 CE vs the arm's HF reference (1e-3), finite losses. The #550
drop budget is xw/xw2.py's. Exit 0 ok, 10 UNSTABLE, 11 INCOMPLETE. --maxstep prints the top step.
  python3 pcheck.py <arm> <seed> <HF refce json> [--maxstep]"""
import glob, json, re, sys
import numpy as np
W = "/weka/oe-eval-default/abhishekr/watch/tpu/h134"
arm, s, REF = sys.argv[1], int(sys.argv[2]), sys.argv[3]
text = "".join(open(f, errors="replace").read() for f in glob.glob(f"{W}/prod/h134-{arm}-c11-s{s}/*.log"))
steps = [int(x) for x in re.findall(r"completed step: (\d+)", text)]
if "--maxstep" in sys.argv: print(max(steps, default=0)); sys.exit(0)
if re.search(r"(loss|lm_loss): -?(nan|inf)\b|\b(ce|moe_lb|z)=-?(nan|inf)\b", text, re.I): print(f"{arm}-s{s}: UNSTABLE non-finite loss"); sys.exit(10)
got = {}
for m in re.finditer(r"BATCH-HASH n=(\d+) host=(\d+) sha256=([0-9a-f]{64})", text): got.setdefault((int(m[2]), int(m[1])), set()).add(m[3])
bad, cmp_ = [], 0
for h in range(8):
    ref = np.load(f"/weka/oe-eval-default/abhishekr/watch/tpu/h122/p5/s{s}/h{h}.npz")["batch_sha"]
    for (p, n), v in got.items():
        if p == 4 * h and 1 <= n <= len(ref):
            cmp_ += 1
            if v != {str(ref[n - 1])}: bad.append((p, n))
ce = sorted({float(x) for x in re.findall(r"LOSS-PARTS step=1 ce=([0-9.]+)", text)})
hf = json.load(open(REF))["ce"]
h20 = sum(1 for (p, n) in got if p % 4 == 0 and p < 32 and n <= 20)
msgs = []
top = max(steps, default=0)
if top >= 2 and not ce: msgs.append("no step-1 LOSS-PARTS ce recorded")
if top >= 30 and h20 < 160: msgs.append(f"steps 1-20 hashes incomplete ({h20}/160)")
if bad: msgs.append(f"BATCH-HASH mismatch at {len(bad)} (process, step), first {sorted(bad)[:3]}")
if ce and max(abs(c - hf) for c in ce) > 1e-3: msgs.append(f"step-1 ce {ce} vs HF {hf:.6f} off by more than 1e-3")
print(f"{arm}-s{s}: step {max(steps, default=0)}, step-1 ce {ce or '-'} (HF {hf:.6f}), hashes compared {cmp_} mismatched {len(bad)} (steps 1-20 seen {h20}/160)"
      + ("; " + "; ".join(msgs) if msgs else ""))
sys.exit(11 if msgs else 0)
