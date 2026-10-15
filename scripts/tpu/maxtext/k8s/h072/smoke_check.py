"""H072 Amendment 1(b): a production run's first 20 steps as the TPU smoke.
  python smoke_check.py <jobset> <stream>      e.g. h072-e2-s0 E2-s0
PASS iff (1) every data host's BATCH-HASH n=1..20 (TPU process 4i) equals the CPU rebuild's
(experiments/h072/cpu_batch_hashes_steps1-20.json), and (2) CE at each step 1-20 is finite and within
[0.9 x min, 1.1 x max] of A0' s0/s1 at that step."""
import glob, json, math, os, re, sys
js, stream = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
cpu = json.load(open(os.path.expanduser("~/repos/ledger-rep/experiments/h072/cpu_batch_hashes_steps1-20.json")))["streams"][stream]
tpu, ce = {}, {}
for f in glob.glob(f"{here}/logs/{js}/*.log"):
    t = open(f).read()
    for m in re.finditer(r"BATCH-HASH n=(\d+) host=(\d+) sha256=([0-9a-f]+)", t):
        tpu.setdefault(int(m[2]), {})[int(m[1])] = m[3]
    for m in re.finditer(r"LOSS-PARTS step=(\d+) ce=([-\w.]+)", t):
        ce[int(m[1])] = float(m[2])
a0 = {}
for s in ("s0", "s1"):
    for line in open(os.path.expanduser(f"~/tpu-runs/h067-early/a0prime-{s}-loss-parts-1-20.txt")):
        m = re.search(r"step=(\d+) ce=([\d.]+)", line)
        a0.setdefault(int(m[1]), []).append(float(m[2]))
bad = []
for key, shas in cpu.items():
    proc = int(key.split("tpu_process")[1])
    for n, sha in enumerate(shas, start=1):
        got = tpu.get(proc, {}).get(n)
        if got != sha:
            bad.append(f"hash host{proc} n={n}: {'missing' if got is None else 'MISMATCH'}")
for n in range(1, 21):
    lo, hi = 0.9 * min(a0[n]), 1.1 * max(a0[n])
    x = ce.get(n)
    if x is None or not math.isfinite(x) or not lo <= x <= hi:
        bad.append(f"ce step {n}: {x} not in [{lo:.4f}, {hi:.4f}]")
print(f"{js}: hashes checked {sum(len(v) for v in cpu.values())}, ce steps {len([n for n in ce if n <= 20])}; "
      + ("PASS" if not bad else f"FAIL {len(bad)}: " + "; ".join(bad[:6])))
sys.exit(1 if bad else 0)
