"""Per-host GCSFS-RETRY check for one JobSet's saved logs (run from the watcher dir, or set WATCH_DIR):
  python3 check_patched.py <jobset> [--expect N].
Exit 1 if any request waited over ALARM_S (360 s of the ~519 s patience) or ran out; exit 2 if --expect N is given and
fewer than N pods logged "GCSFS-RETRY patched", or any logged "not patched"."""
import glob, os, re, sys
ALARM_S = 360.0
js = sys.argv[1]; expect = int(sys.argv[sys.argv.index("--expect") + 1]) if "--expect" in sys.argv else None; here = os.environ.get("WATCH_DIR", os.getcwd())  # the watcher directory that holds logs/<jobset>/*.log
hosts, patched, notp = {}, set(), []
for f in glob.glob(f"{here}/logs/{js}/*.log"):
    pod = os.path.basename(f)[:-4]
    for l in open(f, errors="replace"):
        if "GCSFS-RETRY patched" in l: patched.add(pod)
        if "GCSFS-RETRY not patched" in l: notp.append(pod)
        m = re.search(r"GCSFS-RETRY retried .*attempts=(\d+)/\d+ wait=([0-9.]+)s .*ok=(\w+)", l)
        if m:
            h = hosts.setdefault(pod, [0, 0, 0.0, 0])
            h[0] += 1; h[1] = max(h[1], int(m[1])); h[2] = max(h[2], float(m[2])); h[3] += m[3] != "True"
n = len(glob.glob(f"{here}/logs/{js}/*.log"))
print(f"{js}: pods with logs {n}, patched {len(patched)}, not patched {len(notp)}, hosts that retried {len(hosts)}")
if hosts:
    print(f"  retried requests total {sum(h[0] for h in hosts.values())}, max attempts {max(h[1] for h in hosts.values())}/20, "
          f"max wait {max(h[2] for h in hosts.values()):.1f}s of ~519s, requests that ran out {sum(h[3] for h in hosts.values())}")
    for pod, h in sorted(hosts.items(), key=lambda x: -x[1][2])[:8]:
        print(f"  {pod}: requests {h[0]}, max attempts {h[1]}, max wait {h[2]:.1f}s, failed {h[3]}")
over = [p for p, h in hosts.items() if h[2] > ALARM_S or h[3]]
if over: print(f"ALARM: {len(over)} host(s) waited over {ALARM_S:.0f}s or ran out: {sorted(over)[:5]}")
if expect is not None and (len(patched) < expect or notp):
    print(f"ALARM: {len(patched)} of {expect} pods logged GCSFS-RETRY patched, {len(notp)} not patched {sorted(notp)[:3]}"); sys.exit(2)
sys.exit(1 if over else 0)
