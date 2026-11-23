"""Accuracy and truncation by response length and policy version, from MILES inference records (no text needed).

Usage: python length_by_version.py RUN_NAME [VERSION_BINS ...]   e.g. ... 0 20 40 60
"""
import glob
import json
import statistics as st
import sys

name = sys.argv[1]
edges = [int(e) for e in sys.argv[2:]] or [0, 20, 40, 60]
files = glob.glob(f"/weka/oe-adapt-default/jacobm/olmicro/inference-records/**/{name}-*/records-*.jsonl", recursive=True)
resp = []
for path in files:
    for line in open(path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "responses" not in row or row.get("filter_decision") == "aborted":
            continue
        for x in row["responses"]:
            versions = [int(v) for v in x.get("policy_versions") or ["0"]]
            resp.append((max(versions), int(x["response_tokens"]), float(x["reward"]) >= 1, str(x.get("truncated")) == "True"))
top = max(v for v, *_ in resp)
bins = list(zip(edges, edges[1:] + [top + 1]))
buckets = [(0, 2048), (2048, 8192), (8192, 16384), (16384, 32767), (32767, 10**9)]
for lo, hi in bins:
    seg = [r for r in resp if lo <= r[0] < hi]
    if not seg:
        continue
    print(f"versions {lo}-{hi - 1}: n={len(seg)} acc={st.mean(r[2] for r in seg):.3f} median_len={st.median(r[1] for r in seg):.0f} trunc={st.mean(r[3] for r in seg):.3f}")
    for a, b in buckets:
        s = [r for r in seg if a <= r[1] < b]
        if s:
            print(f"   len {a:>5}-{min(b, 32768):<5} share={len(s) / len(seg):.3f} acc={st.mean(r[2] for r in s):.3f} trunc={st.mean(r[3] for r in s):.3f}")
