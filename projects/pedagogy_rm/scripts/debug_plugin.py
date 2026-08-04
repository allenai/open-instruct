"""Print what the plugin actually produces, next to what the probe produced."""

from __future__ import annotations

import asyncio
import glob
import json
import statistics

import numpy as np
from projects.pedagogy_rm.agreement import load as load_labels
from projects.pedagogy_rm.plugin import PedagogyHead
from projects.pedagogy_rm.test_plugin import sample_for

with open("data/label_slices/slice_1.json") as handle:
    units = json.load(handle)["units"][:8]
by_unit = load_labels(sorted(glob.glob("data/labels/*.json")))

scorer = PedagogyHead(head="data/head.npz")
print("dims:", scorer.dims)
print("meta:", json.dumps(scorer.meta["dimensions"], indent=1)[:400])

contexts = [scorer.context(s) for s in (sample_for(u) for u in units)]
pooled = scorer.states(contexts)
for cell, vecs in pooled.items():
    arr = np.stack(vecs)
    print(f"  states {cell}: shape {arr.shape} finite={np.isfinite(arr).all()} std={arr.std():.4f}")

results = asyncio.run(scorer.score_group([sample_for(u) for u in units]))
for unit, result in zip(units, results, strict=True):
    truth = {}
    for d in scorer.dims:
        s = [r[d] for r in by_unit.get(unit["id"], {}).values() if isinstance(r.get(d), int)]
        truth[d] = round(statistics.fmean(s), 2) if s else None
    raw = {d: round(v, 2) for d, v in result.info["raw"].items()}
    print(f"  {unit['id'][:10]}  probe={raw}  labels={truth}")

# The offline path, on the same units, straight from the stored states.
blob = np.load("data/hidden.npz", allow_pickle=False)
ids = [str(x) for x in blob["ids"]]
layers = [int(x) for x in blob["layers"]]
head = np.load("data/head.npz", allow_pickle=False)
print("\noffline, same units, from data/hidden.npz:")
for unit in units:
    if unit["id"] not in ids:
        continue
    i = ids.index(unit["id"])
    row = {}
    for d in scorer.dims:
        spec = scorer.meta["dimensions"][d]
        v = blob[spec["pooling"]][i, layers.index(spec["layer"]), :].astype(np.float32)
        z = (v - head[f"{d}/mean"]) / head[f"{d}/scale"]
        row[d] = round(float(z @ head[f"{d}/coef"] + head[f"{d}/intercept"]), 2)
    print(f"  {unit['id'][:10]}  {row}")
