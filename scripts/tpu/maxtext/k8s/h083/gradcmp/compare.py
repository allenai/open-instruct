"""H083 1-step gradient comparison: old image vs the big-slice image (ledger-manager ruling 2026-10-23).

Both runs take one plain-SGD step (no clipping, LR 1e8) from the same fp32 weights p0 on the same batch, so
p1 = p0 - lr1 * g and, per tensor and over the whole model,
    |g_bs - g_old| / |g_old| = |p1_bs - p1_old| / |p0 - p1_old|.
The read-out floor is fp32 rounding of p1: about 2**-24 * |p1_old| / |p0 - p1_old|, reported beside each value.
Pass: whole-model value <= 1e-5 (class B, reduction order). Prints JSON to stdout.

  python compare.py <p0 items dir> <p1_old items dir> <p1_bs items dir> [max tensors, for a smoke test]
  (gs:// paths with an orbax _METADATA)
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorstore as ts

CTX = ts.Context({"cache_pool": {"total_bytes_limit": 0}})
DROP = {"params", "value", "raw_value"}


def keys(base):
    bucket, path = base[5:].split("/", 1)
    kv = ts.KvStore.open({"driver": "gcs", "bucket": bucket, "path": path.rstrip("/") + "/"}).result()
    md = json.loads(kv.read("_METADATA").result().value)
    out = {}
    for v in md["tree_metadata"].values():
        k = tuple(x["key"] for x in v["key_metadata"])
        out[tuple(x for x in k if x not in DROP)] = k
    return out


def read(base, k):
    name = ".".join(k)
    err = None
    for drv in ("zarr3", "zarr"):
        try:
            spec = {"driver": drv, "kvstore": {"driver": "ocdbt", "base": base.rstrip("/") + "/", "path": name}}
            return ts.open(spec, context=CTX, open=True).result().read().result()
        except Exception as e:  # noqa: BLE001 - try the other zarr driver
            err = e
    raise err


def main():
    p0, old, bs = sys.argv[1:4]
    k0, ko, kb = keys(p0), keys(old), keys(bs)
    # The step-1 checkpoints also hold non-parameter state (rng counters and keys); compare the weights p0 has.
    assert set(ko) == set(kb) and set(k0) <= set(ko), (len(k0), len(ko), len(kb), sorted(set(k0) - set(ko))[:5])
    extra = sorted("/".join(k) for k in set(ko) - set(k0))

    def one(n):
        a0, ao, ab = (read(p, kk[n]).astype(np.float64) for p, kk in ((p0, k0), (old, ko), (bs, kb)))
        assert a0.shape == ao.shape == ab.shape, n
        return n, {"numel": int(a0.size), "g2": float(np.sum((a0 - ao) ** 2)), "d2": float(np.sum((ab - ao) ** 2)),
                   "p2": float(np.sum(ao**2)), "max_abs_d": float(np.max(np.abs(ab - ao))) if a0.size else 0.0,
                   "bit_identical": bool(np.array_equal(ab, ao))}

    with ThreadPoolExecutor(16) as ex:
        per = dict(ex.map(one, sorted(k0)[: int(sys.argv[4])] if len(sys.argv) > 4 else sorted(k0)))
    g2 = sum(v["g2"] for v in per.values())
    d2 = sum(v["d2"] for v in per.values())
    p2 = sum(v["p2"] for v in per.values())
    rows = []
    for n, v in per.items():
        rel = (v["d2"] ** 0.5) / (v["g2"] ** 0.5) if v["g2"] > 0 else (0.0 if v["d2"] == 0 else float("inf"))
        floor = 2**-24 * (v["p2"] ** 0.5) / (v["g2"] ** 0.5) if v["g2"] > 0 else float("inf")
        rows.append({"tensor": "/".join(n), "rel": rel, "floor": floor, **v})
    rows.sort(key=lambda r: -r["rel"])
    rel = (d2**0.5) / (g2**0.5) if g2 > 0 else float("inf")
    print(json.dumps({
        "rel": rel, "floor": 2**-24 * p2**0.5 / g2**0.5 if g2 > 0 else float("inf"), "pass": rel <= 1e-5,
        "tensors": len(rows), "params": sum(r["numel"] for r in rows),
        "bit_identical_tensors": sum(r["bit_identical"] for r in rows),
        "non_param_keys_skipped": len(extra), "non_param_examples": extra[:5],
        "worst": rows[:10], "zero_grad_tensors": [r["tensor"] for r in rows if r["g2"] == 0],
    }, indent=1))


if __name__ == "__main__":
    main()
