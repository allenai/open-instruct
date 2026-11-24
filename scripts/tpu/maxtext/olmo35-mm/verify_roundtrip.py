"""Bitwise check of a MaxText -> OLMo-core conversion of a TPU multimodal SFT step.

  verify_roundtrip.py OLMO_CORE_DCP_STEP_DIR MAXTEXT_ITEMS_DIR [--out result.json]

Reads the converted OLMo-core DCP back from disk (model.* keys, fp32), converts it OLMo-core -> MaxText with the
converter's own forward path (_convert(to_maxtext=True) + scan, olmo_core_vision_to_maxtext_params), and compares
every leaf with the MaxText training checkpoint's params (fp32, restored straight from GCS) with torch.equal.
Prints ROUNDTRIP_BITWISE_OK / ROUNDTRIP_BITWISE_FAIL and exits 0 / 1. Run through export_to_olmocore.sh.
From allenai/olmo-post-training-ledger@ca639ab0b01ed7cff57a411d73936ecc9f0d8914 experiments/mmsft/tpu/verify_roundtrip.py
(ledger H271's eval pipeline); only the formatting and the import order changed.
"""

import gc
import json
import sys

import fast_maxtext_loader
import numpy as np
import torch
from maxtext.checkpoint_conversion.standalone_scripts import convert_olmo35_multimodal_ckpt as conv
from maxtext.models import olmoe3_vision
from olmo_core.nn.maxtext.checkpoint import load_olmo_core_config
from olmo_core.nn.maxtext.config import get_maxtext_config
from olmo_core.nn.maxtext.convert import _convert

# An Orbax restore of a training checkpoint crawls and gets OOM-killed on a CPU box; read the zarr arrays directly.
fast_maxtext_loader.patch()


def main():
    dcp, items = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    exp_cfg = load_olmo_core_config(dcp)
    mt_cfg = get_maxtext_config(exp_cfg["model"]["lm"])
    n_vit = int(exp_cfg["model"]["vision"].get("image_num_layers", 25))

    lm, vis = conv.load_olmo_core_checkpoint_weights(dcp, exp_cfg)
    n_oc = len(lm) + len(vis)
    n_oc_el = sum(t.numel() for t in lm.values()) + sum(t.numel() for t in vis.values())
    print(
        f"OLMo-core DCP: {len(lm)} LM + {len(vis)} vision/connector tensors, {n_oc_el / 1e9:.3f}B values", flush=True
    )
    got = _convert(mt_cfg, lm, to_maxtext=True, consume=True, round_trip=False)
    got = conv._scan_params_consume(got, mt_cfg.scan_layout, mt_cfg.n_layers)
    vis_mt = olmoe3_vision.olmo_core_vision_to_maxtext_params({k: v.numpy() for k, v in vis.items()}, num_layers=n_vit)
    for k, v in vis_mt.items():
        got[k] = torch.from_numpy(np.ascontiguousarray(v))
    del lm, vis, vis_mt
    gc.collect()

    ref = conv.load_maxtext_params_any(items)
    print(f"MaxText params: {len(ref)} leaves", flush=True)
    res = {
        "dcp": dcp,
        "items": items,
        "olmo_core_tensors": n_oc,
        "olmo_core_values": n_oc_el,
        "maxtext_leaves": len(ref),
        "missing": sorted(set(ref) - set(got)),
        "extra": sorted(set(got) - set(ref)),
        "mismatch": [],
    }
    n_ok = n_el = 0
    for k, r in ref.items():
        if k not in got:
            continue
        g = got[k]
        if tuple(g.shape) != tuple(r.shape) or g.dtype != r.dtype or not torch.equal(g, r):
            d = float((g.float().reshape(r.shape) - r.float()).abs().max()) if g.numel() == r.numel() else None
            res["mismatch"].append(
                {
                    "key": k,
                    "max_abs_diff": d,
                    "shape": [list(g.shape), list(r.shape)],
                    "dtype": [str(g.dtype), str(r.dtype)],
                }
            )
        else:
            n_ok += 1
            n_el += r.numel()
    res["bitwise_equal_leaves"] = n_ok
    res["bitwise_equal_values"] = n_el
    ok = not res["mismatch"] and not res["missing"] and not res["extra"] and n_ok == len(ref)
    res["ok"] = ok
    print(json.dumps(res, indent=1)[:4000])
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    print("ROUNDTRIP_BITWISE_OK" if ok else "ROUNDTRIP_BITWISE_FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
