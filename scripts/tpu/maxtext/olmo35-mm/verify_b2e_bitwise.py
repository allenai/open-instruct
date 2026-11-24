"""Bitwise check: converted MaxText items (mapped back to OLMo-core keys) == B2e DCP fp32 weights + think seeding.

Copied from the session that checked the reference run's start checkpoint (BITWISE_OK on all 829 tensors, 12.92B
values, 2026-11-20); the reverse mapping is the pinned converter's (convert_b2e_to_maxtext.py). Run through
convert_b2e.sh.

Usage: python verify_bitwise.py DCP_STEP_DIR MAXTEXT_ITEMS_DIR [--out result.json]
The reverse mapping (unscan + _convert(to_maxtext=False) + maxtext_vision_to_olmo_core_params) is the converter's own
maxtext_to_olmo_core path; the reference is load_olmo_core_checkpoint_weights on the DCP (fp32 masters) with the
seeding applied independently here.
"""

import gc
import json
import sys

import torch
from maxtext.checkpoint_conversion.standalone_scripts import convert_olmo35_multimodal_ckpt as conv
from maxtext.models import olmoe3_vision
from olmo_core.nn.maxtext.checkpoint import is_scanned, load_olmo_core_config, unscan_params
from olmo_core.nn.maxtext.config import get_maxtext_config
from olmo_core.nn.maxtext.convert import _convert
from olmo_core.train.callbacks.promoted_rows import OLMO35_THINK_ROWS


def main():
    dcp, items = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    exp_cfg = load_olmo_core_config(dcp)
    mt_cfg = get_maxtext_config(exp_cfg["model"]["lm"])
    n_vit = int(exp_cfg["model"]["vision"].get("image_num_layers", 25))

    ref_lm, ref_vis = conv.load_olmo_core_checkpoint_weights(dcp, exp_cfg)
    # Independent seeding (same formula as MultimodalTrainModule.seed_token_rows_from_pieces).
    for name in ("embeddings.weight", "lm_head.w_out.weight"):
        w = ref_lm[name]
        means = {t: w[list(s)].to(torch.float32).mean(dim=0) for t, s in OLMO35_THINK_ROWS}
        for t, m in means.items():
            w[t] = m
    print(f"reference: {len(ref_lm)} LM + {len(ref_vis)} vision/connector tensors", flush=True)

    mt_all = conv.load_maxtext_params_any(items)
    print(f"maxtext leaves: {len(mt_all)}", flush=True)
    leaf_shapes = {k: list(v.shape) for k, v in mt_all.items()}
    mt_vis = {k: mt_all.pop(k) for k in list(mt_all) if k.startswith("vision_encoder/")}
    scanned = is_scanned(mt_all, mt_cfg.scan_layout)
    mt_lm = unscan_params(mt_all, mt_cfg.scan_layout) if scanned else mt_all
    got_lm = _convert(mt_cfg, mt_lm, to_maxtext=False, consume=True, round_trip=False)
    got_vis = olmoe3_vision.maxtext_vision_to_olmo_core_params(
        {k: v.numpy() for k, v in mt_vis.items()}, num_layers=n_vit
    )
    gc.collect()

    res = {"scanned": scanned, "maxtext_leaves": len(leaf_shapes), "mismatch": [], "missing": [], "extra": []}
    n_ok = 0
    n_el = 0
    for ref, got, tag in ((ref_lm, got_lm, "lm."), (ref_vis, got_vis, "")):
        res["missing"] += [tag + k for k in ref if k not in got]
        res["extra"] += [tag + k for k in got if k not in ref]
        for k, r in ref.items():
            if k not in got:
                continue
            g = got[k]
            g = g if isinstance(g, torch.Tensor) else torch.from_numpy(g)
            if tuple(g.shape) != tuple(r.shape) or g.dtype != torch.float32 or not torch.equal(g, r):
                d = float((g.float().reshape(r.shape) - r).abs().max()) if g.numel() == r.numel() else None
                res["mismatch"].append({"key": tag + k, "max_abs_diff": d, "shape": [list(g.shape), list(r.shape)]})
            else:
                n_ok += 1
                n_el += r.numel()
    res["bitwise_equal_tensors"] = n_ok
    res["bitwise_equal_elements"] = n_el
    # Seeded rows as stored in MaxText layout.
    emb = got_lm["embeddings.weight"]
    head = got_lm["lm_head.w_out.weight"]
    res["seeded_rows_check"] = {
        str(t): {
            "emb_equals_mean": bool(torch.equal(emb[t], ref_lm["embeddings.weight"][t])),
            "head_equals_mean": bool(torch.equal(head[t], ref_lm["lm_head.w_out.weight"][t])),
        }
        for t, _ in OLMO35_THINK_ROWS
    }
    res["leaf_shapes_sample"] = dict(list(sorted(leaf_shapes.items()))[:400])
    print(json.dumps({k: v for k, v in res.items() if k != "leaf_shapes_sample"}, indent=1))
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    ok = not res["mismatch"] and not res["missing"] and not res["extra"]
    print("BITWISE_OK" if ok else "BITWISE_FAIL")


if __name__ == "__main__":
    main()
