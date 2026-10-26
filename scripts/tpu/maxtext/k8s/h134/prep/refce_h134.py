"""H134 P2 (gate 1) HF step-1 CE: HF forward of a C11 seed's step-1 global batch (16 rows x 65,536: the 2 trained rows of
each of the 8 data hosts, H122's CPU replay, watch/tpu/h122/p5/s<seed>/h{0..7}.step1.npz) through the 14T base exported
to HF from the converted MaxText params the TPU loads. Loader as H122's refce_h122.py; model load and method as
x16-refce/refce4b1.py (H103/H123's hero-small refs): olmo3moe remote code, each packed segment as its own sequence,
bf16 weights, fp32 logits and CE over targets_segmentation > 0, the token-weighted mean LOSS-PARTS ce reports.
Prints counts and losses only.
  python refce_h134.py <dir with h{0..7}.step1.npz> <hf dir> <out.json>"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM

D, HF, OUT = sys.argv[1:4]
rows = []
for p in range(8):
    b = np.load(f"{D}/h{p}.step1.npz")
    keys = sorted(b.files)
    for i in range(b["inputs"].shape[0]):
        rows.append({k: b[k][i] for k in keys})
print("KEYS", keys, "rows", len(rows), flush=True)
ntok = int(sum((r["targets_segmentation"] > 0).sum() for r in rows))
print("LOSS-TOKENS", ntok, flush=True)
m = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
W = m.lm_head.weight
tot, per_row, t0 = 0.0, [], time.time()
with torch.no_grad():
    for ri, r in enumerate(rows):
        seg, tseg = r["inputs_segmentation"], r["targets_segmentation"]
        rs, rn = 0.0, 0
        for g in sorted(set(seg[seg > 0].tolist())):
            idx = np.nonzero(seg == g)[0]
            assert (np.diff(idx) == 1).all()
            pos = r["inputs_position"][idx]
            assert pos[0] == 0 and (np.diff(pos) == 1).all(), "positions do not reset per segment"
            x = torch.tensor(r["inputs"][idx].astype(np.int64), device="cuda")[None]
            h = m.model(input_ids=x).last_hidden_state[0]
            tgt = torch.tensor(r["targets"][idx].astype(np.int64), device="cuda")
            msk = torch.tensor(tseg[idx] > 0, device="cuda")
            for s in range(0, h.shape[0], 4096):
                lg = (h[s:s + 4096] @ W.T).float()
                nll = torch.nn.functional.cross_entropy(lg, tgt[s:s + 4096], reduction="none")
                rs += float((nll * msk[s:s + 4096]).sum()); rn += int(msk[s:s + 4096].sum())
        tot += rs
        per_row.append({"row": ri, "loss_tokens": rn, "ce": rs / max(rn, 1)})
        print("ROW", ri, rn, round(rs / max(rn, 1), 6), f"{time.time() - t0:.0f}s", flush=True)
res = {"loss_tokens": ntok, "ce": tot / ntok, "rows": per_row, "hf": HF, "batch": D}
json.dump(res, open(OUT, "w"), indent=1)
print("REFCE", json.dumps({"loss_tokens": ntok, "ce": round(tot / ntok, 6)}), flush=True)
