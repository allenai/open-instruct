"""HF step-1 CE of a GPU (OLMo-core) SFT step-1 global batch: ledger H161 GPU companion, gate (1) of the placement
rule, by H134 P2's method (refce_h134.py): olmo3moe remote code, bf16 weights, each document of a packed row forwarded
as its own sequence, fp32 logits and CE, token-weighted over the loss tokens of all rows.

  python h161_refce.py <dir of rankNN.npz from h161_smoke_wrap.py> <hf dir> <out.json>

Labels as OLMo-core builds them: the target of position i is input_ids[i+1] where label_mask[i+1], else ignored.
Documents come from doc_lens (the metadata boundaries the trainer used). A loss token whose target is the first token
of the next document (a label across a boundary) is counted and reported separately; it should not occur, since a
document's first token is a masked prompt token. Prints counts and losses only.
"""

import glob
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

D, HF, OUT = sys.argv[1:4]
paths = sorted(glob.glob(os.path.join(D, "rank*.npz")))
rows = []
for p in paths:
    b = np.load(p)
    for i in range(b["input_ids"].shape[0]):
        dl = b["doc_lens"][i]
        rows.append({"src": os.path.basename(p), "ids": b["input_ids"][i], "lm": b["label_mask"][i].astype(bool),
                     "doc_lens": dl[dl > 0]})
print("ROWS", len(rows), "from", len(paths), "files", flush=True)
ntok, cross = 0, 0
for r in rows:
    assert int(r["doc_lens"].sum()) == len(r["ids"]), (r["src"], int(r["doc_lens"].sum()), len(r["ids"]))
    starts = np.cumsum(r["doc_lens"])[:-1]
    cross += int(r["lm"][starts].sum())
    ntok += int(r["lm"][1:].sum())
print("LOSS-TOKENS", ntok, "CROSS-BOUNDARY", cross, flush=True)
m = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
W = m.lm_head.weight
tot, n_in, per_row, t0 = 0.0, 0, [], time.time()
with torch.no_grad():
    for ri, r in enumerate(rows):
        rs, rn, s0 = 0.0, 0, 0
        for L in r["doc_lens"].tolist():
            ids, lm = r["ids"][s0:s0 + L], r["lm"][s0:s0 + L]
            s0 += L
            if L < 2 or not lm[1:].any():
                continue
            x = torch.tensor(ids.astype(np.int64), device="cuda")[None]
            h = m.model(input_ids=x).last_hidden_state[0][:-1]
            tgt = torch.tensor(ids[1:].astype(np.int64), device="cuda")
            msk = torch.tensor(lm[1:], device="cuda")
            for s in range(0, h.shape[0], 4096):
                lg = (h[s:s + 4096] @ W.T).float()
                nll = torch.nn.functional.cross_entropy(lg, tgt[s:s + 4096], reduction="none")
                rs += float((nll * msk[s:s + 4096]).sum())
                rn += int(msk[s:s + 4096].sum())
        tot += rs
        n_in += rn
        per_row.append({"row": ri, "src": r["src"], "docs": len(r["doc_lens"]), "loss_tokens": rn, "ce": rs / max(rn, 1)})
        print("ROW", ri, r["src"], rn, round(rs / max(rn, 1), 6), f"{time.time() - t0:.0f}s", flush=True)
assert n_in + cross == ntok, (n_in, cross, ntok)
res = {"loss_tokens": n_in, "cross_boundary_tokens": cross, "ce": tot / n_in, "rows": per_row, "hf": HF, "batch": D}
json.dump(res, open(OUT, "w"), indent=1)
print("REFCE", json.dumps({"loss_tokens": n_in, "cross_boundary_tokens": cross, "ce": round(tot / n_in, 6)}), flush=True)
