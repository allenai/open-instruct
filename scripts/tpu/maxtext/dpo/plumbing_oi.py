"""DPO plumbing test, open-instruct side: recompute the loss from plumbing_jax.py's toy model with open-instruct's
calculate_per_token_logps, _get_batch_logps and dpo_loss, laid out as open-instruct pads a batch.
    python plumbing_oi.py out.npz
"""

import sys

import numpy as np
import torch

from open_instruct import dpo_utils

d = np.load(sys.argv[1])
n, beta = int(d["n"]), float(d["beta"])


def logits(prefix, ids):
    tok, pos = torch.tensor(d[f"tok_{prefix}"]), torch.tensor(d[f"pos_{prefix}"])
    return tok[ids] + pos[torch.arange(ids.shape[1])][None]


seqs = [d[f"{s}_input_ids_{i}"] for s in ("chosen", "rejected") for i in range(n)]
labs = [d[f"{s}_labels_{i}"] for s in ("chosen", "rejected") for i in range(n)]
width = max(len(x) for x in seqs)
ids = torch.tensor(np.stack([np.pad(x, (0, width - len(x))) for x in seqs]))
labels = torch.tensor(np.stack([np.pad(x, (0, width - len(x)), constant_values=-100) for x in labs]))

failed = False
for loss_type, average in (("dpo", False), ("dpo_norm", True)):
    lp = {}
    for prefix in ("policy", "ref"):
        per_tok = dpo_utils.calculate_per_token_logps(logits(prefix, ids), labels)
        lp[prefix] = dpo_utils._get_batch_logps(per_tok, labels, average_log_prob=average)
    losses, _, _ = dpo_utils.dpo_loss(lp["policy"][:n], lp["policy"][n:], lp["ref"][:n], lp["ref"][n:], beta=beta)
    oi, mt = losses.mean().item(), float(d[f"loss_{loss_type}"])
    ok = abs(oi - mt) <= 1e-5 * max(1.0, abs(oi))
    failed |= not ok
    print(f"{loss_type}: open-instruct {oi:.6f}  maxtext {mt:.6f}  {'OK' if ok else 'MISMATCH'}")
sys.exit(1 if failed else 0)
