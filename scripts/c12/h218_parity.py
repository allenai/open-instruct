"""Ledger H218 gate (3) diagnosis: the OLMo-core model built from a converted DCP against the HF export, one document.

  python h218_parity.py --dcp <dir>/model_and_optim --config scripts/train/debug/kda_hero_small_sft.json \
      --hf <HF export dir> --batch <rankNN.npz from the smoke> [--doc 0] [--max-tokens 4096] --out <json>

Builds the OLMo-core model exactly as open_instruct/olmo_core_finetune.py does (TransformerConfig from the training
config's "model" section, the tokenizer's padded vocabulary), loads every ``module.<name>.main`` fp32 tensor of the
DCP into it (fp32 mains cast to bf16, as the trainer's model params are), and forwards one document of the dumped
step-1 batch in bf16 on one GPU; the attention and KDA kernels accept only half precision. The HF model (the export's
bundled olmo3moe code, trust_remote_code, bf16 weights, as h161_refce.py runs it) forwards the same tokens; logits
are compared in fp32. Reports, per position, the
max |logit difference|, the token-weighted CE of each side on the document's loss tokens, and the mean KL(HF ||
core); also runs the same on the first 512 tokens so a divergence that grows with position is visible. A faithful
conversion gives bf16-noise logit differences (|d| of order 1e-1 at most on a few positions) and CE equal to ~1e-3.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.distributed.checkpoint as dcp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--doc", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, os.getcwd())
    from olmo_core.nn.transformer import TransformerConfig
    from transformers import AutoModelForCausalLM, AutoConfig

    payload = json.load(open(a.config))
    model_cfg = TransformerConfig.from_dict(payload.get("model", payload))
    hf_cfg = AutoConfig.from_pretrained(a.hf, trust_remote_code=True)
    print(f"OLMo-core config: vocab {model_cfg.vocab_size}, HF vocab {hf_cfg.vocab_size}", flush=True)

    # --- OLMo-core model from the DCP mains (fp32, as the trainer's optimizer holds them) ---
    with open(os.path.join(a.dcp, ".metadata"), "rb") as f:
        md = pickle.load(f).state_dict_metadata
    keys = [k for k in md if k.startswith("module.") and k.endswith(".main")]
    sd = {k: torch.empty(md[k].size, dtype=torch.float32) for k in keys}
    dcp.load(sd, storage_reader=dcp.FileSystemReader(a.dcp), no_dist=True)
    print(f"loaded {len(sd)} mains in {time.time() - t0:.0f}s", flush=True)
    model = model_cfg.build(init_device="cpu")
    model = model.to(torch.bfloat16)
    msd = model.state_dict()
    missing = sorted(set(msd) - {k[len("module."):-len(".main")] for k in keys})
    extra = sorted({k[len("module."):-len(".main")] for k in keys} - set(msd))
    print(f"model params {len(msd)}; mains not in model {extra[:5]} ({len(extra)}); model params without a main {missing[:5]} ({len(missing)})", flush=True)
    for k in keys:
        name = k[len("module."):-len(".main")]
        if name in msd:
            msd[name].copy_(sd[k].reshape(msd[name].shape).to(msd[name].dtype))
    model.load_state_dict(msd)
    del sd
    model = model.cuda().eval()

    # --- HF model ---
    hf = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    import transformers
    print(f"HF loaded ({transformers.__version__}) in {time.time() - t0:.0f}s", flush=True)

    b = np.load(a.batch)
    ids_row, lm_row, dl = b["input_ids"][0], b["label_mask"][0].astype(bool), b["doc_lens"][0]
    dl = dl[dl > 0]
    starts = np.concatenate([[0], np.cumsum(dl)[:-1]])
    s0, L = int(starts[a.doc]), int(dl[a.doc])
    ids = ids_row[s0:s0 + L][: a.max_tokens].astype(np.int64)
    lm = lm_row[s0:s0 + L][: a.max_tokens]
    x = torch.tensor(ids, device="cuda")[None]
    results = {"doc": a.doc, "doc_len": L, "tokens": int(len(ids)), "loss_tokens": int(lm[1:].sum())}
    with torch.no_grad():
        core_out = model(x)
        core_logits = core_out if torch.is_tensor(core_out) else core_out.logits
        core_logits = core_logits[0].float()[:, : hf_cfg.vocab_size]
        hf_logits = hf(input_ids=x).logits[0].float()
    tgt = torch.tensor(ids[1:], device="cuda")
    msk = torch.tensor(lm[1:], device="cuda")

    def ce(logits):
        nll = torch.nn.functional.cross_entropy(logits[:-1], tgt, reduction="none")
        return float((nll * msk).sum() / msk.sum())

    d = (core_logits - hf_logits).abs()
    results.update({
        "max_abs_logit_diff": float(d.max()),
        "mean_abs_logit_diff": float(d.mean()),
        "max_abs_logit_diff_first512": float(d[:512].max()),
        "max_abs_logit_diff_last512": float(d[-512:].max()),
        "argmax_agreement": float((core_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean()),
        "ce_core": ce(core_logits), "ce_hf": ce(hf_logits),
        "kl_hf_core_mean": float(torch.nn.functional.kl_div(
            torch.log_softmax(core_logits, -1), torch.log_softmax(hf_logits, -1), log_target=True, reduction="none").sum(-1).mean()),
        "padded_rows_logit_max": float(core_out[0].float()[:, hf_cfg.vocab_size:].max()) if (core_out if torch.is_tensor(core_out) else core_out.logits).shape[-1] > hf_cfg.vocab_size else None,
        "transformers": transformers.__version__, "seconds": round(time.time() - t0),
    })
    # per-position profile of the difference, coarse
    prof = d.max(-1).values
    results["max_diff_by_block_of_256"] = [float(prof[i:i + 256].max()) for i in range(0, len(prof), 256)]
    json.dump(results, open(a.out, "w"), indent=1)
    print("H218-PARITY", json.dumps({k: v for k, v in results.items() if k != "max_diff_by_block_of_256"}), flush=True)
    print("H218-PARITY-PROFILE", results["max_diff_by_block_of_256"], flush=True)


if __name__ == "__main__":
    sys.exit(main())
