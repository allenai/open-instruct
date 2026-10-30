"""Write B2's LM tensors as an OLMo-core DCP that OLMoDDPTrainModule loads as a base (ledger H161 GPU companion).

  b2_lm_to_olmocore_dcp.py --src <B2 step dir> --ref <B0 olmo-core dir> --out <new dir; DCP goes in <new dir>/model_and_optim>

B2 (olmo35-small-mm-ckpt2-256k-amt step22240) exists only as a weights-only multimodal export: model.safetensors
(bf16, LM tensors under `lm.`) + olmo_core_config.json. The GPU SFT path (open_instruct/olmo_core_finetune.py) loads an
OLMo-core checkpoint with load_optim_state=False; with only `.main` keys present, OLMo-core 2f74ec30 loads the optimizer
main params strictly (a missing key raises) and copies them into the model. This writes exactly that layout, as B0's
DCP has it (H111 loaded B0 this way):

  module.<name>.main  float32, 1-D, numel = the tensor's numel, row-major flatten of the bf16 tensor upcast exactly.

Same selection as H134's strip_vision.py: a key is kept only if, after dropping the `lm.` prefix, it is one of B0's
415 masters (an explicit allowlist); every kept size must equal B0's; every dropped key must not look like an LM key.
It writes only a new directory, re-reads every tensor from the written DCP and checks it bit-for-bit against the
source, and writes RECEIPT.json with sha256 of every written file and of the source files.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

LM_TOP = ("blocks", "embeddings", "embedding_norm", "lm_head")


def sha_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


def fail(msg):
    raise SystemExit(f"CONVERT-FAIL: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lm-prefix", default="lm.")
    a = ap.parse_args()
    t0 = time.time()
    if os.path.exists(a.out):
        fail(f"{a.out} exists; this script writes only new paths")
    st_path = os.path.join(a.src, "model.safetensors")
    with open(os.path.join(a.ref, "model_and_optim", ".metadata"), "rb") as f:
        ref_md = pickle.load(f).state_dict_metadata
    ref = {k[len("module."):-len(".main")]: int(v.size.numel()) for k, v in ref_md.items()
           if k.startswith("module.") and k.endswith(".main")}
    print(f"B0 masters: {len(ref)}", flush=True)

    kept, dropped = {}, []
    with safe_open(st_path, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            name = k[len(a.lm_prefix):] if k.startswith(a.lm_prefix) else None
            if name in ref:
                kept[name] = k
            else:
                dropped.append(k)
                if (name or k).split(".")[0] in LM_TOP:
                    fail(f"dropped key looks like an LM key: {k}")
        missing = sorted(set(ref) - set(kept))
        if missing:
            fail(f"{len(missing)} B0 masters missing from B2, e.g. {missing[:5]}")
        print(f"B2 keys {len(keys)}: kept {len(kept)}, dropped {len(dropped)}", flush=True)
        sd, shapes, src_dtypes = {}, {}, {}
        for name, k in sorted(kept.items()):
            t = f.get_tensor(k)
            if t.numel() != ref[name]:
                fail(f"{name}: numel {t.numel()} != B0 {ref[name]}")
            shapes[name], src_dtypes[name] = list(t.shape), str(t.dtype)
            sd[f"module.{name}.main"] = t.to(torch.float32).reshape(-1).contiguous()
    print(f"loaded+upcast in {time.time() - t0:.0f}s; dtypes {sorted(set(src_dtypes.values()))}", flush=True)

    dcp_dir = os.path.join(a.out, "model_and_optim")  # Checkpointer.load looks here first
    os.makedirs(dcp_dir)
    dcp.save(sd, storage_writer=dcp.FileSystemWriter(dcp_dir, thread_count=8), no_dist=True)
    print(f"saved in {time.time() - t0:.0f}s", flush=True)

    # Re-read from the written DCP and compare bit-for-bit with the source tensors.
    numel_total = int(sum(v.numel() for v in sd.values()))
    back = {k: torch.empty_like(v) for k, v in sd.items()}
    del sd
    dcp.load(back, storage_reader=dcp.FileSystemReader(dcp_dir), no_dist=True)
    with safe_open(st_path, framework="pt") as f:
        bad = [n for n, k in kept.items()
               if not torch.equal(f.get_tensor(k).to(torch.float32).reshape(-1), back[f"module.{n}.main"])]
    if bad:
        fail(f"re-read mismatch on {len(bad)} tensors, e.g. {bad[:5]}")
    print(f"verified {len(back)} tensors bit-for-bit in {time.time() - t0:.0f}s", flush=True)

    files = sorted(os.listdir(dcp_dir))
    receipt = {
        "what": "B2 LM tensors as an OLMo-core DCP of fp32 optimizer main params (module.<name>.main), for GPU SFT",
        "script": "open-instruct scripts/c12/b2_lm_to_olmocore_dcp.py",
        "src": a.src,
        "src_sha256": {n: sha_file(os.path.join(a.src, n)) for n in sorted(os.listdir(a.src))},
        "ref_b0": a.ref,
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_keys": sorted(dropped),
        "numel_total": numel_total,
        "src_dtypes": sorted(set(src_dtypes.values())),
        "shapes": shapes,
        "verified_bitwise": True,
        "files_sha256": {f"model_and_optim/{n}": sha_file(os.path.join(dcp_dir, n)) for n in files},
        "torch": torch.__version__,
        "seconds": round(time.time() - t0),
    }
    with open(os.path.join(a.out, "RECEIPT.json"), "w") as f:
        json.dump(receipt, f, indent=1)
    print(f"RECEIPT sha256 {sha_file(os.path.join(a.out, 'RECEIPT.json'))}", flush=True)
    print("CONVERT-OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
