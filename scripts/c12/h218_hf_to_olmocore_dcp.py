"""Write an HF ``olmo3moe`` SFT export as the OLMo-core DCP that OLMoDDPTrainModule loads as a start (ledger H218).

  h218_hf_to_olmocore_dcp.py --src <HF export dir> --ref <base dir with model_and_optim/.metadata> --out <new dir>

The GPU SFT path (open_instruct/olmo_core_finetune.py) loads an OLMo-core checkpoint with load_optim_state=False; with
only ``.main`` keys present, OLMo-core 2f74ec30 loads the optimizer main params strictly and copies them into the model
(scripts/c12/b2_lm_to_olmocore_dcp.py, H161/H194). H181's C12r checkpoints exist only as HF exports of a MaxText run
(model.safetensors, bf16, transformers ``olmo3moe`` key names), so the HF state is first mapped to OLMo-core names by
OLMo-core's own ``convert_olmo3moe_state_from_hf`` (olmo_core.nn.hf.convert, the inverse of the exporter's mapping;
routed experts are stacked into the fused ``w_up_gate`` / ``w_down`` layout). That converter (OLMo-core 2f74ec30) has
no case for the scalable-softmax scale of the full-attention layers, ``model.layers.N.self_attn.ssmax_scale`` (one
value per head), which the base DCPs hold as ``blocks.N.attention.ssmax_scale``; this script maps those two itself,
by name, before calling it. The HF export holds the embedding and the head at the tokenizer's vocabulary (100,278
rows) while OLMo-core trains them at the padded vocabulary (100,352, a multiple of 128): the missing rows are copied
from the reference base DCP (--pad ref, default; these rows are never input tokens, and H194's step-1 CE matched the
HF forward within 1e-4 with the base's rows in place) or zero-filled (--pad zeros); the receipt records which and the
rows' magnitude. Then everything is written exactly as the base DCPs are:

  module.<name>.main  float32, 1-D, numel = the tensor's numel, row-major flatten of the bf16 tensor upcast exactly.

Checks: (1) a plan pass on meta tensors from the safetensors header: the converted key set must equal the reference
DCP's masters and every numel must match, before any weight is read; (2) every HF key is consumed (the converter
raises otherwise); (3) after writing, every tensor is re-read from the DCP and compared bit-for-bit with the upcast
source. Writes only a new directory and RECEIPT.json with sha256 of the source and of every written file.
"""

import argparse
import hashlib
import json
import os
import pickle
import struct
import sys
import time
import types

import torch
import torch.distributed.checkpoint as dcp
from olmo_core.nn.hf.convert import convert_olmo3moe_state_from_hf
from safetensors import safe_open

ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def sha_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


def fail(msg):
    raise SystemExit(f"CONVERT-FAIL: {msg}")


def read_header(st_path):
    with open(st_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


SSMAX_HF = ".self_attn.ssmax_scale"


def convert_state(config, hf_state):
    """OLMo-core's olmo3moe mapping plus the ssmax_scale tensors it does not know."""
    ssmax = {k: hf_state.pop(k) for k in list(hf_state) if k.endswith(SSMAX_HF)}
    core = convert_olmo3moe_state_from_hf(config, hf_state)
    for k, v in ssmax.items():
        layer = k[len("model.layers."):-len(SSMAX_HF)]
        core[f"blocks.{layer}.attention.ssmax_scale"] = v
    return core


PADDED = ("embeddings.weight", "lm_head.w_out.weight")


def pad_rows(core, ref_dir, ref, mode, meta=False):
    """Extend the two vocabulary matrices to the reference numel; returns the receipt entry."""
    out = {}
    for name in PADDED:
        t = core[name]
        rows, dim = t.shape
        want = ref[name] // dim
        if want * dim != ref[name] or want < rows:
            fail(f"{name}: reference numel {ref[name]} is not a whole number of rows >= {rows} x {dim}")
        extra = want - rows
        if extra == 0:
            out[name] = {"rows": rows, "padded_rows": 0}
            continue
        if meta:
            core[name] = torch.empty((want, dim), dtype=t.dtype, device="meta")
            out[name] = {"rows": rows, "padded_rows": extra, "mode": mode}
            continue
        if mode == "zeros":
            pad = torch.zeros((extra, dim), dtype=t.dtype)
        else:
            key = f"module.{name}.main"
            back = {key: torch.empty(ref[name], dtype=torch.float32)}
            dcp.load(back, storage_reader=dcp.FileSystemReader(os.path.join(ref_dir, "model_and_optim")), no_dist=True)
            pad = back[key].reshape(want, dim)[rows:].to(t.dtype)
        core[name] = torch.cat([t, pad], dim=0).contiguous()
        out[name] = {"rows": rows, "padded_rows": extra, "mode": mode,
                     "pad_abs_max": float(pad.float().abs().max()), "pad_abs_mean": float(pad.float().abs().mean())}
    return out


def ref_masters(ref_dir):
    with open(os.path.join(ref_dir, "model_and_optim", ".metadata"), "rb") as f:
        md = pickle.load(f).state_dict_metadata
    return {k[len("module."):-len(".main")]: int(v.size.numel()) for k, v in md.items()
            if k.startswith("module.") and k.endswith(".main")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF export dir (config.json + model.safetensors)")
    ap.add_argument("--ref", required=True, help="a base DCP dir: its model_and_optim/.metadata names the masters")
    ap.add_argument("--out", required=True, help="new dir; the DCP goes in <out>/model_and_optim")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--pad", choices=("ref", "zeros"), default="ref",
                    help="rows 100,278.. of the embedding and head: the reference base's (default) or zeros")
    a = ap.parse_args()
    t0 = time.time()
    if os.path.exists(a.out):
        fail(f"{a.out} exists; this script writes only new paths")
    st_path = os.path.join(a.src, "model.safetensors")
    config = types.SimpleNamespace(**json.load(open(os.path.join(a.src, "config.json"))))
    hdr = read_header(st_path)
    ref = ref_masters(a.ref)
    print(f"HF keys {len(hdr)}; reference masters {len(ref)}", flush=True)

    # (1) plan on meta tensors: names and sizes only.
    meta = {k: torch.empty(v["shape"], dtype=ST_DTYPES[v["dtype"]], device="meta") for k, v in hdr.items()}
    planned = convert_state(config, meta)
    pad_plan = pad_rows(planned, a.ref, ref, a.pad, meta=True)
    print(f"PAD {json.dumps(pad_plan)}", flush=True)
    missing = sorted(set(ref) - set(planned))
    extra = sorted(set(planned) - set(ref))
    if missing or extra:
        fail(f"key sets differ: {len(missing)} masters missing from the conversion {missing[:8]}; "
             f"{len(extra)} converted keys unknown to the reference {extra[:8]}")
    bad = [(k, planned[k].numel(), ref[k]) for k in ref if planned[k].numel() != ref[k]]
    if bad:
        fail(f"numel differs on {len(bad)} keys, e.g. {bad[:5]}")
    shapes = {k: list(planned[k].shape) for k in sorted(planned)}
    print(f"PLAN-OK {len(planned)} keys match the reference masters in name and numel", flush=True)
    if a.plan_only:
        print(json.dumps({"keys": len(planned), "shapes": shapes}, indent=1))
        return

    # (2) the real pass.
    hf_state = {}
    with safe_open(st_path, framework="pt") as f:
        for k in sorted(hdr):
            hf_state[k] = f.get_tensor(k)
    src_dtypes = sorted({str(t.dtype) for t in hf_state.values()})
    print(f"read {len(hf_state)} tensors in {time.time() - t0:.0f}s; dtypes {src_dtypes}", flush=True)
    core = convert_state(config, hf_state)
    del hf_state
    padding = pad_rows(core, a.ref, ref, a.pad)
    print(f"PAD {json.dumps(padding)}", flush=True)
    if {k: list(v.shape) for k, v in core.items()} != shapes:
        fail("converted shapes differ from the plan")
    sd = {f"module.{k}.main": v.to(torch.float32).reshape(-1).contiguous() for k, v in sorted(core.items())}
    numel_total = int(sum(v.numel() for v in sd.values()))
    print(f"converted+upcast in {time.time() - t0:.0f}s; {numel_total} elements", flush=True)

    dcp_dir = os.path.join(a.out, "model_and_optim")  # Checkpointer.load looks here first
    os.makedirs(dcp_dir)
    dcp.save(sd, storage_writer=dcp.FileSystemWriter(dcp_dir, thread_count=8), no_dist=True)
    print(f"saved in {time.time() - t0:.0f}s", flush=True)

    # (3) re-read from the written DCP and compare bit-for-bit with the upcast source.
    shapes_dtypes = {k: (v.shape, v.dtype) for k, v in sd.items()}
    del sd
    back = {k: torch.empty(s, dtype=d) for k, (s, d) in shapes_dtypes.items()}
    dcp.load(back, storage_reader=dcp.FileSystemReader(dcp_dir), no_dist=True)
    mismatched = [k for k, v in core.items()
                  if not torch.equal(v.to(torch.float32).reshape(-1), back[f"module.{k}.main"])]
    if mismatched:
        fail(f"re-read mismatch on {len(mismatched)} tensors, e.g. {mismatched[:5]}")
    print(f"verified {len(back)} tensors bit-for-bit in {time.time() - t0:.0f}s", flush=True)

    files = sorted(os.listdir(dcp_dir))
    receipt = {
        "what": "HF olmo3moe SFT export as an OLMo-core DCP of fp32 optimizer main params (module.<name>.main), for GPU SFT",
        "script": "open-instruct scripts/c12/h218_hf_to_olmocore_dcp.py",
        "mapping": "olmo_core.nn.hf.convert.convert_olmo3moe_state_from_hf + ssmax_scale by name (this script)",
        "src": a.src,
        "src_sha256": {n: sha_file(os.path.join(a.src, n)) for n in sorted(os.listdir(a.src))
                       if os.path.isfile(os.path.join(a.src, n))},
        "ref": a.ref,
        "keys": len(core),
        "numel_total": numel_total,
        "vocab_padding": padding,
        "src_dtypes": src_dtypes,
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
