"""Strip the vision parts from a multimodal OLMo-core hero-small checkpoint so the text-only converter can take it (H134).

  strip_vision.py inspect <checkpoint step dir> <out json>
  strip_vision.py plan    --ref <B0 masters manifest json> --ref-config <B0 config.json> --src <checkpoint step dir>
                          --out <plan json> [--lm-prefix P] [--lm-config-key model] [--mm-tokenizer-size N]
  strip_vision.py apply   --plan <plan json> --masters-in <dir> --masters-out <dir> --receipt <json>

Stage 0 (plan) reads only the DCP metadata. A key is kept only if, after dropping `module.`/`.main` and the optional
LM prefix, it is one of B0's 415 fp32 masters (an explicit allowlist, not a guess at vision names); every other `.main`
key is dropped and listed. It fails closed on: a missing B0 key, a kept key whose size differs from B0's (except the
embedding and LM-head rows), a non-fp32 kept key, a dropped key that looks like an LM key (blocks.*, embeddings*,
lm_head*, embedding_norm*), an LM config that differs from B0's in anything but vocab_size and init fields, and image
tokens placed in the padding rows (text vocab 100278 < tokenizer size <= 100352), which needs a decision.
Embedding / LM-head rows above B0's padded 100352 (appended image tokens) are sliced off in apply.

Stage 1b (apply) runs on the masters dumped for the kept keys (dump_olmocore_masters_mm.py). It writes each kept
tensor under its B0 name, slices the vocab rows when planned, re-reads every written file and checks it byte-for-byte:
whole tensors against the source, sliced tensors on rows 0..text_vocab-1 (and the padding rows) against the source
rows. It writes a config.json for stage 2 (the LM section with vocab_size = B0's) and a receipt. Writes only new paths.
"""

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import sys

LM_TOP = ("blocks", "embeddings", "embedding_norm", "lm_head")
VOCAB_KEYS = ("embeddings.weight", "lm_head.w_out.weight")
CONFIG_FREE = {"vocab_size", "init_seed", "init_std", "init_method"}  # may differ without changing the LM's shapes
# Runtime-only fields (expert-parallel comms, kernel choice) that don't change the weights or the math, by path suffix.
RUNTIME_SUFFIXES = (".ep.share_combine_out", ".ep.share_dispatch_out", ".sequence_mixer.use_cute_kernel",
                    ".sequence_mixer.use_experimental_kernels")
ST_DTYPES = {"F32": "float32", "BF16": "bfloat16", "F16": "float16"}


class Fail(SystemExit):
    def __init__(self, msg):
        super().__init__(f"STRIP-FAIL: {msg}")


def sha_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _PermissiveUnpickler(pickle.Unpickler):
    """Loads DCP .metadata even when it pickles classes from a codebase that isn't installed (e.g. a vision branch)."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_Stub,), {"__module__": module})


def read_dcp_metadata(step_dir):
    path = os.path.join(step_dir, "model_and_optim", ".metadata")
    with open(path, "rb") as f:
        raw = f.read()
    try:
        md = pickle.loads(raw)
        permissive = False
    except (ImportError, AttributeError):
        md = _PermissiveUnpickler(io.BytesIO(raw)).load()
        permissive = True
    return md, hashlib.sha256(raw).hexdigest(), permissive


def main_keys(md):
    """{full DCP key: (numel, dtype str)} for the fp32 master copies `module.<param>.main`."""
    out = {}
    for k, v in md.state_dict_metadata.items():
        if k.startswith("module.") and k.endswith(".main"):
            size = getattr(v, "size", None)
            if size is None:
                continue
            n = 1
            for s in size:
                n *= int(s)
            dt = getattr(getattr(v, "properties", None), "dtype", None)
            out[k] = (n, str(dt).replace("torch.", "") if dt is not None else "unknown")
    return out


def lm_name(full_key, prefix):
    k = full_key[len("module.") : -len(".main")]
    if prefix and k.startswith(prefix):
        return k[len(prefix) :], True
    return k, not prefix


def looks_lm(name):
    return name.split(".")[0] in LM_TOP or re.match(r"^(model\.)?(blocks|embeddings|lm_head|embedding_norm)\b", name)


def get_path(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            raise Fail(f"LM config key {dotted!r} not in the checkpoint's config.json")
        d = d[part]
    return d


def safetensors_shapes(path):
    import struct

    with open(path, "rb") as f:
        h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    h.pop("__metadata__", None)
    return {k: v["shape"] for k, v in h.items()}


def safetensors_keys(path):
    """{module.<key>.main: (numel, dtype)} from a safetensors header (weights-only export), and the header's sha256."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    h = json.loads(raw)
    h.pop("__metadata__", None)
    out = {}
    for k, v in h.items():
        numel = 1
        for d in v["shape"]:
            numel *= int(d)
        out[f"module.{k}.main"] = (numel, ST_DTYPES.get(v["dtype"], v["dtype"]))
    return out, hashlib.sha256(raw).hexdigest()


def config_diffs(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += config_diffs(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [path]


def cmd_plan(a, keys=None, md_sha="fixture", permissive=False):
    ref = json.load(open(a.ref))["keys"]
    ref_cfg = json.load(open(a.ref_config))["model"]
    src_cfg_path = getattr(a, "src_config", None) or os.path.join(a.src, "config.json")
    src_cfg_all = json.load(open(src_cfg_path))
    src_cfg = get_path(src_cfg_all, a.lm_config_key)
    D, V_ref = int(ref_cfg["d_model"]), int(ref_cfg["vocab_size"])
    text_vocab = a.text_vocab
    st_path = getattr(a, "src_safetensors", None)
    fmt = "safetensors" if st_path else "dcp"
    if keys is None:
        if st_path:
            keys, md_sha = safetensors_keys(st_path)
        else:
            md, md_sha, permissive = read_dcp_metadata(a.src)
            keys = main_keys(md)
    ok_dtypes = {"float32"} | ({"bfloat16"} if getattr(a, "allow_bf16_upcast", False) else set())
    full_shapes = safetensors_shapes(st_path) if st_path else None
    ref_shapes = json.load(open(a.ref_shapes))["shapes"] if getattr(a, "ref_shapes", None) else None
    if full_shapes is not None and ref_shapes is None:
        raise Fail("--ref-shapes is required for a safetensors source: numel alone can't catch a transposed tensor")

    runtime, diffs = [], []
    for d in config_diffs(ref_cfg, src_cfg):
        top = d.split(".")[0]
        if top in CONFIG_FREE or d.endswith(RUNTIME_SUFFIXES):
            runtime.append(d)
        elif d == "n_extra_vocab" and src_cfg.get("n_extra_vocab") in (None, 0) and ref_cfg.get("n_extra_vocab") in (None, 0):
            runtime.append(d)
        else:
            diffs.append(d)
    if diffs:
        raise Fail(f"LM config differs from B0's in {diffs[:20]}")
    if int(src_cfg["d_model"]) != D:
        raise Fail(f"d_model {src_cfg['d_model']} != {D}")

    kept, dropped, seen = {}, [], set()
    for full, (n, dt) in sorted(keys.items()):
        name, prefixed = lm_name(full, a.lm_prefix)
        if prefixed and name in ref:
            if name in seen:
                raise Fail(f"two source keys map to {name}")
            seen.add(name)
            if dt not in ok_dtypes:
                raise Fail(f"{full}: dtype {dt}, expected one of {sorted(ok_dtypes)}")
            want = 1
            for s in ref[name]["shape"]:
                want *= int(s)
            rows = None
            if name in VOCAB_KEYS:
                if n % D:
                    raise Fail(f"{full}: {n} elements is not a multiple of d_model {D}")
                rows = n // D
                if rows < V_ref:
                    raise Fail(f"{full}: {rows} rows < B0's {V_ref}")
            elif n != want:
                raise Fail(f"{full}: {n} elements, B0 has {want}")
            if full_shapes is not None:
                got_shape = full_shapes[full[len("module.") : -len(".main")]]
                if name in VOCAB_KEYS:
                    want_shape = [rows, D]
                else:
                    want_shape = ref_shapes.get(name)
                if list(got_shape) != list(want_shape or []):
                    raise Fail(f"{full}: shape {got_shape}, expected {want_shape} (B0 layout, row-major)")
            kept[name] = {"src_key": full, "numel": n, "rows": rows, "dtype": dt}
        elif looks_lm(name) or (a.lm_prefix and prefixed):
            raise Fail(f"extra LM-like key {full} (not one of B0's masters)")
        else:
            dropped.append({"key": full, "numel": n, "dtype": dt})
    missing = sorted(set(ref) - seen)
    if missing:
        raise Fail(f"{len(missing)} B0 keys missing, e.g. {missing[:10]}")

    rows = {kept[k]["rows"] for k in VOCAB_KEYS}
    if len(rows) != 1:
        raise Fail(f"embedding and LM-head rows differ: {rows}")
    rows = rows.pop()
    tok = a.mm_tokenizer_size
    if tok is None:
        raise Fail("--mm-tokenizer-size is required: without it, image tokens in rows 100278..100351 can't be ruled out")
    pad_rule = getattr(a, "pad_rows", "fail")
    if text_vocab < tok <= V_ref and pad_rule != "passthrough":
        raise Fail(f"tokenizer size {tok}: image tokens sit in the padding rows {text_vocab}..{V_ref - 1}; needs a decision")
    img = []
    for k, v in _walk(src_cfg_all):
        if re.search(r"token_ids?$", k):
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, int) and not isinstance(x, bool) and text_vocab <= x < V_ref:
                    img.append(x)
    if img and pad_rule != "passthrough":
        raise Fail(f"config names token ids {sorted(set(img))} inside the padding rows; needs a ruling (--pad-rows passthrough)")
    if tok > V_ref and rows < tok:
        raise Fail(f"tokenizer size {tok} > embedding rows {rows}")
    if tok <= text_vocab and rows != V_ref:
        raise Fail(f"text-only tokenizer ({tok}) but {rows} embedding rows; expected {V_ref}")
    plan = {
        "src": os.path.abspath(a.src),
        "source_format": fmt,
        "safetensors_path": os.path.abspath(st_path) if st_path else None,
        "src_config_path": os.path.abspath(src_cfg_path),
        "src_config_sha256": sha_file(src_cfg_path),
        "runtime_only_config_diffs": runtime,
        "pad_token_ids_in_padding_rows": sorted(set(img)),
        "pad_rows_rule": pad_rule,
        "upcast_bf16_to_fp32": "bfloat16" in {kept_dt for kept_dt in (keys[v["src_key"]][1] for v in kept.values())},
        "dcp_metadata_sha256": md_sha,
        "metadata_permissive_unpickle": permissive,
        "lm_prefix": a.lm_prefix,
        "lm_config_key": a.lm_config_key,
        "d_model": D,
        "ref_vocab_rows": V_ref,
        "text_vocab": text_vocab,
        "mm_tokenizer_size": tok,
        "src_vocab_rows": rows,
        "vocab_slice": rows > V_ref,
        "kept": kept,
        "n_kept": len(kept),
        "kept_params": sum(v["numel"] for v in kept.values()),
        "dropped": dropped,
        "n_dropped": len(dropped),
        "dropped_params": sum(d["numel"] for d in dropped),
        "lm_config": dict(src_cfg, vocab_size=V_ref),
    }
    with open(a.out, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"PLAN-OK kept {len(kept)} ({plan['kept_params']}) dropped {len(dropped)} ({plan['dropped_params']}) "
          f"rows {rows} -> {V_ref if plan['vocab_slice'] else rows}")
    return plan


def _write(path, arr):
    import numpy as np

    np.save(path, np.ascontiguousarray(arr))


def cmd_apply(a, writer=_write):
    import numpy as np

    plan = json.load(open(a.plan))
    if os.path.exists(a.masters_out):
        raise Fail(f"{a.masters_out} exists; not overwriting")
    tmp = a.masters_out + ".tmp"
    if os.path.exists(tmp):
        raise Fail(f"{tmp} exists; not overwriting")
    rp = os.path.abspath(a.receipt)
    if os.path.exists(rp):
        raise Fail(f"{rp} exists; not overwriting")
    if rp.startswith(os.path.abspath(a.masters_out) + os.sep) or rp.startswith(os.path.abspath(tmp) + os.sep):
        raise Fail("the receipt must not be written inside the output dir")
    os.makedirs(tmp)
    D, V, T = plan["d_model"], plan["ref_vocab_rows"], plan["text_vocab"]
    rec = {"plan_sha256": sha_file(a.plan), "tensors": {}, "pad_rows": {}}
    st = None
    if plan.get("source_format") == "safetensors":
        import torch
        from safetensors import safe_open

        _, hsha = safetensors_keys(plan["safetensors_path"])
        if hsha != plan["dcp_metadata_sha256"]:
            raise Fail(f"safetensors header sha {hsha[:16]} != plan's {plan['dcp_metadata_sha256'][:16]}")
        st = safe_open(plan["safetensors_path"], "pt")
    for name, info in sorted(plan["kept"].items()):
        src_name = info["src_key"][len("module.") : -len(".main")]
        src_bits = None
        if st is not None:
            t = st.get_tensor(src_name).contiguous()
            if str(t.dtype).replace("torch.", "") != info.get("dtype", "float32"):
                raise Fail(f"{src_name}: dtype {t.dtype}, planned {info.get('dtype')}")
            bits_dt = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[t.dtype]
            src_bits = t.view(bits_dt).numpy().reshape(-1)
            src = t.float().numpy().reshape(-1)  # bf16 -> fp32 is exact
            if t.dtype != torch.float32 and not torch.equal(torch.from_numpy(src).to(t.dtype).view(bits_dt), t.view(bits_dt).reshape(-1)):
                raise Fail(f"{src_name}: fp32 upcast does not round-trip to the source bits")
        else:
            src = np.load(os.path.join(a.masters_in, src_name + ".npy"), mmap_mode="r")
        if src.dtype != np.float32 or src.size != info["numel"]:
            raise Fail(f"{src_name}: loaded {src.dtype} x {src.size}, planned float32 x {info['numel']}")
        out = os.path.join(tmp, name + ".npy")
        if info["rows"] is not None and info["rows"] > V:
            writer(out, src[: V * D])
        else:
            writer(out, src)
        got = np.load(out, mmap_mode="r")
        r = {"src_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(src if src_bits is None else src_bits))).hexdigest()}
        if info["rows"] is not None:
            r["text_rows_sha256"] = hashlib.sha256(memoryview(np.ascontiguousarray(src[: T * D]))).hexdigest()
            r["pad_rows_sha256"] = hashlib.sha256(memoryview(np.ascontiguousarray(src[T * D : V * D]))).hexdigest()
            if got.size != V * D or not np.array_equal(got[: V * D].view(np.uint32), src[: V * D].view(np.uint32)):
                raise Fail(f"{name}: written rows 0..{V - 1} differ from the source rows")
            r["dropped_rows"] = info["rows"] - V
            pad = np.linalg.norm(np.asarray(got[T * D : V * D], dtype=np.float64).reshape(V - T, D), axis=1)
            rec["pad_rows"][name] = {"first_row": T, "l2": [round(float(x), 6) for x in pad],
                                     "n_nonzero": int((pad > 0).sum())}
        elif got.size != src.size or not np.array_equal(got.view(np.uint32), src.view(np.uint32)):
            raise Fail(f"{name}: written tensor differs from the source")
        r["out_sha256"] = hashlib.sha256(memoryview(np.ascontiguousarray(got))).hexdigest()
        rec["tensors"][name] = r
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump({"model": plan["lm_config"]}, f, indent=1)
    os.rename(tmp, a.masters_out)
    rec.update(
        n_kept=len(rec["tensors"]),
        kept_params=plan["kept_params"],
        n_dropped=plan["n_dropped"],
        dropped_params=plan["dropped_params"],
        dropped_keys=[[d["key"], d["numel"]] for d in plan["dropped"]],
        vocab={k: plan[k] for k in ("src_vocab_rows", "ref_vocab_rows", "text_vocab", "mm_tokenizer_size", "vocab_slice")},
        source_format=plan.get("source_format", "dcp"),
        upcast_bf16_to_fp32=plan.get("upcast_bf16_to_fp32", False),
        pad_token_ids_in_padding_rows=plan.get("pad_token_ids_in_padding_rows", []),
        pad_rows_rule=plan.get("pad_rows_rule", "fail"),
        runtime_only_config_diffs=plan.get("runtime_only_config_diffs", []),
        dcp_metadata_sha256=plan["dcp_metadata_sha256"],
        config_sha256=sha_file(os.path.join(a.masters_out, "config.json")),
        manifest_sha256=hashlib.sha256(json.dumps(rec["tensors"], sort_keys=True).encode()).hexdigest(),
        text_weights_unchanged=True,
    )
    with open(a.receipt, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"APPLY-OK {rec['n_kept']} tensors, manifest {rec['manifest_sha256'][:16]}, dropped {rec['n_dropped']} keys")
    return rec


def cmd_inspect(a):
    """Read-only facts for the amendment: DCP .main keys (numel/dtype), non-.main key families, config scheduler fields."""
    md, md_sha, permissive = read_dcp_metadata(a.src)
    keys = main_keys(md)
    other = {}
    for k in md.state_dict_metadata:
        if not (k.startswith("module.") and k.endswith(".main")):
            fam = re.sub(r"\d+", "N", k)
            other[fam] = other.get(fam, 0) + 1
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    sched = {}

    def walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                p = f"{path}.{k}" if path else k
                if re.search(r"sched|lr|warmup|decay|max_duration|t_max|alpha|duration|steps", k, re.I) and not isinstance(v, dict):
                    sched[p] = v
                walk(v, p)

    walk(cfg)
    out = {
        "src": os.path.abspath(a.src),
        "dcp_metadata_sha256": md_sha,
        "metadata_permissive_unpickle": permissive,
        "n_main": len(keys),
        "main_params": sum(n for n, _ in keys.values()),
        "main": {k: [n, dt] for k, (n, dt) in sorted(keys.items())},
        "other_key_families": dict(sorted(other.items())),
        "config_top": sorted(cfg),
        "config_sched_fields": sched,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"INSPECT-OK {len(keys)} .main keys, {out['main_params']} params, {len(other)} other key families")


def _walk(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            yield p, v
            yield from _walk(v, p)


def cmd_verify_receipt(a):
    r = json.load(open(os.path.join(a.dir, "RECEIPT.json")))
    for f, v in r["files"].items():
        if sha_file(os.path.join(a.dir, f)) != v["sha256"]:
            raise Fail(f"{f}: sha256 differs from RECEIPT.json")
    print(f"RECEIPT-OK {len(r['files'])} files")


def build_parser():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("inspect")
    i.add_argument("src")
    i.add_argument("out")
    pl = sp.add_parser("plan")
    pl.add_argument("--ref", required=True)
    pl.add_argument("--ref-config", required=True)
    pl.add_argument("--src", required=True)
    pl.add_argument("--out", required=True)
    pl.add_argument("--lm-prefix", default="")
    pl.add_argument("--lm-config-key", default="model")
    pl.add_argument("--mm-tokenizer-size", type=int)
    pl.add_argument("--text-vocab", type=int, default=100278)
    pl.add_argument("--ref-shapes", help="B0's 2-D parameter shapes (json {shapes: {name: shape}}); required for safetensors")
    pl.add_argument("--src-safetensors", help="weights-only OLMo-core export (model.safetensors) instead of a DCP")
    pl.add_argument("--src-config", help="config file when it isn't <src>/config.json (e.g. olmo_core_config.json)")
    pl.add_argument("--allow-bf16-upcast", action="store_true", help="accept bf16 weights; written as exact fp32 upcasts")
    pl.add_argument("--pad-rows", choices=("fail", "passthrough"), default="fail",
                    help="token ids inside the padding rows: fail (default) or pass every row through (a ruling)")
    vr = sp.add_parser("verify-receipt")
    vr.add_argument("dir")
    ap = sp.add_parser("apply")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--masters-in", required=True)
    ap.add_argument("--masters-out", required=True)
    ap.add_argument("--receipt", required=True)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"inspect": cmd_inspect, "plan": cmd_plan, "apply": cmd_apply, "verify-receipt": cmd_verify_receipt}[args.cmd](args)
    sys.exit(0)
