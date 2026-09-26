"""Disposable, source-audited KDA decode ablations; no production defaults change."""

import hashlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

from olmo_sglang.kda import packed_decode

VARIANTS = ("baseline", "q_round", "k_round", "qk_round", "qk_round_mul", "norm_mul")


def variant_source(source, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    marker = "    query *= scale\n"
    if source.count(marker) != 1:
        raise ValueError("Pinned decode source changed")
    if variant in {"qk_round_mul", "norm_mul"}:
        for name in ("query", "key"):
            old = f"{name} = {name} / tl.sqrt(tl.sum({name} * {name}) + 1e-6)"
            new = f"{name} = {name} * (1.0 / tl.sqrt(tl.sum({name} * {name}) + 1e-6))"
            if source.count(old) != 1:
                raise ValueError("Pinned normalization source changed")
            source = source.replace(old, new)
    rounded = {
        "q_round": ("query",),
        "k_round": ("key",),
        "qk_round": ("query", "key"),
        "qk_round_mul": ("query", "key"),
    }.get(variant, ())
    casts = "".join(f"    {name} = {name}.to(mixed_qkv.dtype.element_ty).to(tl.float32)\n" for name in rounded)
    return source.replace(marker, casts + marker)


def load_variant(variant, root):
    source = inspect.getsource(packed_decode)
    updated = variant_source(source, variant)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"kda_{variant}.py"
    path.write_text(updated)
    name = f"_diagnostic_kda_{variant}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.olmo_packed_kda_decode, hashlib.sha256(updated.encode()).hexdigest()


def install():
    variant = os.environ.get("OI_KDA_ABLATION", "baseline")
    if variant == "baseline":
        return
    function, digest = load_variant(variant, Path("/tmp/selective-kda-kernels"))
    packed_decode.olmo_packed_kda_decode = function
    print("KDA_ABLATION", variant, digest, flush=True)
