"""Unit tests for strip_vision.py on small synthetic tensors (no DCP, no torch). Run with pytest or as a script."""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strip_vision as sv  # noqa: E402

D, V, T = 2, 100352, 100278
REF = {
    "embeddings.weight": [V * D],
    "embedding_norm.weight": [D],
    "blocks.0.attention.w_q.weight": [D * D],
    "lm_head.norm.weight": [D],
    "lm_head.w_out.weight": [V * D],
}
LM_CFG = {"d_model": D, "vocab_size": V, "n_layers": 1, "block": {"name": "x"}}


def setup(tmp, rows=V, prefix="llm.", extra=None, drop=None, dtype="float32", cfg=None, tok=T):
    ref = os.path.join(tmp, "ref.json")
    json.dump({"keys": {k: {"shape": s, "dtype": "float32"} for k, s in REF.items()}}, open(ref, "w"))
    refcfg = os.path.join(tmp, "ref_config.json")
    json.dump({"model": LM_CFG}, open(refcfg, "w"))
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    json.dump({"model": dict(cfg or LM_CFG, vocab_size=rows)}, open(os.path.join(src, "config.json"), "w"))
    masters = os.path.join(tmp, "masters")
    os.makedirs(masters)
    rng = np.random.default_rng(0)
    keys = {}
    for k, s in REF.items():
        if k == drop:
            continue
        n = rows * D if k in sv.VOCAB_KEYS else s[0]
        full = f"module.{prefix}{k}.main"
        keys[full] = (n, dtype)
        np.save(os.path.join(masters, f"{prefix}{k}.npy"), rng.standard_normal(n).astype(np.float32))
    for k, n in (extra or {"vision_backbone.patch_embed.weight": 12, "connector.proj.weight": 8}).items():
        keys[f"module.{k}.main"] = (n, "float32")
    plan_args = SimpleNamespace(ref=ref, ref_config=refcfg, src=src, out=os.path.join(tmp, "plan.json"),
                                lm_prefix=prefix, lm_config_key="model", mm_tokenizer_size=tok, text_vocab=T)
    apply_args = SimpleNamespace(plan=plan_args.out, masters_in=masters, masters_out=os.path.join(tmp, "out"),
                                 receipt=os.path.join(tmp, "receipt.json"))
    return keys, plan_args, apply_args


def fails(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit as e:
        assert str(e).startswith("STRIP-FAIL"), e
        return str(e)
    raise AssertionError("expected STRIP-FAIL")


def test_vision_keys_dropped_and_appended_rows_sliced():
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, aa = setup(tmp, rows=V + 64, tok=V + 64)
        plan = sv.cmd_plan(pa, keys=keys)
        assert plan["n_kept"] == 5 and plan["n_dropped"] == 2 and plan["vocab_slice"]
        rec = sv.cmd_apply(aa)
        out = np.load(os.path.join(aa.masters_out, "embeddings.weight.npy"))
        src = np.load(os.path.join(aa.masters_in, "llm.embeddings.weight.npy"))
        assert out.size == V * D and np.array_equal(out, src[: V * D])
        assert rec["tensors"]["embeddings.weight"]["dropped_rows"] == 64
        assert json.load(open(os.path.join(aa.masters_out, "config.json")))["model"]["vocab_size"] == V


def test_text_only_vocab_passes_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, aa = setup(tmp)
        assert not sv.cmd_plan(pa, keys=keys)["vocab_slice"]
        rec = sv.cmd_apply(aa)
        for r in rec["tensors"].values():
            assert r["src_sha256"] == r["out_sha256"]


def test_tampered_text_row_fails():
    def bad_writer(path, arr):
        arr = np.array(arr)
        if arr.size > 1000:
            arr[5] += 1.0  # a text row of the embedding changes on the way out
        np.save(path, arr)

    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, aa = setup(tmp, rows=V + 64, tok=V + 64)
        sv.cmd_plan(pa, keys=keys)
        assert "differ" in fails(sv.cmd_apply, aa, writer=bad_writer)
        assert not os.path.exists(aa.masters_out)


def test_missing_lm_key_fails():
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, _ = setup(tmp, drop="blocks.0.attention.w_q.weight")
        assert "missing" in fails(sv.cmd_plan, pa, keys=keys)


def test_extra_lm_like_key_fails():
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, _ = setup(tmp, extra={"llm.blocks.16.attention.w_q.weight": D * D})
        assert "extra LM-like" in fails(sv.cmd_plan, pa, keys=keys)


def test_dtype_and_shape_and_config_fail():
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, _ = setup(tmp, dtype="bfloat16")
        assert "dtype" in fails(sv.cmd_plan, pa, keys=keys)
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, _ = setup(tmp)
        keys["module.llm.embedding_norm.weight.main"] = (D + 1, "float32")
        assert "elements" in fails(sv.cmd_plan, pa, keys=keys)
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, _ = setup(tmp, cfg=dict(LM_CFG, n_layers=2))
        assert "config differs" in fails(sv.cmd_plan, pa, keys=keys)


def test_image_tokens_in_padding_rows_fail_and_tokenizer_required():
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, _ = setup(tmp, tok=T + 10)
        assert "padding rows" in fails(sv.cmd_plan, pa, keys=keys)
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, _ = setup(tmp, tok=None)
        assert "required" in fails(sv.cmd_plan, pa, keys=keys)


def test_no_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        keys, pa, aa = setup(tmp)
        sv.cmd_plan(pa, keys=keys)
        os.makedirs(aa.masters_out)
        assert "exists" in fails(sv.cmd_apply, aa)


def setup_st(tmp, img_token=100280):
    """A weights-only bf16 safetensors export with lm./vision./connector. keys and image_patch_token_id in the pad rows."""
    import torch
    from safetensors.torch import save_file

    ref = os.path.join(tmp, "ref.json")
    json.dump({"keys": {k: {"shape": s, "dtype": "float32"} for k, s in REF.items()}}, open(ref, "w"))
    refcfg = os.path.join(tmp, "ref_config.json")
    json.dump({"model": LM_CFG}, open(refcfg, "w"))
    src = os.path.join(tmp, "src")
    os.makedirs(src)
    lm = dict(LM_CFG, init_seed=7, n_extra_vocab=0, block={"name": "x", "ep": {"share_combine_out": True}})
    refb = dict(LM_CFG, block={"name": "x", "ep": {"share_combine_out": False}})
    json.dump({"model": refb}, open(refcfg, "w"))
    json.dump({"model": {"lm": lm, "vision": {"d": 3}, "image_patch_token_id": img_token}},
              open(os.path.join(src, "olmo_core_config.json"), "w"))
    g = torch.Generator().manual_seed(0)
    t = {}
    for k, s in REF.items():
        shape = (V, D) if k in sv.VOCAB_KEYS else (s[0],)
        t[f"lm.{k}"] = torch.randn(shape, generator=g).to(torch.bfloat16)
    t["vision.patch_embed.weight"] = torch.randn(6, generator=g).to(torch.bfloat16)
    t["connector.projector.w1.weight"] = torch.randn(4, generator=g).to(torch.bfloat16)
    st = os.path.join(src, "model.safetensors")
    save_file(t, st)
    pa = SimpleNamespace(ref=ref, ref_config=refcfg, src=src, out=os.path.join(tmp, "plan.json"), lm_prefix="lm.",
                         lm_config_key="model.lm", mm_tokenizer_size=T, text_vocab=T, src_safetensors=st,
                         src_config=os.path.join(src, "olmo_core_config.json"), allow_bf16_upcast=True,
                         pad_rows="passthrough")
    aa = SimpleNamespace(plan=pa.out, masters_in=None, masters_out=os.path.join(tmp, "out"),
                         receipt=os.path.join(tmp, "receipt.json"))
    return t, pa, aa


def test_safetensors_bf16_passthrough():
    with tempfile.TemporaryDirectory() as tmp:
        t, pa, aa = setup_st(tmp)
        plan = sv.cmd_plan(pa)
        assert plan["n_kept"] == 5 and plan["n_dropped"] == 2 and plan["upcast_bf16_to_fp32"]
        assert plan["pad_token_ids_in_padding_rows"] == [100280] and "block.ep.share_combine_out" in plan["runtime_only_config_diffs"]
        rec = sv.cmd_apply(aa)
        out = np.load(os.path.join(aa.masters_out, "embeddings.weight.npy"))
        assert np.array_equal(out, t["lm.embeddings.weight"].float().numpy().reshape(-1))
        assert rec["pad_rows"]["embeddings.weight"]["first_row"] == T


def test_safetensors_needs_rulings():
    with tempfile.TemporaryDirectory() as tmp:
        _, pa, _ = setup_st(tmp)
        pa.pad_rows = "fail"
        assert "padding rows" in fails(sv.cmd_plan, pa)
    with tempfile.TemporaryDirectory() as tmp:
        _, pa, _ = setup_st(tmp)
        pa.allow_bf16_upcast = False
        assert "dtype" in fails(sv.cmd_plan, pa)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
