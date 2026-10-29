"""Export a MaxText `olmoe3` (Olmo 3.5) parameter checkpoint to HF `Olmo3MoeForCausalLM` safetensors.

  export_maxtext_to_hf.py <params dir> <reference HF dir> <out dir>

<params dir> is either to_maxtext's layout (…/0/items, tree under params/params) or a Tunix SFT checkpoint's
model_params (leaves wrapped as {"value": …}). <reference HF dir> is an OLMo-core HF export of the same
architecture (e.g. H045's hf_step11768): its tensor names and shapes are the target, and its config.json, modeling
code, tokenizer and chat template are copied. The inverse of convert_olmocore_to_maxtext.py composed with
OLMo-core's olmo3moe HF converter: kernels (in, out) -> torch Linear (out, in); the full-attention query kernel
splits into q_proj | g_proj per head; wi_0 is the gate, wi_1 the up projection; layers 0-7 are unscanned, 8-15 sit
in scanned_blocks with the scan axis at position 1; the padded vocabulary is trimmed to the reference's.
"""

import json
import os
import shutil
import struct
import sys

import jax
import numpy as np
import torch
from orbax import checkpoint as ocp
from safetensors.torch import save_file

src, ref, out = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
    raise ValueError("out must be a new or empty directory; do not mix exports")
CYCLE = 8

ckpt = ocp.PyTreeCheckpointer()
meta = ckpt.metadata(src)
meta = meta.item_metadata.tree if hasattr(meta, "item_metadata") else meta.tree
state = ckpt.restore(src, restore_args=jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta))
p = state["params"]["params"] if "params" in state else state
is_wrapped = lambda x: isinstance(x, dict) and set(x) == {"value"}  # noqa: E731
p = jax.tree.map(lambda x: x["value"] if is_wrapped(x) else x, p, is_leaf=is_wrapped)


def drop_rngs(t):  # Tunix also saves the modules' RNG streams (moe_block/rngs/*/{key,count})
    return {k: drop_rngs(v) for k, v in t.items() if k != "rngs"} if isinstance(t, dict) else t


p = drop_rngs(p)

index_path = os.path.join(ref, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path) as f:
        reference_shards = sorted(set(json.load(f)["weight_map"].values()))
else:
    reference_shards = ["model.safetensors"]
target = {}
for shard in reference_shards:
    with open(os.path.join(ref, shard), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        target.update({k: v["shape"] for k, v in json.loads(f.read(n)).items() if k != "__metadata__"})
with open(os.path.join(ref, "config.json")) as f:
    cfg = json.load(f)
V, L = cfg["vocab_size"], cfg["num_hidden_layers"]
NH, HD = cfg["num_attention_heads"], cfg["head_dim"]


def scanned_take(i):
    """Layer i's parameters; scanned leaves carry the scan axis at position 1."""
    if i < CYCLE:
        return p["decoder"]["layers_0"][f"layer_{i}"]
    r, k = divmod(i - CYCLE, CYCLE)
    return jax.tree.map(lambda x: np.take(x, r, axis=1), p["decoder"]["scanned_blocks"][f"layer_{k}"])


T = lambda x: np.ascontiguousarray(np.asarray(x).T)  # noqa: E731
hf = {
    "model.embed_tokens.weight": p["token_embedder"]["embedding"][:V],
    "model.embed_norm.weight": p["decoder"]["embedding_norm"]["scale"],
    "model.norm.weight": p["decoder"]["decoder_norm"]["scale"],
    "lm_head.weight": T(p["decoder"]["logits_dense"]["kernel"][:, :V]),
}
for i in range(L):
    lp, b = scanned_take(i), f"model.layers.{i}."
    hf[b + "pre_attention_layernorm.weight"] = lp["attn_in_norm"]["scale"]
    hf[b + "post_attention_layernorm.weight"] = lp["attn_out_norm"]["scale"]
    hf[b + "pre_feedforward_layernorm.weight"] = lp["ffn_in_norm"]["scale"]
    hf[b + "post_feedforward_layernorm.weight"] = lp["ffn_out_norm"]["scale"]
    a, m = b + "self_attn.", lp["mixer"]
    if cfg["layer_types"][i] == "full_attention":
        at = m["attention"]
        q = at["query"]["kernel"]  # (D, NH, 2*HD): [query | output gate] per head
        D = q.shape[0]
        hf[a + "q_proj.weight"] = T(q[..., :HD].reshape(D, NH * HD))
        hf[a + "g_proj.weight"] = T(q[..., HD:].reshape(D, NH * HD))
        hf[a + "k_proj.weight"] = T(at["key"]["kernel"].reshape(D, -1))
        hf[a + "v_proj.weight"] = T(at["value"]["kernel"].reshape(D, -1))
        hf[a + "o_proj.weight"] = T(at["out"]["kernel"].reshape(-1, D))
        hf[a + "q_norm.weight"] = at["query_norm"]["scale"]
        hf[a + "k_norm.weight"] = at["key_norm"]["scale"]
        hf[a + "ssmax_scale"] = at["ssmax_scale"]
    else:
        hf[a + "A_log"] = m["A_log"]
        hf[a + "dt_bias"] = m["dt_bias"]
        for hn, mn in [
            ("q_proj", "w_q"),
            ("k_proj", "w_k"),
            ("v_proj", "w_v"),
            ("f_proj_1", "f_proj_1"),
            ("f_proj_2", "f_proj_2"),
            ("beta_proj", "w_b"),
            ("g_proj_1", "g_proj_1"),
            ("g_proj_2", "g_proj_2"),
            ("o_proj", "w_out"),
        ]:
            hf[a + hn + ".weight"] = T(m[mn]["kernel"])
        hf[a + "g_proj_2.bias"] = m["g_proj_2"]["bias"]
        for c in "qkv":
            hf[a + f"{c}_conv1d.weight"] = m[f"{c}_conv"][:, None, :]
        hf[a + "o_norm.weight"] = m["o_norm"]["scale"]
    sh = lp["shared_ffn"]
    if "moe_block" in lp:
        mlp = b + "mlp."
        hf[mlp + "shared_expert.gate_proj.weight"] = T(sh["wi_0"]["kernel"])
        hf[mlp + "shared_expert.up_proj.weight"] = T(sh["wi_1"]["kernel"])
        hf[mlp + "shared_expert.down_proj.weight"] = T(sh["wo"]["kernel"])
        hf[mlp + "latent_down_proj.weight"] = T(lp["latent_down"]["kernel"])
        hf[mlp + "latent_up_proj.weight"] = T(lp["latent_up"]["kernel"])
        mb = lp["moe_block"]
        hf[mlp + "router.gate.weight"] = T(mb["gate"]["kernel"])
        for e in range(mb["wi_0"].shape[0]):  # wi_* (E, D_lat, H), wo (E, H, D_lat)
            hf[mlp + f"experts.{e}.gate_proj.weight"] = T(mb["wi_0"][e])
            hf[mlp + f"experts.{e}.up_proj.weight"] = T(mb["wi_1"][e])
            hf[mlp + f"experts.{e}.down_proj.weight"] = T(mb["wo"][e])
    else:  # dense layer: the shared FFN is the whole MLP
        hf[b + "mlp.gate_proj.weight"] = T(sh["wi_0"]["kernel"])
        hf[b + "mlp.up_proj.weight"] = T(sh["wi_1"]["kernel"])
        hf[b + "mlp.down_proj.weight"] = T(sh["wo"]["kernel"])

assert set(hf) == set(target), (sorted(set(target) - set(hf))[:10], sorted(set(hf) - set(target))[:10])
bad = [(k, list(np.shape(v)), target[k]) for k, v in hf.items() if list(np.shape(v)) != target[k]]
assert not bad, bad[:10]
os.makedirs(out, exist_ok=True)
tensors = {
    k: torch.from_numpy(np.ascontiguousarray(np.asarray(v, np.float32))).to(torch.bfloat16) for k, v in hf.items()
}
save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(ref):
    # Copy model/tokenizer metadata, never the reference's weights or an index pointing to them.
    if f.endswith((".safetensors", ".bin", ".safetensors.index.json", ".bin.index.json")):
        continue
    if os.path.isfile(os.path.join(ref, f)):
        shutil.copy(os.path.join(ref, f), os.path.join(out, f))
    elif f == "chat_templates":
        shutil.copytree(os.path.join(ref, f), os.path.join(out, f))
print(f"wrote {len(tensors)} tensors to {out}")
