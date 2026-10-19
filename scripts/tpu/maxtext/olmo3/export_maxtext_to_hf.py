"""Export a MaxText `olmo3-7b` parameter checkpoint to HF `Olmo3ForCausalLM` safetensors, without MaxText or a TPU.

  export_maxtext_to_hf.py <params dir> <reference HF dir> <out dir> [--check-equal]

<params dir> is either to_maxtext's layout (…/0/items, tree under params/params) or a Tunix SFT checkpoint's
model_params (leaves wrapped as {"value": …}). <reference HF dir> is the base model (e.g. Olmo-3-1025-7B): its tensor
names and shapes are the target and its config.json is copied. Tokenizer and generation config are the caller's to
add. MaxText's layout: the 32 layers are scanned as 4 layer kinds (decoder/layers/layers_<k>, k = i % 4) with the
scan axis (r = i // 4) at position 1; kernels are (in, out) with heads split out. --check-equal requires every
exported tensor to equal the reference's bit for bit (a conversion round trip on the untrained base).
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
from safetensors.torch import load_file, save_file

src, ref, out = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
check_equal = "--check-equal" in sys.argv[4:]
if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
    raise ValueError("out must be a new or empty directory; do not mix exports")

ckpt = ocp.PyTreeCheckpointer()
meta = ckpt.metadata(src)
meta = meta.item_metadata.tree if hasattr(meta, "item_metadata") else meta.tree
state = ckpt.restore(src, restore_args=jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta))
p = state["params"]["params"] if "params" in state else state
is_wrapped = lambda x: isinstance(x, dict) and set(x) == {"value"}  # noqa: E731
p = jax.tree.map(lambda x: x["value"] if is_wrapped(x) else x, p, is_leaf=is_wrapped)


def drop_rngs(t):  # Tunix also saves the modules' RNG streams (attention/rngs/*/{key,count})
    return {k: drop_rngs(v) for k, v in t.items() if k != "rngs"} if isinstance(t, dict) else t


p = drop_rngs(p)

with open(os.path.join(ref, "model.safetensors.index.json")) as f:
    reference_shards = sorted(set(json.load(f)["weight_map"].values()))
target = {}
for shard in reference_shards:
    with open(os.path.join(ref, shard), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        target.update({k: v["shape"] for k, v in json.loads(f.read(n)).items() if k != "__metadata__"})
with open(os.path.join(ref, "config.json")) as f:
    cfg = json.load(f)
V, L = cfg["vocab_size"], cfg["num_hidden_layers"]
kinds = len(p["decoder"]["layers"])

T = lambda x: np.ascontiguousarray(np.asarray(x).T)  # noqa: E731
hf = {
    "model.embed_tokens.weight": p["token_embedder"]["embedding"][:V],
    "model.norm.weight": p["decoder"]["decoder_norm"]["scale"],
    "lm_head.weight": T(p["decoder"]["logits_dense"]["kernel"][:, :V]),
}
for i in range(L):
    lp = jax.tree.map(lambda x, r=i // kinds: np.take(x, r, axis=1), p["decoder"]["layers"][f"layers_{i % kinds}"])
    b, at = f"model.layers.{i}.", lp["attention"]
    D = at["query"]["kernel"].shape[0]
    hf[b + "self_attn.q_proj.weight"] = T(at["query"]["kernel"].reshape(D, -1))
    hf[b + "self_attn.k_proj.weight"] = T(at["key"]["kernel"].reshape(D, -1))
    hf[b + "self_attn.v_proj.weight"] = T(at["value"]["kernel"].reshape(D, -1))
    hf[b + "self_attn.o_proj.weight"] = T(at["out"]["kernel"].reshape(-1, D))
    hf[b + "self_attn.q_norm.weight"] = at["query_norm"]["scale"]
    hf[b + "self_attn.k_norm.weight"] = at["key_norm"]["scale"]
    hf[b + "post_attention_layernorm.weight"] = lp["post_self_attention_layer_norm"]["scale"]
    hf[b + "post_feedforward_layernorm.weight"] = lp["post_mlp_layer_norm"]["scale"]
    hf[b + "mlp.gate_proj.weight"] = T(lp["mlp"]["wi_0"]["kernel"])
    hf[b + "mlp.up_proj.weight"] = T(lp["mlp"]["wi_1"]["kernel"])
    hf[b + "mlp.down_proj.weight"] = T(lp["mlp"]["wo"]["kernel"])

assert set(hf) == set(target), (sorted(set(target) - set(hf))[:10], sorted(set(hf) - set(target))[:10])
bad = [(k, list(np.shape(v)), target[k]) for k, v in hf.items() if list(np.shape(v)) != target[k]]
assert not bad, bad[:10]
tensors = {
    k: torch.from_numpy(np.ascontiguousarray(np.asarray(v, np.float32))).to(torch.bfloat16) for k, v in hf.items()
}
if check_equal:
    diff = []
    for shard in reference_shards:
        for k, v in load_file(os.path.join(ref, shard)).items():
            if not torch.equal(v, tensors[k]):
                diff.append(k)
    assert not diff, f"{len(diff)} tensors differ from the reference, e.g. {diff[:5]}"
    print(f"round trip: all {len(tensors)} tensors equal the reference bit for bit")
os.makedirs(out, exist_ok=True)
save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
shutil.copy(os.path.join(ref, "config.json"), os.path.join(out, "config.json"))
print(f"wrote {len(tensors)} tensors to {out}")
