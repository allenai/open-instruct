"""Convert an OLMo-core Olmo 3.5 hero checkpoint (OLMoDDP distcp, olmoe3 / MoE-v2 + KDA) to MaxText `olmoe3`.

  convert_olmocore_to_maxtext.py <checkpoint step dir (holds config.json)> <masters dir> <out dir>

Stage 2. <masters dir> holds the fp32 master copies (`module.<param>.main`, stored flattened by
OLMoDDPOptimizer) as .npy files, written by dump_olmocore_masters.py. This restores each
parameter's shape from the run's config.json, and writes MaxText's parameter tree (`params/params/...`) as an
Orbax checkpoint under <out dir>/0/items, the layout `load_parameters_path` expects.

Layouts, as OLMo-core's olmo3moe HF converter documents them:
  routed_experts.w_up_gate (E, 2H, D_latent), up half first;  routed_experts.w_down (E, H, D_latent)
  shared_experts.w_up_gate (D, 2H), up half first;            shared_experts.w_down (H, D)
  routed_experts_router.weight (E, D);  every other projection is torch Linear (out, in).
MaxText `olmoe3` (branch olmo35-rebase): wi_0 is the gate (silu input), wi_1 the up projection; kernels are (in, out);
the full-attention query kernel packs [query | output gate] per head; layers 0-7 sit unscanned under
decoder/layers_0/layer_<i>, layers 8-15 under decoder/scanned_blocks/layer_<i-8> with the scan axis at position 1.
The vocabulary stays at OLMo-core's padded size (its padded rows were trained).
"""

import json
import os
import sys

import numpy as np
import orbax.checkpoint as ocp

src, masters, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(os.path.join(src, "config.json")) as f:
    cfg = json.load(f)["model"]
D, V, L = cfg["d_model"], cfg["vocab_size"], cfg["n_layers"]
blk = cfg["block"]
overrides = {int(k): v for k, v in (cfg.get("block_overrides") or {}).items()}
kda = blk["sequence_mixer"]
H_KDA, HD_KDA = kda["n_heads"], kda["head_dim"]
VD_KDA = int(HD_KDA * kda["expand_v"])
E = blk["routed_experts"]["num_experts"]
H_MOE = blk["routed_experts"]["hidden_size"]
D_LAT = blk["latent_moe"]["latent_dim"]
CYCLE = 8  # MaxText's inhomogeneous_layer_cycle_interval for olmo35


def block_cfg(i):
    return overrides.get(i, blk)


def is_full_attention(i):
    return block_cfg(i)["sequence_mixer"].get("type") == "attention"


p = {
    f[: -len(".npy")]: np.load(os.path.join(masters, f), mmap_mode="r")
    for f in os.listdir(masters)
    if f.endswith(".npy")
}
print(f"loaded {len(p)} tensors, {sum(v.size for v in p.values()) / 1e9:.3f}B params")
used = set()


def get(name, shape):
    used.add(name)
    x = p[name]
    assert x.size == int(np.prod(shape)), (name, x.shape, shape)
    return x.reshape(shape)


def linear_t(name, out_dim, in_dim):  # torch Linear (out, in) -> MaxText kernel (in, out)
    return get(name, (out_dim, in_dim)).T


tree = {
    "token_embedder": {"embedding": get("embeddings.weight", (V, D))},
    "decoder": {
        "embedding_norm": {"scale": get("embedding_norm.weight", (D,))},
        "decoder_norm": {"scale": get("lm_head.norm.weight", (D,))},
        "logits_dense": {"kernel": linear_t("lm_head.w_out.weight", V, D)},
    },
}


def layer(i):
    b = f"blocks.{i}."
    lt = {
        "attn_in_norm": {"scale": get(b + "attention_input_norm.weight", (D,))},
        "attn_out_norm": {"scale": get(b + "attention_norm.weight", (D,))},
        "ffn_in_norm": {"scale": get(b + "feed_forward_input_norm.weight", (D,))},
        "ffn_out_norm": {"scale": get(b + "feed_forward_norm.weight", (D,))},
    }
    a = b + "attention."
    if is_full_attention(i):
        sm = block_cfg(i)["sequence_mixer"]
        nh, nkv, hd = sm["n_heads"], sm["n_kv_heads"], sm["head_dim"]
        wq = linear_t(a + "w_q.weight", nh * hd, D).reshape(D, nh, hd)
        wg = linear_t(a + "w_g.weight", nh * hd, D).reshape(D, nh, hd)
        lt["mixer"] = {
            "attention": {
                "query": {"kernel": np.concatenate([wq, wg], axis=-1)},
                "key": {"kernel": linear_t(a + "w_k.weight", nkv * hd, D).reshape(D, nkv, hd)},
                "value": {"kernel": linear_t(a + "w_v.weight", nkv * hd, D).reshape(D, nkv, hd)},
                "out": {"kernel": linear_t(a + "w_out.weight", D, nh * hd)},
                "query_norm": {"scale": get(a + "q_norm.weight", (nh, hd))},
                "key_norm": {"scale": get(a + "k_norm.weight", (nkv, hd))},
                "ssmax_scale": get(a + "ssmax_scale", (nh,)),
            }
        }
    else:
        qk, vv = H_KDA * HD_KDA, H_KDA * VD_KDA
        lt["mixer"] = {
            "A_log": get(a + "A_log", (H_KDA,)),
            "dt_bias": get(a + "dt_bias", (qk,)),
            "w_q": {"kernel": linear_t(a + "w_q.weight", qk, D)},
            "w_k": {"kernel": linear_t(a + "w_k.weight", qk, D)},
            "w_v": {"kernel": linear_t(a + "w_v.weight", vv, D)},
            "f_proj_1": {"kernel": linear_t(a + "f_proj_1.weight", VD_KDA, D)},
            "f_proj_2": {"kernel": linear_t(a + "f_proj_2.weight", qk, VD_KDA)},
            "w_b": {"kernel": linear_t(a + "w_b.weight", H_KDA, D)},
            "g_proj_1": {"kernel": linear_t(a + "g_proj_1.weight", VD_KDA, D)},
            "g_proj_2": {
                "kernel": linear_t(a + "g_proj_2.weight", vv, VD_KDA),
                "bias": get(a + "g_proj_2.bias", (vv,)),
            },
            "q_conv": get(a + "q_conv1d.weight", (qk, 1, kda["conv_size"]))[:, 0, :],
            "k_conv": get(a + "k_conv1d.weight", (qk, 1, kda["conv_size"]))[:, 0, :],
            "v_conv": get(a + "v_conv1d.weight", (vv, 1, kda["conv_size"]))[:, 0, :],
            "o_norm": {"scale": get(a + "o_norm.weight", (VD_KDA,))},
            "w_out": {"kernel": linear_t(a + "w_out.weight", D, vv)},
        }
    sh = block_cfg(i)["shared_experts"]["hidden_size"]
    up_gate = get(b + "shared_experts.w_up_gate", (D, 2 * sh))
    lt["shared_ffn"] = {
        "wi_0": {"kernel": up_gate[:, sh:]},  # gate
        "wi_1": {"kernel": up_gate[:, :sh]},  # up
        "wo": {"kernel": get(b + "shared_experts.w_down", (sh, D))},
    }
    if block_cfg(i).get("routed_experts") is not None:
        wug = get(b + "routed_experts.w_up_gate", (E, 2 * H_MOE, D_LAT))
        lt["latent_down"] = {"kernel": linear_t(b + "latent_down_proj.weight", D_LAT, D)}
        lt["latent_up"] = {"kernel": linear_t(b + "latent_up_proj.weight", D, D_LAT)}
        lt["moe_block"] = {
            "gate": {"kernel": get(b + "routed_experts_router.weight", (E, D)).T},
            "wi_0": wug[:, H_MOE:, :].transpose(0, 2, 1),  # gate, (E, D_lat, H)
            "wi_1": wug[:, :H_MOE, :].transpose(0, 2, 1),  # up
            "wo": get(b + "routed_experts.w_down", (E, H_MOE, D_LAT)),
        }
    return lt


import jax  # noqa: E402

tree["decoder"]["layers_0"] = {f"layer_{i}": layer(i) for i in range(min(CYCLE, L))}
if L > CYCLE:
    reps = (L - CYCLE) // CYCLE
    assert reps * CYCLE == L - CYCLE, "layers after the first cycle must fill whole cycles"
    scanned = {}
    for k in range(CYCLE):
        per = [layer(CYCLE + r * CYCLE + k) for r in range(reps)]
        scanned[f"layer_{k}"] = jax.tree.map(lambda *xs: np.stack(xs, axis=1), *per)
    tree["decoder"]["scanned_blocks"] = scanned

missing = sorted(set(p) - used)
assert not missing, f"OLMo-core tensors not converted: {missing[:20]}"
tree = jax.tree.map(lambda x: np.ascontiguousarray(x, dtype=np.float32), tree)
mgr = ocp.CheckpointManager(out, item_names=("items",))
mgr.save(0, args=ocp.args.Composite(items=ocp.args.PyTreeSave({"params": {"params": tree}})))
mgr.wait_until_finished()
print(f"wrote {out}/0/items")
