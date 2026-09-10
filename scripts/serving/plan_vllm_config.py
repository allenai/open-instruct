"""Derive an optimal vLLM serving configuration for a large MoE model on B300.

Answers, for any HuggingFace repo id, the question we otherwise spend hours of
research on: how many GPUs, which parallelism strategy, what memory footprint,
and what output throughput should we expect from memory bandwidth.

Everything is computed from the model's own config.json geometry, cross-checked
against the parameter census the HuggingFace API reports for the safetensors
shards. When those two disagree by more than a few percent the geometry model is
missing something (an MTP head, a vision tower, tied embeddings) and the result
is flagged rather than quietly trusted.

Usage:
    uv run --no-project --with huggingface_hub python \\
        scripts/serving/plan_vllm_config.py zai-org/GLM-5.2-FP8
    ... --gpu b200 --context 131072 --target-concurrency 256
"""

import argparse
import json
import urllib.request

# --- hardware -----------------------------------------------------------------
# Capacity for B300 is genuinely ambiguous in NVIDIA's own materials (262.5 GiB
# for HGX B300, 277.8 for GB300 NVL72, 288 for DGX B300). We use what the device
# actually reports on our nodes -- nvidia-smi says 275040 MiB -- rather than a
# marketing figure, and treat the rest as headroom.
GPUS = {
    # name: (bytes_hbm, bytes_per_sec, native numeric formats for tensor cores)
    "b300": (275040 * 1024**2, 8.0e12, {"bf16", "fp16", "fp8", "fp6", "fp4", "nvfp4", "mxfp4"}),
    "b200": (180 * 1000**3, 8.0e12, {"bf16", "fp16", "fp8", "fp6", "fp4", "nvfp4", "mxfp4"}),
    "h100": (80 * 1000**3, 3.35e12, {"bf16", "fp16", "fp8", "int8"}),
}

# INT4 has no tensor-core path on Hopper or Blackwell: it was dropped after
# Ampere and did not return. vLLM serves such checkpoints through Marlin, which
# dequantizes to BF16 in-register and runs BF16 MMA -- a footprint and bandwidth
# win with no FLOPs win, and a compute loss at large batch. Any checkpoint whose
# weights are INT4 therefore gets costed at BF16 compute.
# Roofline efficiency, and the band we actually trust.
#
# Calibrated against our own measurements on 4x B300, TP=4, 128K context,
# concurrency 256, vLLM 0.28 (1000 prompts x 8 samples of real reasoning traces):
#
#   model              predicted   measured   ratio
#   DeepSeek-V3.2-Exp      2092       2124     0.98x
#   Kimi-K2.6 (INT4)       1310        814     1.61x
#   Qwen3.5-397B-FP8       8295       4687     1.77x
#
# A single constant cannot absorb a 0.98-1.77x spread, so the tool reports a
# BAND rather than a point estimate. The model is most accurate when KV traffic
# dominates the decode step (DeepSeek) and over-predicts when weight traffic
# dominates (Qwen), which is the regime where MoE expert-routing locality,
# kernel launch overhead and all-reduce cost are least well captured.
EFFICIENCY = 0.55
OVERPREDICTION_RANGE = (1.0, 1.8)

NO_NATIVE_TENSOR_CORE = {"int4", "uint4", "int4_packed"}


def fetch_json(url: str, timeout: int = 45):
    return json.load(urllib.request.urlopen(url, timeout=timeout))


def model_config(repo: str) -> dict:
    cfg = fetch_json(f"https://huggingface.co/{repo}/resolve/main/config.json")
    # Multimodal checkpoints nest the language model; the serving math only cares
    # about the text stack.
    return cfg.get("text_config") or cfg, cfg


def census(repo: str) -> dict:
    """Parameter count by dtype, as reported for the repo's safetensors shards."""
    d = fetch_json(f"https://huggingface.co/api/models/{repo}")
    st = d.get("safetensors") or {}
    return {"total": st.get("total"), "by_dtype": st.get("parameters") or {}}


def weight_bytes_per_param(by_dtype: dict) -> float:
    """Average bytes per parameter, from the actual dtype mix in the checkpoint.

    Checkpoints are rarely uniform: an "FP8" repo can hold hundreds of BF16
    modules, and a 4-bit repo stores packed values in INT32 containers with
    separate scales. Averaging over the real mix avoids trusting the repo name.
    """
    width = {
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I32": 0.5,  # 4-bit values packed 8-per-int32; ~0.5 B/param plus scales
        "I8": 1,
        "U8": 1,
        "I64": 8,
        "I16": 2,
    }
    tot = sum(by_dtype.values()) or 1
    return sum(width.get(k, 2) * v for k, v in by_dtype.items()) / tot


def attention_layers(t: dict) -> int:
    """Layers that actually hold a KV cache.

    Hybrid models (Qwen3.5's Gated DeltaNet, Olmo-Hybrid, Jamba) interleave
    linear-attention layers that keep a fixed-size recurrent state instead of a
    growing KV cache. Counting all layers overstates KV per token several-fold:
    Qwen3.5-397B has 60 layers but only 15 full-attention ones.
    """
    lt = t.get("layer_types")
    if lt:
        n = sum(1 for x in lt if "full" in str(x))
        if n:
            return n
    interval = t.get("full_attention_interval")
    if interval:
        return max(1, t["num_hidden_layers"] // interval)
    return t["num_hidden_layers"]


def recurrent_state_bytes(t: dict) -> float:
    """Fixed per-sequence state for linear-attention layers, if any.

    Independent of context length, so it is a floor on per-sequence memory that
    KV-only accounting misses entirely.
    """
    nv = t.get("linear_num_value_heads")
    if not nv:
        return 0.0
    kd = t.get("linear_key_head_dim", 128)
    vd = t.get("linear_value_head_dim", 128)
    lt = t.get("layer_types") or []
    n_lin = sum(1 for x in lt if "linear" in str(x)) or (t["num_hidden_layers"] - attention_layers(t))
    width = 4 if str(t.get("mamba_ssm_dtype", "float32")).endswith("32") else 2
    return n_lin * nv * kd * vd * width


def kv_bytes_per_token(t: dict, kv_dtype: str) -> tuple[float, str]:
    """KV cache bytes per token, computed from attention geometry.

    MLA (DeepSeek/GLM/Kimi lineage) caches a compressed latent plus a RoPE part,
    NOT per-head K and V. Under FP8 vLLM also stores per-block FP32 scales, and
    keeps the RoPE portion in BF16 -- so the FP8 saving is real but smaller than
    halving. Getting this wrong overstates concurrency by double digits.
    """
    layers = attention_layers(t)
    lora = t.get("kv_lora_rank")
    if lora:
        rope = t.get("qk_rope_head_dim", 64)
        # fp8 latent + fp32 block scales + bf16 rope; NOT simply half of bf16
        per_layer = lora + 16 + rope * 2 if kv_dtype == "fp8" else (lora + rope) * 2
        return per_layer * layers, "MLA (compressed latent)"
    kvh = t.get("num_key_value_heads") or t["num_attention_heads"]
    hd = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    elem = 1 if kv_dtype == "fp8" else 2
    return kvh * hd * 2 * elem * layers, "GQA/MHA"


def attention_params(t: dict) -> float:
    """Attention parameters per layer, for MLA or GQA/MHA."""
    h = t["hidden_size"]
    nh = t["num_attention_heads"]
    lora = t.get("kv_lora_rank")
    if lora:
        qk_nope = t.get("qk_nope_head_dim", 128)
        qk_rope = t.get("qk_rope_head_dim", 64)
        v_head = t.get("v_head_dim", qk_nope)
        q_lora = t.get("q_lora_rank")
        q = (h * q_lora + q_lora * nh * (qk_nope + qk_rope)) if q_lora else h * nh * (qk_nope + qk_rope)
        kv = h * (lora + qk_rope) + lora * nh * (qk_nope + v_head)
        return q + kv + nh * v_head * h
    hd = t.get("head_dim") or h // nh
    kvh = t.get("num_key_value_heads") or nh
    return h * nh * hd + 2 * h * kvh * hd + nh * hd * h


def param_breakdown(t: dict) -> dict:
    """Parameters by role, computed from architecture.

    Deliberately NOT derived by subtracting experts from the checkpoint's total:
    that folds MTP heads, vision towers and tied-embedding differences into the
    dense stack and silently inflates the active-parameter count.
    """
    h = t["hidden_size"]
    layers = t["num_hidden_layers"]
    # Hybrid/MoE-only configs may omit intermediate_size entirely.
    inter = (
        t.get("intermediate_size") or t.get("shared_expert_intermediate_size") or t.get("moe_intermediate_size") or 0
    )
    vocab = t.get("vocab_size", 0)
    ne = t.get("n_routed_experts") or t.get("num_experts") or 0
    mi = t.get("moe_intermediate_size") or inter
    shared = t.get("n_shared_experts") or 0
    topk = t.get("num_experts_per_tok") or 0
    dense_layers = t.get("first_k_dense_replace", 0) if ne else layers
    moe_layers = layers - dense_layers

    embed = vocab * h * 2  # input embedding + lm_head
    attn = layers * attention_params(t)
    dense_mlp = dense_layers * 3 * h * inter
    one_expert = 3 * h * mi
    experts = moe_layers * (ne + shared) * one_expert
    router = moe_layers * ne * h

    non_expert = embed + attn + dense_mlp + router
    active = non_expert + moe_layers * (topk + shared) * one_expert
    return {
        "embed": embed,
        "attn": attn,
        "dense_mlp": dense_mlp,
        "router": router,
        "experts": float(experts),
        "non_expert": float(non_expert),
        "active": float(active),
        "total_geom": float(non_expert + experts),
        "one_expert": one_expert,
        "moe_layers": moe_layers,
        "ne": ne,
        "topk": topk,
    }


def plan(
    repo: str, gpu: str, ctx: int, target_conc: int, util: float, kv_dtype: str, allow_multinode: bool = False
) -> dict:
    t, full = model_config(repo)
    c = census(repo)
    bpp = weight_bytes_per_param(c["by_dtype"])
    total_p = c["total"] or 0
    wbytes = total_p * bpp
    pb = param_breakdown(t)
    act_p = pb["active"]
    kvpt, kvkind = kv_bytes_per_token(t, kv_dtype)
    rec_bytes = recurrent_state_bytes(t)
    attn_layers = attention_layers(t)
    hbm, bw, native = GPUS[gpu]

    quant = ((full.get("quantization_config") or {}).get("quant_method") or "").lower()
    dtypes = set(k.lower() for k in c["by_dtype"])
    int4ish = "i32" in dtypes and ("pack" in quant or "compressed" in quant or bpp < 1.0)
    # INT4 has no tensor-core path: vLLM dequantizes to BF16 and computes there,
    # so cost compute at 2 bytes/param even though storage is ~0.5.
    compute_bpp = 2.0 if int4ish else max(bpp, 1.0)

    rows = []
    max_gpus = 16 if allow_multinode else 8
    for n in (1, 2, 4, 8, 16):
        if n > max_gpus:
            continue
        budget = hbm * util * n
        # Expert parallel shards experts across ranks and replicates the dense
        # stack once per rank. Tensor parallel shards every weight -- but for MLA
        # the KV cache has a single latent head that cannot be split, so TP
        # REPLICATES the cache n times. That replication, not weight size, is
        # usually what limits concurrency on these models.
        ep_total_w = (pb["non_expert"] * bpp) * n + pb["experts"] * bpp
        tp_total_w = wbytes
        for strat, wtot, kv_repl in (
            ("TP", tp_total_w, n if kvkind.startswith("MLA") else 1),
            ("DP+EP", ep_total_w, 1),
        ):
            kv_avail = budget - wtot
            if kv_avail <= 0:
                rows.append((n, strat, wtot, None, None, None))
                continue
            per_seq = kvpt * ctx + rec_bytes
            seqs = (kv_avail / kv_repl) / per_seq
            conc = max(1.0, min(seqs, float(target_conc)))
            b_per_gpu = conc / n
            # Decode is memory-bandwidth bound. Per step each GPU streams the
            # weights it holds -- for MoE only the experts actually routed to,
            # which approaches all of them as batch grows -- plus the KV it must
            # read for its own sequences.
            coverage = 1.0 - (1.0 - pb["topk"] / max(pb["ne"], 1)) ** max(b_per_gpu, 1.0) if pb["ne"] else 1.0
            dense_bytes = pb["non_expert"] * compute_bpp / (1 if strat == "DP+EP" else n)
            expert_bytes = pb["experts"] * compute_bpp * coverage / n
            kv_bytes = b_per_gpu * (ctx / 2) * kvpt
            step_bytes = dense_bytes + expert_bytes + kv_bytes
            step_s = step_bytes / bw
            tps = (b_per_gpu / step_s) * n * EFFICIENCY
            rows.append((n, strat, wtot, seqs, tps, conc))
    return {
        "repo": repo,
        "gpu": gpu,
        "total_params": total_p,
        "active_params": act_p,
        "bytes_per_param": bpp,
        "weights_bytes": wbytes,
        "expert_frac": pb["experts"] / max(pb["total_geom"], 1),
        "geom_total": pb["total_geom"],
        "breakdown": pb,
        "kv_bytes_per_token": kvpt,
        "recurrent_state_bytes": rec_bytes,
        "attn_layers": attn_layers,
        "kv_kind": kvkind,
        "int4_no_native_path": int4ish,
        "quant": quant or "none",
        "dtypes": c["by_dtype"],
        "rows": rows,
        "layers": t["num_hidden_layers"],
        "experts": t.get("n_routed_experts") or t.get("num_experts"),
        "topk": t.get("num_experts_per_tok"),
        "ctx": ctx,
        "kv_dtype": kv_dtype,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo")
    ap.add_argument("--gpu", default="b300", choices=sorted(GPUS))
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--target-concurrency", type=int, default=256)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--kv-dtype", default="fp8", choices=("fp8", "bf16"))
    a = ap.parse_args()

    p = plan(a.repo, a.gpu, a.context, a.target_concurrency, a.gpu_util, a.kv_dtype)
    G = 1000**3
    print(f"\n{p['repo']}  on  {p['gpu'].upper()}   (ctx {p['ctx']}, kv {p['kv_dtype']})")
    print("=" * 78)
    print(f"  total params      {p['total_params'] / 1e9:8.1f} B")
    print(
        f"  active params     {p['active_params'] / 1e9:8.1f} B   ({100 * p['active_params'] / p['total_params']:.1f}%)"
    )
    print(
        f"  bytes/param       {p['bytes_per_param']:8.2f}     dtype mix: "
        f"{ {k: round(v / 1e9, 1) for k, v in p['dtypes'].items()} }"
    )
    print(
        f"  weights           {p['weights_bytes'] / G:8.0f} GB   experts are {100 * p['expert_frac']:.0f}% of params"
    )
    print(f"  layers/experts    {p['layers']} layers, {p['experts']} experts, top-{p['topk']}")
    print(
        f"  KV/token          {p['kv_bytes_per_token'] / 1024:8.1f} KiB  [{p['kv_kind']}, "
        f"{p['attn_layers']}/{p['layers']} layers cache KV]"
    )
    if p["recurrent_state_bytes"]:
        print(
            f"  recurrent state   {p['recurrent_state_bytes'] / 1024**2:8.0f} MiB/sequence "
            f"(fixed, independent of context)"
        )
    if p["int4_no_native_path"]:
        print("  !! INT4 checkpoint: no tensor-core path on Hopper/Blackwell.")
        print("     vLLM dequantizes via Marlin and computes in BF16 -- memory win, no FLOPs win.")
    print()
    b = p["breakdown"]
    print(
        f"  geometry check    {p['geom_total'] / 1e9:8.1f} B computed vs "
        f"{p['total_params'] / 1e9:.1f} B in checkpoint "
        f"({100 * abs(p['geom_total'] - p['total_params']) / max(p['total_params'], 1):.1f}% diff)"
    )
    print(
        f"  dense stack       {b['non_expert'] / 1e9:8.1f} B  (embed {b['embed'] / 1e9:.1f}, "
        f"attn {b['attn'] / 1e9:.1f}, dense-mlp {b['dense_mlp'] / 1e9:.1f}, router {b['router'] / 1e9:.2f})"
    )
    print()
    print(f"  {'GPUs':>4} {'strategy':<8} {'weights':>8} {'GB/GPU':>7} {'seqs@ctx':>9} {'est tok/s':>15}")
    print("  " + "-" * 62)
    best = None
    for n, strat, w, seqs, tps, conc in p["rows"]:
        if seqs is None:
            print(f"  {n:>4} {strat:<8} {w / G:7.0f}G {w / G / n:7.0f} {'no KV room':>9}")
            continue
        lo, hi = tps / OVERPREDICTION_RANGE[1], tps / OVERPREDICTION_RANGE[0]
        print(f"  {n:>4} {strat:<8} {w / G:7.0f}G {w / G / n:7.0f} {seqs:9.0f} {lo:7.0f}-{hi:<7.0f}")
        if seqs >= 16 and (best is None or tps > best[4]):
            best = (n, strat, w, seqs, tps, conc)
    if best:
        n, strat, w, seqs, tps, conc = best
        print()
        print(f"  RECOMMENDED: {n} x {p['gpu'].upper()}, {strat}, kv-cache-dtype={p['kv_dtype']}")
        print(
            f"    weights {w / G:.0f} GB total = {w / G / n:.0f} GB/GPU, leaving "
            f"{(GPUS[p['gpu']][0] * 0.90 - w / n) / G:.0f} GB/GPU for KV"
        )
        print(
            f"    ~{seqs:.0f} concurrent seqs at {p['ctx']} ctx; "
            f"est {tps / OVERPREDICTION_RANGE[1]:.0f}-{tps:.0f} output tok/s"
        )
        if strat == "DP+EP":
            print(f"    --data-parallel-size {n} --enable-expert-parallel --enable-ep-weight-filter \\")
        else:
            print(f"    --tensor-parallel-size {n} \\")
        print(f"    --kv-cache-dtype fp8 --max-model-len {p['ctx']} --async-scheduling")
    print()


if __name__ == "__main__":
    main()
