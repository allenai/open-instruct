"""Why is training slow on long sequences? Time Core fwd+bwd for the same token count at different document lengths.

Mirrors MILES's trainer setup for the hybrid model: Core config `model` subtree with attention backend overridden,
fp32 parameters, bf16 autocast, full activation checkpointing, packed documents via doc_lens/max_doc_lens.
Loss is a logsumexp over the logits (stands in for the log-prob computation). Single GPU, random weights.

Run in the MILES image from /opt/core-rl: python bench_long_seq.py CORE_CONFIG_JSON --out OUT.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

CATEGORIES = [
    ("gdn_chunk", ("chunk_gated_delta", "chunk_fwd", "chunk_bwd", "kkt", "wy_", "solve_tril", "fwd_h", "bwd_dh", "bwd_dqkwg", "bwd_dv", "chunk_local_cumsum", "chunk_scaled_dot")),
    ("gdn_other", ("l2norm", "fused_recurrent", "gated", "rmsnorm_gated", "fla")),
    ("conv1d", ("conv1d", "causal_conv")),
    ("attention", ("flash", "fmha", "attn", "cute")),
    ("gemm", ("gemm", "cutlass", "xmma", "nvjet", "matmul", "sm90", "sm100")),
    ("softmax_loss", ("softmax", "logsumexp", "cross_entropy")),
]


def categorize(name):
    lowered = name.lower()
    for category, keys in CATEGORIES:
        if any(k in lowered for k in keys):
            return category
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("core_config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--doc-lens", type=int, nargs="+", default=[1024, 4096, 16384, 32768])
    parser.add_argument("--backend", default="flash_4")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--profile", type=int, nargs="*", default=[1024, 32768], help="doc lengths to kernel-profile")
    parser.add_argument("--no-ac", action="store_true")
    args = parser.parse_args()

    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode

    from miles.backends.core_utils import hybrid_models

    document = json.loads(args.core_config.read_text())
    model_dict = document.get("model", document)
    hybrid_models._override_attention_backend(model_dict, args.backend)
    config = TransformerConfig.from_dict(model_dict)
    config.lm_head.loss_implementation = LMLossImplementation.default
    model = config.build(init_device="cuda")
    model.init_weights(device=torch.device("cuda"))
    model = model.float()
    if not args.no_ac:
        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    vocab = config.vocab_size
    torch.manual_seed(0)
    results = {"backend": args.backend, "tokens": args.tokens, "activation_checkpointing": not args.no_ac, "runs": []}

    def step(doc_len):
        n = args.tokens // doc_len
        ids = torch.randint(0, vocab, (1, n * doc_len), device="cuda")
        doc_lens = torch.tensor([[doc_len] * n], device="cuda", dtype=torch.int32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, doc_lens=doc_lens, max_doc_lens=[doc_len])
        loss = logits.float().logsumexp(-1).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)

    for doc_len in args.doc_lens:
        for _ in range(2):  # warm-up: JIT compiles and autotuning for this shape
            step(doc_len)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            step(doc_len)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        run = {"doc_len": doc_len, "docs": args.tokens // doc_len, "seconds": times, "median_s": sorted(times)[len(times) // 2],
               "tok_per_s": round(args.tokens / sorted(times)[len(times) // 2]), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        if doc_len in args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                step(doc_len)
                torch.cuda.synchronize()
            by_cat, by_kernel = defaultdict(float), defaultdict(float)
            for evt in prof.key_averages():
                cuda_us = getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)
                if cuda_us:
                    by_cat[categorize(evt.key)] += cuda_us / 1e6
                    by_kernel[evt.key] += cuda_us / 1e6
            run["kernel_seconds_by_category"] = {k: round(v, 3) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])}
            run["top_kernels"] = [(k[:120], round(v, 3)) for k, v in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:15]]
        results["runs"].append(run)
        print("BENCH " + json.dumps({k: v for k, v in run.items() if k != "top_kernels"}), flush=True)
        torch.cuda.reset_peak_memory_stats()
    args.out.write_text(json.dumps(results, indent=1))
    print("BENCH_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
