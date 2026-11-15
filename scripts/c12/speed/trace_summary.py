"""Summarize a torch-profiler chrome trace of training steps: GPU time by kernel category.

  python trace_summary.py TRACE.json[.gz] [--top 40]

Reports, over the profiled window on the traced GPU:
  - wall time (first kernel start to last kernel end) and per-ProfilerStep wall times,
  - GPU busy (union over streams), idle, and communication not overlapped by compute ("exposed comm"),
  - summed kernel time per category (kernels on different streams can overlap, so sums can exceed wall),
  - the split by launching thread: the main thread launches forward, loss and optimizer kernels; the autograd
    thread launches backward kernels, which include activation-checkpoint recompute,
  - the top kernels by summed time.
"""

import argparse
import collections
import gzip
import json
import re

CATEGORIES = [
    ("comm:nccl", re.compile(r"nccl", re.I)),
    ("comm:nvshmem/symm", re.compile(r"nvshmem|symm|deep_?ep|rowwise_(get|put)|_ep_|dispatch|combine", re.I)),
    ("attn:flash", re.compile(r"flash|fmha", re.I)),
    ("kda/fla", re.compile(r"kda|chunk_|fla_|gated_delta|wy_|recurrent|l2norm|tilelang|kernel_kernel|solve_tril|causal_conv|conv1d", re.I)),
    ("moe:grouped_gemm", re.compile(r"group|gmm|_grouped_mm|ptr_array|batched", re.I)),
    ("moe:permute/route", re.compile(r"permut|scatter|gather|index_|sort|topk|histc|bincount|cumsum|moe|router|radix", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|nvjet|cublas|sm\d+_|matmul|wgmma|ampere|hopper", re.I)),
    ("loss/lm_head", re.compile(r"cross_entropy|liger|logsumexp|fused_linear|lse", re.I)),
    ("optimizer", re.compile(r"adam|multi_tensor|foreach|_amp_|lerp", re.I)),
    ("norm", re.compile(r"norm|rms", re.I)),
    ("memcpy/memset", re.compile(r"memcpy|memset|Memcpy|Memset", re.I)),
    ("elementwise/reduce", re.compile(r"elementwise|vectorized|reduce|unrolled|triton|copy|cat|fill|mul|add", re.I)),
]


def category(name: str) -> str:
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other"


def union_length(intervals):
    total = 0.0
    end = None
    start = None
    for s, e in sorted(intervals):
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    if end is not None:
        total += end - start
    return total


def subtract_length(a, b):
    """Length of union(a) minus union(b)."""

    def merge(iv):
        out = []
        for s, e in sorted(iv):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out

    a, b = merge(a), merge(b)
    total, j = 0.0, 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                total += b[k][0] - cur
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            total += e - cur
    return total


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def summarize(path, top=40, out=print):
    trace = load(path)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
    if not kernels:
        out(f"{path}: no kernel events")
        return
    runtime = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "args" in e and "correlation" in e["args"]:
            runtime[e["args"]["correlation"]] = e.get("tid")
    steps = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"], e.get("tid"))
        for e in events
        if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep#")
    )
    main_tid = steps[0][3] if steps else None
    t0 = min(k["ts"] for k in kernels)
    t1 = max(k["ts"] + k["dur"] for k in kernels)
    wall = t1 - t0
    by_cat = collections.Counter()
    by_name = collections.Counter()
    count_name = collections.Counter()
    by_thread = collections.Counter()
    by_thread_cat = collections.defaultdict(collections.Counter)
    comm_iv, comp_iv, all_iv = [], [], []
    for k in kernels:
        name = k.get("name", "?")
        cat = category(name)
        d = k["dur"]
        by_cat[cat] += d
        by_name[name[:110]] += d
        count_name[name[:110]] += 1
        iv = (k["ts"], k["ts"] + d)
        all_iv.append(iv)
        (comm_iv if cat.startswith("comm") else comp_iv).append(iv)
        tid = runtime.get(k.get("args", {}).get("correlation"))
        thread = "main(fwd/loss/optim)" if tid == main_tid else "autograd(bwd+recompute)" if tid is not None else "unknown"
        by_thread[thread] += d
        by_thread_cat[thread][cat] += d
    busy = union_length(all_iv)
    exposed = subtract_length(comm_iv, comp_iv)
    ms = lambda us: f"{us / 1000:9.1f} ms"
    out(f"== trace {path}")
    out(f"profiled steps: {[(n, round((e - s) / 1e6, 3)) for s, e, n, _ in steps]} (CPU-side wall, s)")
    out(f"GPU window {ms(wall)}; busy {ms(busy)} ({100 * busy / wall:.1f}%); idle {ms(wall - busy)}; "
        f"comm union {ms(union_length(comm_iv))}; exposed comm {ms(exposed)} ({100 * exposed / wall:.1f}% of window)")
    out("summed kernel time by category (share of window):")
    for cat, d in by_cat.most_common():
        out(f"  {cat:24s} {ms(d)}  {100 * d / wall:5.1f}%")
    out("by launching thread:")
    for th, d in by_thread.most_common():
        cats = ", ".join(f"{c} {v / 1000:.0f}" for c, v in by_thread_cat[th].most_common(6))
        out(f"  {th:26s} {ms(d)}  {100 * d / wall:5.1f}%   [{cats} ms]")
    out(f"top {top} kernels by summed time:")
    for name, d in by_name.most_common(top):
        out(f"  {ms(d)} x{count_name[name]:6d}  [{category(name)}] {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    summarize(a.trace, a.top)
