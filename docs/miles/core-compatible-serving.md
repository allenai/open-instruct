# Optional Core-compatible serving

`OLMO_SGLANG_CORE_COMPAT=1` selects an experimental SGLang execution mode that
follows OLMo-core's BF16 rounding, expert layouts, normalization and full-attention
arithmetic. It is **off by default**. This is a serving flag; the separate
`OLMO_HF_MOE_CORE_REFERENCE` flag affects only the exported Transformers model.

Use an image built from this branch's runtime lock, which includes the compatible
olmo-sglang implementation. Setting the variable on an older image does not add
the implementation. Add these settings to an existing run under ignored `runs/`:

```toml
[launch.env]
OLMO_SGLANG_CORE_COMPAT = "1"

[inference]
rollout_tensor_parallel_size = 1
sglang_cuda_graph_backend_decode = "disabled"
sglang_cuda_graph_backend_prefill = "disabled"
```

Merge the keys into existing tables rather than defining duplicate TOML tables.
Use the usual [MILES plan, validate and launch workflow](launching.md). The
trainer continues to use Core. This mode currently requires unquantized BF16,
TP1/EP1 serving and bias-free, no-RoPE full/KDA attention. Other geometries,
speculative decoding and CUDA graphs are rejected explicitly.

## What changes

Expert execution uses persistent Core weight layouts, Torch grouped GEMMs,
separate BF16 SiLU/multiply and down-projection results, and FP32 weighted
unpermutation. Dense/shared MLPs use Core's packed layout. RMS normalization uses
FP32 intermediates. Full attention uses separate Q/K/V projections and Torch
SDPA with explicit repeated KV heads, reading SGLang's ordinary request/cache
mapping. KDA retains the existing FLA chunk and recurrent implementations.

The installed Transformer Engine 2.17 index permutation sorts on the CUDA default
stream. The adapter orders that operation with SGLang's model stream and records
tensor lifetimes across the handoff. This avoids relying on global CUDA
synchronization. Weight publication updates the tensors used for computation;
there are no derived weight caches to refresh.

This mode does not promise bitwise equality across cached decoding, different
batch sizes or chunk boundaries. It also gives up inference kernel fusion and
CUDA graphs. Measure both probability differences and warmed generation
throughput before selecting it for a workload; numerical agreement alone does
not establish an RL learning benefit.

## Reproducible workload comparison

[`benchmark_core_compat.py`](../../scripts/miles/benchmark_core_compat.py) freezes
real RL prompt identities/token IDs, runs isolated serving processes with mode
off/on, and scores each retained rollout through actual Core. A third
`default_graphs` arm measures ordinary serving with full decode graphs. Timing
excludes checkpoint loading, warm-up and Core scoring. Shared forced trajectories
separate checkpoint comparisons from differences in sampled continuations.

The benchmark reports generated-token log-probability errors and the distribution
of `p_Core / p_serving`, including fractions outside 10% and 20%. Its fixed output
budget ignores EOS for equal work; this is a throughput and probability study,
not an evaluation of completed answers or learning quality.
