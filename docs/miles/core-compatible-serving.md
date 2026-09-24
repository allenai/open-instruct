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

[core]
attention_backend = "torch"
```

Merge the keys into existing tables rather than defining duplicate TOML tables.
Use the usual [MILES plan, validate and launch workflow](launching.md). The
trainer continues to use Core. This mode currently requires unquantized BF16,
TP1/EP1 serving and bias-free, no-RoPE full/KDA attention. Other geometries,
speculative decoding and CUDA graphs are rejected explicitly.

The numerical reference uses Core's Torch attention backend. Remove a conflicting
`trainer_flash_attention_version` override when using this recipe. This does not
claim exact agreement with the historical SFT run's Flash4 attention or distributed
expert execution; other trainer backends need a separate comparison.

## What changes

Expert execution uses persistent Core weight layouts, Torch grouped GEMMs,
separate BF16 SiLU/multiply and down-projection results, and FP32 weighted
unpermutation. Dense/shared MLPs use Core's packed layout. RMS normalization uses
FP32 intermediates. Full attention uses separate Q/K/V projections and Torch
SDPA with explicit repeated KV heads, reading SGLang's ordinary request/cache
mapping. KDA prefill dispatches each request through FLA with Core's `[K,V]`
state orientation, then converts final/intermediate states to SGLang's `[V,K]`
cache format. Cached prefixes retain their state; packed recurrent decode remains
available. Both KDA dispatch and state orientation affect long-prefix fidelity.

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

The [checkpoint/system comparison and workload measurement](measurements/core-compatible-serving-20260923.md)
separates historical checkpoints, short diagnostic interventions, full-prefix
scoring and cached-generation probability errors, with measured generation timing.

On the measured 16-prompt H100 workload, this mode nearly eliminates full-prefix
scoring differences but improves mean cached-generation error by only 3–8%, while
costing about 6.5–6.6× the generation time of ordinary graph-enabled serving. Keep
it as an experimental numerical-reference option; the measurement does not
justify enabling it by default for RL. The qualified immutable image is
`01M38F6YS8W767GHG7WSW5EX1W` (application `4d9d46a95`, serving `c33ed68`).

## Graph-compatible rounding and norms

Set `OLMO_SGLANG_CORE_COMPAT = "rounding"` in `[launch.env]` to opt into the
intermediate arithmetic mode. Set `sglang_cuda_graph_backend_decode = "full"`
and retain disabled prefill graphs in `[inference]`. Use unquantized BF16,
TP1/EP1 and the Triton MoE backend. This mode retains ordinary attention,
KDA dispatch and expert weight layouts; it changes BF16 activation/down-output
rounding, FP32 expert combination and RMS norms. The existing `1`/`full` mode
continues to select the eager reference. Neither mode is enabled by default.

The workload benchmark accepts `--mode rounding_graphs` and `--mode rounding`
for paired graph/eager checks. Qualify the actual graph replay, cached rollouts
and weight refresh before selecting this experimental mode for RL.

The [original tensor implementation](measurements/graph-compatible-rounding-20260923.md)
lost 24–25% throughput. Its arithmetic is retained as a diagnostic control with
`OLMO_SGLANG_ROUNDING_KERNELS = "torch"` in `[launch.env]`.

The [fused implementation and workload comparison](measurements/fused-rounding-20260923.md)
removes the separate activation, expert-combination and norm intermediates while
preserving the BF16 boundaries and the pinned PyTorch reduction grouping. It is
the default implementation **within opt-in rounding mode**; ordinary serving
remains the default overall. The full eager reference is unchanged.

On the measured H100 workload, fused rounding ran within a few percent of default
serving at batches four and sixteen. It matched tensor rounding's tokens and
log-probabilities exactly across 73,728 generated tokens and 8,192 fixed-prefix
scores, recovering the earlier throughput penalty.

Use image `01M38NY00GWVJ7WH26S3V94C9Z` (application `7ef67f07e`, serving
`cc78529`). Its strict component/replay suite passed 105 tests and its engine
cache/full-weight-refresh exercise passed. Matching the tensor rounding control
does not imply exact agreement between cached serving probabilities and Core's
teacher-forced scores or establish an RL learning benefit. Consult the measurement
for throughput, probability comparisons and qualified shapes; repeat qualification
after runtime changes.
