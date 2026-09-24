# Fused rounding: reference arithmetic at ordinary serving speed

The ordered fused implementation removes the measured 20–25% serving penalty
while matching the earlier tensor rounding path exactly on the measured
workloads. It retains SGLang's tuned expert GEMMs, alignment, weight storage,
attention and KDA execution. Three small kernels fuse activation, weighted
expert combination and RMS normalization. There is no new architecture fork,
global kernel monkeypatch or derived weight cache.

`OLMO_SGLANG_CORE_COMPAT=rounding` remains opt-in and now uses the fused kernels.
`OLMO_SGLANG_ROUNDING_KERNELS=torch` retains the slower control. Ordinary serving
remains the overall default; `OLMO_SGLANG_CORE_COMPAT=1` / `full` retains the eager
full reference.

## Final performance and equivalence

All rows use full decode graphs. Batch size is four except where marked.
Throughput counts warmed generation only, excluding engine loading, graph
capture, warm-up and Core scoring.

| Checkpoint / batch | Default tokens/s | Tensor rounding tokens/s | Ordered fused tokens/s | Fused / default |
|---|---:|---:|---:|---:|
| Hero base | 992.7 | 752.4 | 1010.5 | 1.018× |
| Hero EMO SFT | 988.1 | 750.6 | 1000.4 | 1.012× |
| Hero non-EMO SFT | 982.5 | 746.9 | 1004.2 | 1.022× |
| Hero EMO SFT, batch 16 | 2486.4 | 2042.7 | 2512.8 | 1.011× |

Ordered fusion and tensor rounding produced identical output tokens and exactly
equal reported log-probabilities across **73,728 generated tokens** and **8,192
fixed-prefix token scores**. This covers the three checkpoints at batch four
and EMO at batch 16. The small apparent gains over default should be read as
near-default throughput, not an established universal speedup.

The batch-16 run repeats the default arm after the compatibility arms to check
timing drift. Its measured initial/late throughputs are
**2486.4 / 2490.1 tokens/s**.

## Arithmetic retained inside fusion

1. The activation kernel computes SiLU in FP32, explicitly rounds it to BF16
   and back to FP32, multiplies the up projection, then writes BF16.
2. The existing down GEMM writes **unweighted BF16** outputs. The combination
   kernel multiplies routing weights and accumulates in FP32, then writes BF16.
   It uses the pinned PyTorch sum's four independent accumulators and ordered
   final combination, with FP32 contraction disabled.
3. The norm kernel keeps squares, variance, normalization and gain multiplication
   in FP32. It follows PyTorch's per-thread grouping and cross-warp/intra-warp
   reduction order and uses the matching CUDA reciprocal-square-root primitive.
   Only the final result is rounded to BF16.

Thus the down GEMM and combine still use the same number of stages as ordinary
serving: weighting moves into the already-needed combine stage. Explicit casts
inside activation do not require extra kernel launches or BF16 temporary tensors.
Top-k above 16 and norm widths above 4096, or non-vectorizable widths at least
128, fall back to the tensor implementation. Those larger shapes were not
workload-qualified in this study.

## Final Core probability comparison

"Rollout error" compares cached serving probabilities with actual BF16 Core's
teacher-forced probabilities for the same chosen tokens and causal prefixes.
"Fixed-prefix error" uses four shared, complete continuations and prefill
scoring. Errors are absolute natural-log-probability differences. The outside-20%
fraction measures `p_Core / p_serving` outside `[0.8, 1.2]` at unchanged weights;
it is not an observed PPO clipping rate after an update.

| Checkpoint | Mode | Rollout mean | Rollout p99 | Outside 20% | Fixed-prefix mean | Median batch seconds |
|---|---|---:|---:|---:|---:|---:|
| Hero base | Default | 0.03706 | 0.21258 | 1.282% | 0.03476 | 2.064 |
| Hero base | Tensor rounding | 0.03735 | 0.20911 | 1.062% | 0.03545 | 2.722 |
| Hero base | Ordered fused rounding | 0.03735 | 0.20911 | 1.062% | 0.03545 | 2.024 |
| Hero EMO SFT | Default | 0.04234 | 0.33033 | 3.766% | 0.04258 | 2.075 |
| Hero EMO SFT | Tensor rounding | 0.04151 | 0.32525 | 3.656% | 0.04226 | 2.724 |
| Hero EMO SFT | Ordered fused rounding | 0.04151 | 0.32525 | 3.656% | 0.04226 | 2.048 |
| Hero non-EMO SFT | Default | 0.03497 | 0.27612 | 2.545% | 0.04004 | 2.084 |
| Hero non-EMO SFT | Tensor rounding | 0.03531 | 0.27025 | 2.380% | 0.04100 | 2.737 |
| Hero non-EMO SFT | Ordered fused rounding | 0.03531 | 0.27025 | 2.380% | 0.04100 | 2.040 |
| Hero EMO SFT, batch 16 | Default | 0.04088 | 0.32652 | 3.546% | 0.04258 | 3.296 |
| Hero EMO SFT, batch 16 | Tensor rounding | 0.04286 | 0.33735 | 3.925% | 0.04226 | 4.002 |
| Hero EMO SFT, batch 16 | Ordered fused rounding | 0.04286 | 0.33735 | 3.925% | 0.04226 | 3.260 |

Fusion preserves the slower rounding path's probabilities; it does **not** remove
that path's remaining differences from Core. Mean-error changes versus ordinary
serving remain small and mixed. The [full eager reference](core-compatible-serving-20260923.md)
still has much closer full-prefix scores and a large performance cost. These
measurements do not establish an RL learning-quality benefit.

## Why the first fused candidate was revised

The first candidate preserved BF16 cast locations but used generic Triton
reductions. At 256-token component shapes, it changed 13 of 262,144 weighted
outputs and one of 262,144 norm outputs. Full-model fixed-prefix log-probabilities
nevertheless differed from the tensor control by means of 0.0342–0.0413. Separate
EMO interventions isolated changes from both expert and norm fusion:

| Initial candidate | Tokens/s | Mean fixed-prefix difference from tensor rounding | Mean rollout error versus Core |
|---|---:|---:|---:|
| Tensor control | 751.9 | 0.00000 | 0.04151 |
| Fuse experts only | 780.9 | 0.03729 | 0.04253 |
| Fuse norms only | 945.9 | 0.04149 | 0.04331 |
| Fuse both | 1019.4 | 0.04130 | 0.03955 |

The initial/late default controls in that run measured 984.9 / 979.8 tokens/s.
The evidence shows why close component tolerances were insufficient; it does
not trace every changed route. Preserving the reference's grouping and norm
arithmetic removed the measured full-model differences. The generic first
candidate is retained in immutable source/run artifacts, not selected by the
final implementation.

## Component timings and qualification

Times below use CUDA events around repeated captured operations. They are
microbenchmarks, not an estimate obtained by adding component times together.
Hero dimensions are hidden/expert width 1024 and top-k 16. Every listed ordered
output matched its tensor control exactly.

| Tokens | Operation | Tensor µs | Ordered fused µs |
|---:|---|---:|---:|
| 1 | silu_mul | 4.09 | 1.41 |
| 1 | weighted_sum | 8.45 | 1.44 |
| 1 | norm | 14.11 | 1.54 |
| 4 | silu_mul | 4.61 | 1.43 |
| 4 | weighted_sum | 8.69 | 1.45 |
| 4 | norm | 15.51 | 1.51 |
| 256 | silu_mul | 25.04 | 11.13 |
| 256 | weighted_sum | 34.45 | 3.08 |
| 256 | norm | 18.74 | 1.96 |

The final image passed **105 targeted GPU tests**, including every finite BF16
SiLU input with multipliers 1 and 0.731; strict weighted-sum equality for top-k
2, 3 and 16; exact norm equality at widths 64, 128, 1024, 1536 and 2048 with
1, 4, 16, 81 and 673 rows; and graph replay after changing inputs, routes, weights
and norm gains. The engine cache/full-weight-refresh exercise passed with decode
graphs enabled, including restored weights and comparison with a fresh engine.
Portable checks including the benchmark passed 26 tests with 33 CUDA skips.
Ruff and the documentation build passed.

## Workload and provenance

This reuses the [earlier study's frozen RL basket](graph-compatible-rounding-20260923.md):
16 prompts, four each from math, code, IFEval and general tasks; 512 generated
tokens/request; temperature/top-p 1; top-k -1; EOS ignored for equal work. SFT
arms use two repetitions at batch four and base one; the batch-16 EMO arm uses
four repetitions. All modes use identical prompt IDs and four shared fixed
continuations. Default and rounding may sample different continuations, so
their rollout aggregates are each measured on their own generated work.

All runs used one H100 on `jupiter-cs-aus-167.reviz.ai2.in`. The final immutable
image is `01M38NY00GWVJ7WH26S3V94C9Z`, Docker ID
`sha256:a3487ea3f493ac4a8ab2dd8e6ce85e5ec4f5c59e450954238491c6ebfed43456`.
Application source is `7ef67f07e`, serving source
`cc785298304fe3e868df944ac545a5363ba4643d`, and Core source
`e505356353aa7ce1f6ff83e24d6eb945f463714e`. The runtime uses PyTorch
`2.13.0+cu130` and Triton `3.7.1`; repeat exact-equality and workload checks when
changing those dependencies. Later serving documentation-only commit `9cb4601`
does not change the implementation used by this image.

Qualification is scoped to TP1/EP1, unquantized BF16, Triton expert GEMMs, full
decode graphs and eager prefill. It does not establish tensor/expert parallelism,
quantization, prefill graphs, torch.compile or an RL optimizer-step outcome.

Runs:

1. Final ordered implementation, three checkpoints, exact equality and qualification: [Beaker](https://beaker.org/ex/01M38P0HAQ887MFD65XEQM4FY5).
2. Final ordered implementation at batch 16 with tensor and repeated default controls: [Beaker](https://beaker.org/ex/01M38PKVY7J5WY8KF9FHNJJN5V).
3. Initial generic fusion, three checkpoints: [Beaker](https://beaker.org/ex/01M38MCJEFGV9YVX95NPG4WQNM).
4. Initial generic fusion, expert/norm isolation and batch 16: [Beaker](https://beaker.org/ex/01M38NEYBNGQEV26RVMZB2BB7C).

The [machine-readable report](fused-rounding-20260923.json) retains full precision
statistics, batch timings, domain results, source/image/result IDs and report
hashes. Raw tokens and per-token probabilities remain in those Beaker results.

## Selection

Keep the ordered fused kernels as the implementation of opt-in rounding mode,
with the tensor path as a diagnostic control and full compatibility as the slow
reference. The measured throughput penalty is recovered without changing the
tensor control's measured behavior. This makes rounding mode practical for the
next scoped RL exercise, while leaving the default unchanged and keeping the
remaining Core/serving mismatch and learning impact explicit.
