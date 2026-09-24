# Graph-compatible expert rounding and RMS norms

This study tests the intermediate `OLMO_SGLANG_CORE_COMPAT=rounding` serving mode against ordinary graph-enabled serving and the same rounding mode with graphs disabled. Both modes remain opt-in; `1` / `full` retains the earlier eager Core reference.

The new path uses ordinary SGLang expert weight storage, token alignment, tuned Triton GEMMs, full attention, and KDA execution. It makes BF16 SiLU rounding explicit before multiplication, writes BF16 expert down outputs before routing weights, and combines experts in FP32 before a final BF16 cast. Ordinary RMS norms use Core's FP32 expression. Dense/shared MLPs already have separate BF16 activation operations. It does not adopt Core's grouped GEMM layouts or change KDA state orientation.

These changes use normal captured tensor operations and the existing expert loaders, with no global activation monkeypatch or cached copy of weights. Qualification separately tests changed inputs/routes/weights under graph replay and populated cache invalidation across full weight updates. Initial scope is TP1/EP1, unquantized BF16, the Triton MoE backend, full decode graphs and eager prefill, without speculation. This does not qualify tensor/expert parallelism, quantization, torch.compile, prefill graphs or an RL optimizer step.

## Origin of the original arithmetic

The Olmo adapter selected SGLang's generic `FusedMoE`. The multiply-before-rounding order was already present in the inherited SGLang kernel, rather than specified by the exported checkpoint or introduced by this investigation. At pinned SGLang source `3145136dcd1238754e0ea2b2ffd546532119c71c`, `python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py` lines 310–314 and 609–613 multiply the FP32 down-projection accumulator by its routing weight, then cast to the output computation dtype. The caller in `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py` enables that weighting on the down projection. The adapter's `FusedMoE` selection is present at adapter revision `72f194a35045f02cc7d87980819bd0e4652cc931` in `src/olmo_sglang/models/olmo3_moe.py`.

Core instead rounds the down-projection output to BF16 before FP32 routing-weight multiplication and combination. The formulas are mathematically equivalent but have different finite-precision behavior. Our integration inherited SGLang's generic choice; our compatibility modes deliberately change it to follow Core. The available upstream Git history is shallow, so this source inspection establishes inherited behavior, not the original introducing commit or author.

## Qualification

The final image passed **75 targeted GPU tests**, including graph replay with changed inputs, routes and weights, and an independent per-route Torch computation check. The engine-level cache exercise passed with decode graphs enabled: cold and warm requests, mixed batches, chunked prefill, full weight updates, restored weights and comparison with a fresh engine. The unchanged standard/fused loader storage remains in use. The portable subset passed 54 tests with 16 CUDA-dependent skips; the benchmark's two unit tests passed in the image. Ruff passed.

## Measurement contract

The frozen 16-prompt real RL basket and checkpoint paths are identical to the [full compatibility benchmark](core-compatible-serving-20260923.md): four prompts each from math, code, IFEval and general tasks, batches of four, 512 generated tokens/request, temperature 1, top-p 1, and EOS ignored for equal work. Each SFT arm has two repetitions and the base arm one. Timing excludes engine loading, warm-up and Core scoring. Every retained rollout is scored through actual BF16 Core with Torch full attention. Four shared fixed continuations provide a separate full-prefix scoring control.

All three modes in this study run on the same H100 on `jupiter-cs-aus-220.reviz.ai2.in`. This differs from host 219 used in the earlier full-compatibility study; use the new within-job default baseline when assessing throughput changes. The final image is `01M38HZ1ZXZHQFXBQRTXV23J3M`, Docker ID `sha256:9a4288970be880004400d4cabeb41cd771689a7894a2b170ac2ed6bba84fe751`, application `daeee120f`, serving `ca852b80156d07ec1fcf0d5c321d3eba3a7b530c`, and Core `e505356353aa7ce1f6ff83e24d6eb945f463714e`.

"Rollout error" compares the probability returned during cached generation with Core's teacher-forced probability for the same chosen token and causal prefix. "Fixed-prefix error" scores the same four complete continuations through every model/mode; this is a prefill scoring control, not cached generation. Both are absolute natural-log-probability differences. The outside-20% measure counts tokens for which `p_Core / p_serving` is outside `[0.8, 1.2]` at unchanged weights; it is not an observed PPO clipping rate after a training update.

Default and rounding modes can sample different continuations. Their rollout aggregates therefore compare each mode on its own generated work, not paired token-by-token interventions. The fixed-prefix control uses identical continuations. The small prompt basket and repetition count do not establish a learning-quality effect or a statistically robust ordering of small error differences. Throughput describes warmed generation at batch four on one H100, not end-to-end RL iteration time.

## Results

| Checkpoint | Serving mode | Rollout mean error | Rollout p99 | Ratios outside 20% | Fixed-prefix mean error | Tokens/s | Median batch seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| Hero base | Default + graphs | 0.03706 | 0.21258 | 1.282% | 0.03476 | 1001.0 | 2.049 |
| Hero base | Rounding + graphs | 0.03735 | 0.20911 | 1.062% | 0.03545 | 751.6 | 2.720 |
| Hero base | Rounding, eager | 0.03735 | 0.20911 | 1.062% | 0.03545 | 174.8 | 11.723 |
| Hero EMO SFT | Default + graphs | 0.04231 | 0.32679 | 3.790% | 0.04258 | 991.1 | 2.068 |
| Hero EMO SFT | Rounding + graphs | 0.04151 | 0.32525 | 3.656% | 0.04226 | 757.0 | 2.707 |
| Hero EMO SFT | Rounding, eager | 0.04151 | 0.32525 | 3.656% | 0.04226 | 175.7 | 11.615 |
| Hero non-EMO SFT | Default + graphs | 0.03499 | 0.27612 | 2.545% | 0.04004 | 992.3 | 2.064 |
| Hero non-EMO SFT | Rounding + graphs | 0.03531 | 0.27025 | 2.380% | 0.04100 | 755.0 | 2.710 |
| Hero non-EMO SFT | Rounding, eager | 0.03531 | 0.27025 | 2.380% | 0.04100 | 183.0 | 11.207 |

Relative to the within-job default, rounding with graphs loses 23.6% throughput on EMO, 23.9% on non-EMO and 24.9% on base; equal-work generation takes 1.31–1.33× as long. It is 4.1–4.3× faster than the same rounding arithmetic with graphs disabled. Mean rollout error decreases 1.9% on EMO, increases 0.9% on non-EMO and increases 0.8% on base. The outside-20% fraction decreases by 0.13, 0.16 and 0.22 percentage points respectively.

Across all three checkpoints, graph-enabled and eager rounding produce identical sampled token sequences and exactly equal reported log-probabilities, for both their own rollouts and the fixed-prefix controls. This includes 40,960 generated tokens and 6,144 fixed-prefix token scores per mode. It establishes graph/eager equivalence for this workload, not every possible request shape.

The tail is not uniformly better: maximum rollout error increases from 1.391 to 1.826 for EMO and 1.358 to 1.568 for non-EMO, while base improves from 0.519 to 0.496. Even where mean or p99 improves, isolated discrepancies remain. Domain results and full precision values are retained in the [machine-readable report](graph-compatible-rounding-20260923.json).

### Earlier full-reference comparison

The earlier full mode ran on H100 host 219, with the same prompt basket and fixed continuations. These are historical reference rows, not another arm of this new within-job timing comparison. Its own default baseline ran at 994–1,002 tokens/s.

| Checkpoint | Full-mode rollout mean error | Full-mode fixed-prefix mean error | Full-mode tokens/s |
|---|---:|---:|---:|
| Hero base | 0.03635 | 2.33e-7 | 151.7 |
| Hero EMO SFT | 0.03938 | 4.38e-7 | 153.7 |
| Hero non-EMO SFT | 0.03339 | 4.26e-7 | 151.1 |

See the [full-mode measurement](core-compatible-serving-20260923.md) for its matched baseline and validation. The intermediate mode retains ordinary attention, KDA dispatch/state layout and expert GEMMs. Rounding/norm changes alone therefore do not recover full mode's near-exact long-prefix scoring. The earlier two-natural-prompt diagnostic improvement does not generalize into a large mean improvement on this longer workload.

## Decision

Keep both compatibility modes opt-in. The graph-compatible implementation is qualified for this scoped serving workload and useful for arithmetic investigations, but a 24–25% throughput cost with small, mixed mean-error changes does not justify making it the RL default. It also misses the initial 10–20% throughput-loss target. Full compatibility remains a slow reference for isolating discrepancies. Neither study establishes whether these differences materially change learning.

If further compatibility work is pursued, the next useful measurement is an isolated graph-safe KDA dispatch/state-layout intervention against these same fixed continuations, followed by cached rollout scoring. Kernel fusion that explicitly preserves the BF16 rounding boundaries could reduce this mode's cost, but the current data does not establish that such optimization would buy a meaningful RL benefit.

Final experiment status: completed, exit code 0. Immutable result dataset: `01M38J0P3C3H7A54MF5S9E216V`.

Runs:

1. Qualification and workload comparison: [Beaker](https://beaker.org/ex/01M38J0P363W3M94AMMN43Q806).
2. Initial standalone-test fixture failure (execution config not initialized): [Beaker](https://beaker.org/ex/01M38HCQGCXZJ15XDHJCXGKJQZ).
3. Hostname-constrained attempt, stopped while queued: [Beaker](https://beaker.org/ex/01M38HKR9PMAVR4X68BHE1K9AQ).
4. Standalone-test fixture constructor failure: [Beaker](https://beaker.org/ex/01M38HR4EWR2AF8WBG11HPYNTC).

The early test-fixture attempts stopped before workload execution; they are not numerical results. The isolated fixture configuration was checked in the exact container before the final attempt. The production serving implementation is unchanged across these fixture fixes.
