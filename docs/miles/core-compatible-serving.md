# Core-compatible serving modes

Fused rounding is the default for the qualified hero checkpoint family when
using an image built from the current runtime lock. The adapter selects it from
the checkpoint configuration and resolved serving settings; no environment flag
is required. Other checkpoint geometries and serving setups retain ordinary
SGLang execution. The slow full reference remains opt-in.

## Choose a mode

Set overrides in the run's `[launch.env]` table. Unset means `auto`.

| `OLMO_SGLANG_CORE_COMPAT` | Behavior | When to use |
| --- | --- | --- |
| unset or `auto` | Fused rounding for the qualified profile below; ordinary serving otherwise | Normal operation |
| `0` or `off` | Ordinary SGLang arithmetic, including its original expert rounding/reduction | Reproduce the old baseline or opt out |
| `rounding` | Core-style BF16 boundaries and FP32 expert reduction/norms; retains SGLang GEMMs, attention, KDA and decode graphs | Explicit compatible arithmetic, or qualification of another supported shape |
| `1` or `full` | Core layouts/grouped GEMMs, eager norms, SDPA and Core-style KDA prefill; no graphs | Numerical investigation; substantial performance cost |

Within `rounding`, `OLMO_SGLANG_ROUNDING_KERNELS` defaults to `fused`.
Set it to `torch` for the original tensor control, or `moe` / `norms` to fuse
only that component for diagnosis. This second flag alone does not enable
rounding. `OLMO_HF_MOE_CORE_REFERENCE` is a separate Transformers flag and has
no effect on SGLang.

The automatic profile is the 12.5B hero geometry used by the measured base,
EMO SFT and non-EMO SFT checkpoints: hidden size 1024, 16 layers with full attention
at layers 7 and 15 and KDA elsewhere, 512 experts/top-16, latent width 512,
expert/shared width 1024, dense width 8192 and the measured norm/gating settings.
The serving implementation's [`_ROUNDING_PROFILE`](https://github.com/allenai/olmo-sglang/blob/e0b0849d09509224879ed02aecfca41eff11e50d/src/olmo_sglang/core_compat.py)
is the exact field contract.
Selection requires unquantized BF16 (including `dtype=auto` when resolved to BF16
by the model loader), TP1/EP1, the `auto`/`triton` MoE backend,
full or disabled decode graphs, disabled prefill graphs, no speculation and no
`torch.compile`. It is independent of checkpoint path and EMO ancestry.
Unsupported settings automatically keep the old path; explicit `rounding` and
`full` retain their runtime validation. The model logs the resolved mode at
construction. This does not change graph settings or the trainer backend.

For the measured fast configuration, retain:

```toml
[inference]
rollout_tensor_parallel_size = 1
sglang_cuda_graph_backend_decode = "full"
sglang_cuda_graph_backend_prefill = "disabled"
```

For an explicit opt-out, add:

```toml
[launch.env]
OLMO_SGLANG_CORE_COMPAT = "0"
```

## Why this default

Fused rounding restores Core's BF16 SiLU and down-projection boundaries, followed
by FP32 expert weighting/reduction and FP32 RMS normalization. It keeps the fast
SGLang GEMMs and decode graphs. Correct reduction grouping mattered: the first
fusion attempt produced rare different BF16 values that changed full-model
routing; the final kernels reproduce the tensor control's accumulation order.

Measured warmed generation throughput on one H100, using real RL prompts:

| Checkpoint / concurrent requests | Ordinary serving (tok/s) | Tensor rounding (tok/s) | Fused rounding (tok/s) |
| --- | ---: | ---: | ---: |
| Hero base / 4 | 993 | 752 | 1,010 |
| Hero EMO SFT / 4 | 988 | 751 | 1,000 |
| Hero non-EMO SFT / 4 | 983 | 747 | 1,004 |
| Hero EMO SFT / 16 | 2,486 | 2,043 | 2,513 |

Treat the small apparent speed gains as essentially equal throughput, not a
claimed speedup. Fused and tensor rounding matched exactly on 73,728 generated
tokens and 8,192 fixed-prefix token scores. The earlier 24–25% penalty was the
separate tensor operations, not an unavoidable cost of the arithmetic.

This **does not establish exact Core parity or an RL learning improvement**.
Mean absolute generated-token log-probability differences against actual Core
were small and mixed: ordinary → rounding was 0.03706 → 0.03735 (base),
0.04234 → 0.04151 (EMO), 0.03497 → 0.03531 (non-EMO), and
0.04088 → 0.04286 (EMO, batch 16). These compare serving probabilities recorded
during cached generation with Core's teacher-forced scores on the same tokens;
they are not errors measured against a separately sampled Core response, nor
proof of a cache bug. Attention/KDA dispatch and GEMM batch shapes still differ.
The reason for the default is the intended arithmetic at essentially no measured
serving cost, not universal dominance on every probability metric.

## Evidence and continuing the investigation

- [Final fused benchmark](measurements/fused-rounding-20260923.md) and its linked
  JSON retain checkpoint paths, exact source pins, immutable images, Beaker runs,
  timing protocol, probability tables and component ablations.
- [Tensor rounding benchmark](measurements/graph-compatible-rounding-20260923.md)
  records the initial slowdown and graph comparison.
- [Full reference and historical checkpoint comparison](measurements/core-compatible-serving-20260923.md)
  separates short probes, full-prefix scores and cached-generation measurements.

The automatic selection is a subsequent policy change; the final benchmark image
`01M38NY00GWVJ7WH26S3V94C9Z` contains the same rounding kernels but still needs
an explicit `rounding` flag. Build a new image from the current runtime lock to
get automatic selection (serving revision `5514bf5`, which also handles
the normal MILES `dtype=auto` setting). The selection
change passed 47 portable tests; 34 CUDA-only tests were skipped locally. It
changes no rounding kernels; the 105-test GPU and workload qualification cited
below precedes this default-selection change. Do not relabel that earlier image
as having the new default. Results are qualified on H100 with the pinned Torch/Triton/SGLang stack;
other hardware, batch shapes and updated dependencies need measurement.

To continue: preserve prompt/token identities, compare actual cached behavior
probabilities against Core on those same tokens, retain an ordinary (`0`) control
and a tensor-rounding control, and measure warmed generation separately from
loading and Core scoring. The benchmark's `default_graphs` label intentionally
means the historical ordinary control and explicitly sets `0`, even now that
production auto selection can choose rounding. Rerun strict component equality,
graph replay and cache/weight-refresh checks when changing kernels or runtime
pins. Any decision based on learning quality needs a paired RL experiment.

## Full eager reference

`OLMO_SGLANG_CORE_COMPAT=1` selects an experimental SGLang execution mode that
follows OLMo-core's BF16 rounding, expert layouts, normalization and full-attention
arithmetic. The full reference is **off by default**. This is a serving flag; the separate
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

Set `OLMO_SGLANG_CORE_COMPAT = "rounding"` in `[launch.env]` to explicitly select
the intermediate arithmetic mode (automatic for the qualified profile above). Set `sglang_cuda_graph_backend_decode = "full"`
and retain disabled prefill graphs in `[inference]`. Use unquantized BF16,
TP1/EP1 and the Triton MoE backend. This mode retains ordinary attention,
KDA dispatch and expert weight layouts; it changes BF16 activation/down-output
rounding, FP32 expert combination and RMS norms. The existing `1`/`full` mode
continues to select the eager reference. Only the qualified rounding profile is selected automatically.

The workload benchmark accepts `--mode rounding_graphs` and `--mode rounding`
for paired graph/eager checks. Qualify the actual graph replay, cached rollouts
and weight refresh before selecting this experimental mode for RL.

The [original tensor implementation](measurements/graph-compatible-rounding-20260923.md)
lost 24–25% throughput. Its arithmetic is retained as a diagnostic control with
`OLMO_SGLANG_ROUNDING_KERNELS = "torch"` in `[launch.env]`.

The [fused implementation and workload comparison](measurements/fused-rounding-20260923.md)
removes the separate activation, expert-combination and norm intermediates while
preserving the BF16 boundaries and the pinned PyTorch reduction grouping. It is
the default implementation within rounding mode, which is now selected
automatically for the qualified profile. The full eager reference is unchanged.

On the measured H100 workload, fused rounding ran within a few percent of default
serving at batches four and sixteen. It matched tensor rounding's tokens and
log-probabilities exactly across 73,728 generated tokens and 8,192 fixed-prefix
scores, recovering the earlier throughput penalty.

The benchmark used image `01M38NY00GWVJ7WH26S3V94C9Z` (application `7ef67f07e`, serving
`cc78529`). Its strict component/replay suite passed 105 tests and its engine
cache/full-weight-refresh exercise passed. Matching the tensor rounding control
does not imply exact agreement between cached serving probabilities and Core's
teacher-forced scores or establish an RL learning benefit. Consult the measurement
for throughput, probability comparisons and qualified shapes; repeat qualification
after runtime changes.
