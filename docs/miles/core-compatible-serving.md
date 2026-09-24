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

## Optimized serving versus a strict parity check

**Use automatic fused rounding for normal RL rollouts on the qualified hero
profile.** It restores the intended rounding and normalization while retaining
fast kernels and decode graphs. Read the model's startup log to confirm that
`auto` resolved to `rounding`; an unset flag alone does not establish which path
ran. Keep Core as the training/scoring policy and monitor probability differences
on the actual rollout tokens.

A strict parity investigation has a different execution contract:

| Path | What to compare | What agreement establishes |
| --- | --- | --- |
| Core versus HF conversion reference | Identical exported weights and token IDs; BF16 forward weights, FP32 router math; `OLMO_HF_MOE_CORE_REFERENCE=1`, matched Torch/SDPA attention (math SDPA for the strict converter), no cache, same sequence and batch shapes | A controlled conversion/forward oracle; the earlier short controls were exact |
| Core versus SGLang `full` | Same full prefixes, Core Torch attention, TP1/EP1, disabled serving graphs | Closest available serving reference; earlier long-prefix selected scores differed by only about 3.5e-7 mean / 1.9e-6 maximum |
| Core versus SGLang automatic `rounding` | Actual cached rollout scores and token choices, rescored by Core on those same prefixes | The practical training/serving discrepancy with the recommended fast implementation |

The **pure parity path is the first row**, with every operator/backend and shape
held fixed and the outputs checked explicitly. Neither flag alone promises
bitwise equality on every workload. Use the
[Core/HF fidelity diagnostic](../../scripts/miles/hero_core_fidelity.py) and its
[recorded source/attention controls](measurements/hero-core-fidelity-20260923.md)
when validating an export. `OLMO_HF_MOE_CORE_REFERENCE=1` does not alter SGLang,
and enabling SGLang `full` does not turn cached generation into the same
computation as a full-prefix Core forward.

Use SGLang `full` to isolate a numerical discrepancy, not as the default RL
recipe. On the earlier batch-four workload it cost approximately 6.5 times the
generation time and only modestly improved cached-generation probability errors.
The optimized mode recovered the arithmetic changes without the tensor-control
slowdown. No measured learning-quality benefit yet justifies the full mode's
cost.

## Interpreting token and probability agreement

Compare **Core versus SGLang within each checkpoint**. EMO and non-EMO have
different weights and are not expected to produce identical answers. Their
shared architecture does not imply the same sensitivity to arithmetic changes;
the EMO-specific excess error has not been causally isolated.

The existing 0.05 mean absolute log-probability gate is a bounded mechanics
criterion, not a requirement that every token be close or that greedy choices
match. The [paired four-update smoke](measurements/hero-rl-smoke-20260924.md)
passed that mean criterion for both checkpoints, but its worst individual
probability gaps were 0.734 nats (EMO) and 1.199 nats (non-EMO). It did not capture
Core's argmax choices. A small average alone does not make those tails harmless.

A greedy disagreement means the two systems prefer different next tokens on an
identical prefix. Inspect the top-two margin: a near tie can flip with a tiny
probability change, whereas a large margin calls for closer investigation.
Once independently generated responses diverge, later prefixes differ too;
response equality is a different question from same-prefix forward parity.
Conversely, matching argmax tokens does not imply matching policy probabilities
or identical stochastic samples. Do not infer full-distribution KL from selected
token probabilities or just the top two alternatives.

Core full-sequence teacher forcing and Core prefix-at-a-time forwards also use
different matrix shapes. Retain both when diagnosing serving differences:
causal prefixes match semantically, while finite-precision execution can still
differ. A cached serving/full-sequence scoring discrepancy is not, by itself,
evidence of a cache implementation bug.

## Same-prefix greedy probe — September 24, 2026

The [bounded inference run](https://beaker.org/ex/01M391832GK9QPG5XS39SBFX07)
completed successfully on one H100 80GB. Each checkpoint/mode generated 64 greedy
tokens for each of four frozen real RL prompts (math, code, instruction following,
general; prompt lengths 161, 167, 156 and 113), one request at a time. Each prompt
received an equal-length warm-up. Timing excludes model loading, warm-up and Core
scoring; it includes returning selected-token and top-two probabilities. This is
256 measured generated positions per row, not an answer-quality evaluation.

**Neither serving mode gives 100% cached-generation agreement with Core.** Here
Core recomputes each same prefix separately without a cache; agreement compares
its argmax with the token SGLang actually generated. Probability differences are
absolute natural-log differences for that generated token.

| Checkpoint | Serving mode | Same next token | Mean log-prob difference | Max difference | Warm tok/s | Seconds / 256 tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EMO SFT | Automatic fused rounding | 251/256 (98.05%) | 0.025563 | 0.380364 | 258.9 | 0.989 |
| EMO SFT | Full reference | 253/256 (98.83%) | 0.020367 | 0.644444 | 34.7 | 7.383 |
| Non-EMO SFT | Automatic fused rounding | 252/256 (98.44%) | 0.019451 | 0.186106 | 264.3 | 0.969 |
| Non-EMO SFT | Full reference | 255/256 (99.61%) | 0.015440 | 0.208349 | 34.9 | 7.325 |

All four means are below 0.05, but that does not establish identical decisions.
The full reference costs 7.47×/7.56× the generation time here. These are short,
single-request estimates, not replacements for the longer batched benchmark
below. Modes can generate different continuations, so their cached error columns
do not isolate arithmetic on a common trajectory.

The optimized EMO disagreements all select Core's second-ranked token; Core's
preferred token leads by 0.125–0.375 nats. Non-EMO has three second-choice
selections with margins 0.0625–0.125 and one exact tie under Core. Full reference
has two strict EMO disagreements and one Core tie, plus one strict non-EMO
disagreement. Its largest EMO disagreement selects a token ranked fifth under
Core, 1.125 nats behind Core's favorite. Thus the remaining differences are
**not all tie-breaking**, and the slower mode does not uniformly eliminate tails.
Whether they matter to RL learning remains unmeasured.

On the optimized trajectories, the first Core/SGLang token-choice disagreement
occurs at response position 17 for both checkpoints (EMO math; a Core tie on
non-EMO instruction following), counting from one. The first strict non-EMO
disagreement is at position 38 on the code prompt.
The four-prompt probe does not independently generate complete Core responses:
Core keeps receiving the retained SGLang prefix after a disagreement. Across
serving modes themselves, two of four 64-token continuations match exactly for
each checkpoint; the other two begin diverging at positions 30/62 for EMO and
17/38 for non-EMO. Those are within-checkpoint comparisons.

### Full-sequence control

Both serving modes also rescore the **same optimized continuations** as complete
input sequences. Compare these input-token probabilities with one complete
Core forward, with matching sequence lengths:

| Checkpoint | Optimized mean difference | Full-reference mean difference | Full-reference maximum |
| --- | ---: | ---: | ---: |
| EMO SFT | 0.019569 | 4.20e-7 | 1.32e-6 |
| Non-EMO SFT | 0.017857 | 4.10e-7 | 1.08e-6 |

This near-exact full-sequence result is the useful numerical baseline. It does
not describe cached generation. Even Core changes its argmax at 1/256 EMO and
2/256 non-EMO positions when switching from full-sequence scoring to separate
prefix forwards on this cohort. Comparing full-reference input scoring with
those separate Core prefixes gives means 0.012154 and 0.008261. This demonstrates
shape-dependent arithmetic; it does not isolate every remaining cached-decoding
kernel difference.

For continuity with earlier teacher-forced rollout measurements, cached scores
versus **full-sequence** Core give means 0.022699/0.020174 (EMO optimized/full)
and 0.019286/0.016229 (non-EMO optimized/full), with actual greedy-token agreement
252/253 and 254/255 out of 256 respectively. Keep this scorer definition distinct
from the prefix-at-a-time table above.

### Artifacts and reproduction

The [machine-readable summary](measurements/core-token-choices-20260924.json)
retains disagreement token IDs and positions, top-two probabilities, Core ranks
and margins, engine settings, source pins and timings. Raw artifacts are Beaker
result dataset `01M391832WNY710ATQJ8QTT5MM`: `samples.json`, both modes' serving
JSON, both Core scorer outputs, checkpoint config/source digests and logs.
The two checkpoints are the corrected-tokenizer 4T Dolci Think step5402 exports
in the [checkpoint audit](measurements/hero-sft-20260923.md).

Runtime image `01M38YYQFVP5EBXRG9D83RW3CQ` contains application `d5b60f2ebbd2`,
Core `e505356`, olmo-sglang `5514bf5` and SGLang `3145136`. The job overlaid only
committed diagnostic `37c7ef386`; it did not replace runtime model code.
[`probe_core_choices.py`](../../scripts/miles/probe_core_choices.py) runs serving
and scoring in separate processes, with BF16 weights, TP1/EP1, Triton attention,
no radix reuse or overlap, eager prefill, and full decode graphs only for `auto`.
Core uses its Torch attention backend. No training or weight update occurs.

The original diagnostic's `CHOICES` log lines used the first reported top-two
entry, which can differ from the actual greedy token on exact ties. The summary
recomputes cached agreement from retained `output_ids`; do not use those raw
log counts as generated-token agreement. The current diagnostic fixes this
reporting issue. Forced-score choice fields still denote a reported top-one
representative, with ties ambiguous; the full-sequence table above reports
probabilities rather than claiming a separately generated token sequence.

To regenerate the summary:

```bash
beaker dataset fetch 01M391832WNY710ATQJ8QTT5MM -o runs/core-choices/results
python scripts/miles/summarize_core_choices.py \
  runs/core-choices/results runs/core-choices/summary.json \
  --experiment 01M391832GK9QPG5XS39SBFX07 \
  --result-dataset 01M391832WNY710ATQJ8QTT5MM
```

The local RTX 4090 was usable: a BF16 CUDA operation passed outside the agent
sandbox. After CUDA initialization it had 23.02 GiB free, while hero BF16
weights alone require approximately 23.28 GiB, before activations and serving
caches. That memory constraint, not an absent GPU, motivated the H100 run.

### Historical Open Instruct comparison

The older `grpo.py` Core/vLLM trainer records mean, maximum and standard deviation
of absolute trainer/rollout log-probability gaps through
[`compute_vllm_local_debug_metrics`](../../open_instruct/grpo_utils.py).
A verified numerical baseline for that particular backend pair was not recovered
in this investigation. The retained [dense Olmo 3 framework comparison](measurements/gsm8k-dense-test-20260916.md)
reports approximately 0.01 mean gaps for both **HF/DeepSpeed + vLLM** and
**MILES/Core + SGLang**; it is not a Core/vLLM result or a matched hero comparison.
The old metric uses the current trainer forward against stored rollout scores,
so asynchronous policy lag or intervening updates can contribute. An unchanged-
weight, same-prefix comparison is needed to isolate numerical drift.

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
