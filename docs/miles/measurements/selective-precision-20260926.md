# Selective precision: frozen-weight attribution

This investigation tests whether a few inexpensive precision changes can remove
roughly half of the inference/training mismatch without sacrificing serving
speed. The initial larger comparison does **not** support that hypothesis:
the FP32-output LM-head remains the most reliable small change, and several
apparently promising single-site changes do not add together. These are
process-local diagnostic patches, not new production defaults.

## Method

Compare SGLang cached decoding against Core full-sequence scoring at the same
SFT weights and **identical token trajectories**. A sampler hook supplies each
next token from a frozen continuation while retaining its original model
probability. The four-prompt baseline reproduces the earlier ordinary sampler's
log-probabilities exactly. Every arm checks token alignment, prompt/model
provenance and unchanged Core parameter version counters. No optimizer runs.

The screening workload is four prompts with 512 response tokens each. The
larger comparisons use 16 prompts, 8,192 cached response tokens per arm, including
12 prompts outside the initial screen. A separate control scores the first four
complete prefixes through serving prefill. Error is mean absolute natural-log
probability difference, not KL divergence. The ratio-tail statistic is the
fraction of `p_Core / p_serving` outside [0.8, 1.2]; it is not an observed PPO
clipping rate or evidence of learning harm.

Timing uses ordinary sampling, separately from forced-token scoring. Each arm
warms all prompts for a full 512-token generation, then times three repetitions
in batches of four. Forced sampling's CPU synchronization is excluded. Serving
retains fused rounding kernels, packed KDA decode, optimized experts and full
CUDA decode graphs. Graph-audit snapshots record successful graph execution;
prefill is not captured. Timings are small single-H100 measurements, not a
large-concurrency throughput qualification.

The runtime is image `01M3APTWSR4VH2PFYMMSPQ8TD2`, Core
`e505356353aa7ce1f6ff83e24d6eb945f463714e`, serving adapter
`5514bf5885e690f7cfcd9cfb08e03111fbd70a78`, and SGLang
`3145136dcd1238754e0ea2b2ffd546532119c71c`. The model has 14 KDA
layers and 15 latent MoE blocks; its first block is dense. Both step-5402 SFT
checkpoints and the frozen prompts are the same as the
[LM-head study](lm-head-fp32-20260925.md). Prompt dataset
`01M391832WNY710ATQJ8QTT5MM` has SHA256
`671e219280fb30b526cfdd0099b0a5d3c06ab122f1eb49a0c04ae88fbb0b6c99`.

## What the component replay isolates

Capture real Q/K/V, gate and beta inputs at KDA indices 0, 6 and 13 on two
512-token continuations. Feed the **same tensors** to chunked full-sequence KDA
and prefix-prefill plus recurrent decoding. This controls for upstream model
errors; it does not measure their contribution.

* Rounding normalized Q/K to BF16 inside packed decode reduces local output
  mean error by 6.7–11.3%. Replacing division with reciprocal multiplication
  alone produces no change in this sample.
* Widening the KDA inputs and running its kernels in strict FP32 reduces local
  relative RMS discrepancy from 0.0028–0.0039 to roughly 0.0000028–0.0000067:
  a 400–1,400-fold improvement with the same projected input values.
* That improvement does **not** translate into a comparable full-model gain.
  The matched four-prompt full-model comparison improves only 4.0%; adding
  the FP32-output head improves 15.0% in total.

The persistent recurrent state is already FP32. BF16 chunk computation still
has reduced-precision normalized inputs, transformed values, attention terms
and temporary state views. A separate forward-only attribution probe starts
from an FP32 chunk computation and rounds one returned intermediate at a time.
Transformed values (`u`, `v_new`) and intra-chunk attention (`Aqk`) introduce
larger discrepancies than `w` or `kg` on these captures. Normalized Q/K and
chunk state snapshots also contribute. This is **not an additive error budget**:
rounding an intermediate after its computation does not reproduce all internal
casts in a BF16 kernel.

The practical implication is narrower than “KDA causes the model mismatch.”
KDA contains a reproducible precision discrepancy, but fixing that discrepancy
with identical inputs leaves most end-to-end error. The complete paths can
already disagree at the KDA inputs and can introduce further differences in
subsequent layers.

## Initial single-site screen

Non-EMO, four fixed cached continuations, default runtime arithmetic:

| Change | Mean error | Reduction | Tokens/s |
|---|---:|---:|---:|
| Baseline | 0.036592 | — | 985 |
| Q/K normalization rounded like Core | 0.034866 | 4.7% | 986 |
| Core-compatible KDA prefill only | 0.035128 | 4.0% | 971 |
| Both above | 0.034039 | 7.0% | 974 |
| Gate final projection FP32 output | 0.034249 | 6.4% | 1,058 |
| Beta projection FP32 output | 0.035187 | 3.8% | 1,050 |
| LM-head FP32 output | 0.032370 | 11.5% | 992 |
| Q/K rounding + LM-head | 0.030672 | 16.2% | 983 |

The gate and beta patches keep BF16 GEMM operands and preserve FP32 outputs.
They are applied at the matching Core and serving sites. Apparent throughput
improvements in this small screen should not be treated as speedups: the arms
are sequential, with no ending baseline in this first experiment.

## Larger combination comparison

Non-EMO, 16 fixed continuations. The repeated baseline is numerically identical.
“Small combination” means Q/K rounding, FP32-output gate and beta projections,
and FP32-output LM-head. The latent change keeps latent-up projection output,
shared-expert addition and following normalization in FP32, then returns BF16
to the residual stream. It leaves the large routed expert bank unchanged.

| Change | Mean error | Reduction | Error p99 | Ratios outside 20% |
|---|---:|---:|---:|---:|
| Baseline | 0.035414 | — | 0.27258 | 2.209% |
| LM-head | 0.031286 | 11.7% | 0.25539 | 1.929% |
| Latent-up path | 0.036333 | −2.6% | 0.27347 | 2.466% |
| Latent-up + head | 0.031607 | 10.7% | 0.25121 | 1.880% |
| Small combination | 0.032466 | 8.3% | 0.26766 | 2.368% |
| Small combination + latent-up | 0.031454 | 11.2% | 0.24220 | 1.855% |

Measured throughput was 1,009 tokens/s for both starting and ending baselines,
1,008 for the head, 1,003 for latent-up, 1,004 for latent-up plus head,
1,010 for the small combination and 993 for the combination plus latent-up.
Each graph audit recorded 10,220 cached decode calls using CUDA graphs.
The head difference is about −0.2%; these data support a low serving cost
on this workload, not an inference speedup.

The extra changes do not improve mean mismatch beyond the head alone. Some tail
statistics improve, others worsen; none of these results establishes that
selective precision improves RL learning or makes long asynchronous runs safe.

## Arithmetic controls

FP32 KDA experiments disable TF32 explicitly in PyTorch, NVIDIA and Triton,
and use IEEE precision for FLA's triangular solve. They have their own matched
BF16 baseline. Those global settings also change numerical results outside
the widened KDA region; comparing against the earlier default-arithmetic
baseline would confound the attribution.

Under that strict control, four-prompt non-EMO cached mean error is 0.035135.
Widening KDA only in Core gives 0.034497; widening KDA on both sides gives
0.033726; widening both sides and the head gives 0.029859. Serving start/end
baseline probabilities are exactly equal. No clear throughput loss appears
in this small inference screen, but this says nothing by itself about the
cost of the training backward pass.

## Reproduction and provenance

The committed diagnostic entry points are:

* `scripts/miles/probe_selective_precision.py`: cached teacher forcing, ordinary
  serving timing, Core rescoring and unchanged-weight checks.
* `scripts/miles/selective_precision_runtime.py`: process-local precision
  patches, graph audit and forcing hook.
* `scripts/miles/probe_kda_boundaries.py` and
  `scripts/miles/selective_kda_precision.py`: captured-input chunk/decode replay.
* `scripts/miles/probe_kda_intermediates.py`: intermediate-rounding attribution.
* `scripts/miles/probe_kda_training_cost.py`: bounded backward feasibility/cost.

Every experiment uses a source overlay from a committed revision, records file
hashes and the immutable runtime image, and runs through the MILES committed-image
wrapper. Rendered/submitted specifications and launch files are retained under
Git-ignored `runs/selective-precision-20260926/`; immutable result datasets retain
the numerical data and provenance. These are standalone forward/backward
probes, not new GRPO configurations or an alternative training backend.

| Measurement | Beaker experiment | Result dataset |
|---|---|---|
| Captured KDA replay | [component-r1](https://beaker.org/ex/01M3E191YVV13HVR2ZK1TRNCBM) | `01M3E191ZBCF4EMF5H7JZ5HB6X` |
| Single-site screen | [screen-r1](https://beaker.org/ex/01M3E1M609H1TNM9Q77KBH7YVT) | `01M3E1M60QKQK4T98QYP6WRQKH` |
| KDA FP32 serving collection | [chunk-r1](https://beaker.org/ex/01M3E1V08EE7YT179S4HBZ7BD1) | `01M3E1V08QAQHYB9QQ9TFW7J34` |
| Completed KDA rescoring | [chunk-score-r2](https://beaker.org/ex/01M3E41N0ZHRXPXMWSD0EPK8MG) | `01M3E41N1A9SVAEQDH6K0ZFYJR` |
| Combination serving collection | [combination-r2](https://beaker.org/ex/01M3E3B18GR5E0QBX4TN919NAE) | `01M3E3B18XMS56NVV356NP3172` |
| Combination rescoring/backward | [combination-score-r3](https://beaker.org/ex/01M3E4DVHH3VNWBV63709KKRBM) | `01M3E4DVHWPA52KZ3ZB2XTBK4P` |
| EMO serving/intermediates | [emo-r1](https://beaker.org/ex/01M3E3EAW48J4ZVFJDKMD3FD8N) | `01M3E3EAWD7WFQJEAFHMGK774D` |

The serving-collection jobs for KDA, combinations and EMO were deliberately
stopped **after all inference arms completed**, then rescored in separate jobs.
Their initial Core checkpoint loader oversubscribed host CPUs; limiting CPU
threads to four fixed this without changing retained serving measurements.
An earlier combination setup failed a guard expecting 16 latent blocks; the
model has 15, plus a dense first block. That attempt supplies no comparison
results. No failed or interrupted partial arm is counted as a successful run.
