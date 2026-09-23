# Hero SFT numerical mismatch isolation — September 23, 2026

For the 16-token diagnostic prefix, the HF/SGLang mismatch begins inside the
first routed expert computation, with identical inputs, selected experts and
routing weights. Longer prefixes also expose earlier, much smaller RMSNorm
differences. Replacing serving
arithmetic with the checkpoint's HF arithmetic removes the observed probability
mismatch for both SFT checkpoints. This is a diagnostic intervention, not a
production fix or a relaxed qualification gate.

The checkpoints and native/HF weight audit are described in the
[SFT qualification report](hero-sft-20260923.md). EMO/non-EMO below describes
pretraining ancestry; neither SFT checkpoint has active EMO routing.

## Runs and method

1. [Initial boundary capture](https://beaker.org/ex/01M37XNV3Y8RHH58N0PVR2HCPD):
   captured both checkpoints, but report assembly failed because SGLang calls
   `model.forward` directly, bypassing the root module hook. Its layer traces
   remain usable. The corrected harness hooks the called logits processor.
2. [Same-input replay and substitutions](https://beaker.org/ex/01M37YB09ZEB5FPA67HXSK2VE8):
   both checkpoints completed successfully, source `f9ac03adaa0bb31fa3d507ccc2df2ab807902bd2`.
3. [Expert rounding replay](https://beaker.org/ex/01M37YTZK2R1VZH2418DQ5T3JJ):
   source `8fa9e126e`, stopped before arithmetic because the diagnostic expected
   a shard index, while these exports use a single safetensors file. The reader
   was corrected to support both layouts; the full-model results above are unaffected.
4. [First corrected replay](https://beaker.org/ex/01M37ZJ2D497X5N6PRB6V8V1AC):
   the pinned grouped GEMM rejected FP32 output from BF16 inputs. No numerical
   result is claimed from this attempt.
5. [Successful expert rounding replay](https://beaker.org/ex/01M37ZP0J1M7HKYGTNGVMWTSBE):
   both checkpoints completed successfully, source `b48f1fa62`, using a per-expert
   FP32 down-projection reference with TF32 disabled. A compact immutable subset
   of the first capture (`01M37ZF98KKFMAKFXBXMYFG9RJ`) supplies unchanged inputs
   and routing. The final two probes ran on cached-image Jupiter host
   `jupiter-cs-aus-184.reviz.ai2.in`.

All use immutable image `01M37Q3A7H80X3RWHRXNESKV29`, whose application is
`1c8aa1edcf379b27356e3a17c59772e8e78d21ad`, with committed diagnostic source
embedded in the submitted spec. Dependency pins are unchanged from the SFT
qualification report. Each experiment uses one H100 on Jupiter, positive minimum
runtime, read-only source checkpoint access, and the committed-image launch
wrapper. Raw traces, source and reports are retained in Beaker result datasets.

The controlled forward comparison uses TP1, disabled graphs, no radix cache or
overlap scheduling, prefill chunk 128, and one request at a time. Prefixes are
16 and 81 synthetic token IDs (`5 + i % 30`), plus the 81-token prefix followed
by `[26, 27]`. Each request generates one token. These are full-prefill diagnostic
cases, not a repeat of the broader chunked-prefill/cached-decode qualification.

Hooks preserve inputs before in-place kernels. HF replay supplies each component
with its captured SGLang input, separating local arithmetic differences from
accumulated upstream differences. Substitutions instead run inside SGLang and
replace selected component outputs using HF arithmetic on the current SGLang
input. Model weights and input token IDs stay fixed.

## First divergence and router precision

On the 16-token prefix, both checkpoints match exactly through embedding
normalization, the first dense block (index 0), and layer-1 attention. Inside
layer-1 MoE, latent-down projection, router selections and combine weights match
exactly. The routed expert output first differs: maximum absolute difference
0.0625 for each checkpoint, relative L2 difference 0.003898 for EMO and 0.004102
for non-EMO. The shared expert still matches exactly.

For the same captured hidden states, all 15 routers in each checkpoint select
the same expert IDs and produce exactly matching FP32 logits and combine weights
on all three prefixes: 90 layer/checkpoint/prefix comparisons, all exact.
SGLang stores router weights in BF16 and computes router logits in FP32. Thus
BF16 router storage with FP32 computation is not the initial source of this
HF/SGLang divergence. Later free-running selections differ after the hidden
states have already diverged; top-k slot agreement is order-sensitive and must
not be described as the fraction of distinct experts shared.

On all three prefixes, all 14 KDA attention components match exactly when given
identical inputs. Full-attention blocks 7 and 15 retain local numerical differences.
Occasional RMSNorm differences are much smaller (for example EMO layer-1
post-feedforward norm relative L2 about 4.6e-6), but can also propagate through
later routing decisions.

For the 81- and 83-token prefixes, the first differing boundary is
`model.layers.0.post_attention_layernorm` for EMO and
`model.layers.1.post_attention_layernorm` for non-EMO. The EMO difference has
maximum absolute value 1.5259e-5. Thus the first divergence is input-dependent;
the 16-token experiment isolates the routed experts without earlier norm drift.

## End-to-end interventions

Maximum absolute next-token log-probability difference over the serving result's
top 20 tokens:

| Lineage | Prefix | Original SGLang | HF experts only | HF attention only | HF norms + attention + MLP |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMO | 16 | 0.235563 | 0.073586 | 0.173157 | 0 |
| EMO | 81 | 0.219171 | 0.234490 | 0.289288 | 0 |
| EMO | 83 | 0.266971 | 0.479816 | 0.445111 | 0 |
| Non-EMO | 16 | 0.442460 | 0.119844 | 0.299433 | 0 |
| Non-EMO | 81 | 0.497918 | 0.577418 | 0.425699 | 0 |
| Non-EMO | 83 | 0.141428 | 0.218149 | 0.337335 | 0 |

Replacing the entire MLP yields the same results as replacing only its routed
experts. No single component substitution fixes every prefix, and changes need
not improve error monotonically: later routing depends on the changed hidden
states. Combined substitution also matches log probabilities across the full
vocabulary; replacing the LM head adds no improvement.

These results establish that component arithmetic differences can account for
the measured forward mismatch. They do not establish that any individual kernel
is mathematically incorrect, or that the slow HF substitutions are suitable for
training or serving.

## Isolated expert rounding

The final probe replays only layer-1 routed experts on the 16-token prefix.
Inputs and routes match between the captured implementations. The reconstructed
HF formula reproduces the captured HF expert output exactly for both checkpoints,
validating the reference before any interventions.

Two arithmetic differences are visible in the pinned implementations:

1. HF computes BF16 `silu(gate)` and then a BF16 multiply by `up`. SGLang's fused
   activation keeps the SiLU/multiply intermediate in FP32 and rounds once.
2. HF first rounds the expert down-projection to BF16, then multiplies by the
   routing weight in FP32, sums the weighted experts in FP32 and casts to BF16.
   SGLang applies the routing weight to the down-projection's FP32 accumulator,
   rounds each weighted route to BF16, then reduces the routed outputs.

Changing both choices in the controlled replay nearly reproduces SGLang:

| Lineage | Original relative L2 error versus SGLang | SGLang-style rounding relative L2 error | Exactly matching output elements |
| --- | ---: | ---: | ---: |
| EMO | 0.00389785 | 0.00022749 | 99.8779% |
| Non-EMO | 0.00410058 | 0.00007455 | 99.8535% |

This reduces the L2 discrepancy by approximately 94.2% and 98.2%, respectively.
Neither choice alone achieves this agreement. This is strong evidence that
these two rounding boundaries account for most of the first expert mismatch;
it is not an exact reproduction of the fused kernel. The explicit FP32 reference
uses a different GEMM accumulation order, and rounding its down-projection to
BF16 already leaves relative L2 differences of roughly 6e-5–8e-5 versus grouped
GEMM. Kernel activation approximations and accumulation order remain possible
contributors to the residual. These percentages describe one layer and prefix,
not the fraction of final-model probability error explained.

## Scope and remaining qualification

This experiment compares HF with SGLang. The earlier native/Core/HF audit checked
weight conversion, not the current SFT Core forward pass. The MILES Core factory
uses BF16 parameters, including router parameters, and FP32 router computation;
native FP32 optimizer masters are a separate representation. Casting a BF16
weight to FP32 does not recover precision lost during export. A direct Core
forward comparison is still needed before claiming trainer/serving parity.

The existing probability gate remains unchanged and failed; hero RL remains
unqualified. A production arithmetic alignment would need the full probability
gate rerun, including cached decode, chunking, graphs and both GPU architectures,
followed by Core/serving scoring and a short training exercise.

Focused hook and checkpoint-reader regression tests pass in the exact image
(4 tests); focused Ruff checks and the documentation build pass. No production
arithmetic, runtime pin or checkpoint was changed. Compact machine-readable
results are retained in `hero-numerics-20260923.json`; the full traces remain in
the linked experiments.
