# Hero SFT Core/HF fidelity investigation — September 23, 2026

This follows the [HF/SGLang numerical isolation](hero-numerics-20260923.md)
and asks which execution represents the training policy. The checkpoints are
both corrected-tokenizer 4T Dolci Think step5402 exports; EMO/non-EMO describes
ancestry, not active SFT routing. The [checkpoint audit](hero-sft-20260923.md)
contains their paths, architecture, and exhaustive native/HF tensor checks.

## Source evidence and intended arithmetic

The saved native configurations identify SFT source
`444025b6de8013025c351d9c0015b760ef866217`. Export receipts identify
`bdfdcbfd092b631d9c7214bacb115a5040210612`. The bundled HF modeling file has
SHA256 `81c4427030be8a6e982d131ed7615d44c4f2747df6bc4c537cd2b9430703913a`,
exactly matching that export revision's file. These are distinct from the
current MILES Core pin `e505356353aa7ce1f6ff83e24d6eb945f463714e`.

1. **BF16 forward weights, FP32 router projection are intentional.** The
   [actual SFT model initializer](https://github.com/allenai/OLMo-core/blob/444025b6de8013025c351d9c0015b760ef866217/src/olmo_core/nn/ddp/model.py#L907)
   casts the model to BF16, as does `apply_dp`. The saved `dtype=float32`
   configuration and FP32 optimizer master tensors do not describe forward
   parameter precision. The router explicitly promotes inputs and weights for
   FP32 projection. The
   [HF router precision change](https://github.com/allenai/OLMo-core/commit/8cef5cb85b6408acaf852d83b68f3b548c5a53d0)
   explains the intent and adds an exact projection/routing regression test.
   Equivalent comments and arithmetic are present in the actual exported file.
2. **Core training rounds SiLU before multiplication.** The eager routed-expert
   path evaluates `up * F.silu(gate)` with BF16 tensors. Our current runtime's
   [scoring correction](https://github.com/allenai/OLMo-core/commit/290d2ca4521373bef0bf7fe4244673cc79dcc004)
   makes its fused no-gradient path retain that intermediate rounding, with an
   exact BF16/FP16 regression test. This is an inherited MILES runtime change,
   not independent evidence of an upstream endorsement. The original SFT
   revision's no-gradient fast path lacked this option; its eager training
   expression already had the rounding being preserved.
3. **FP32 weighted expert accumulation is intentional.** The
   [HF correction](https://github.com/allenai/OLMo-core/commit/0cda42877a8257d154fb53e144dc247e5081c21c)
   preserves FP32 routing weights and combines BF16 down-projection outputs in
   FP32 before one final BF16 cast. Its comment explicitly identifies Core's
   Transformer Engine unpermutation as the target. A fractional-weight unit
   check guards against premature BF16 rounding. This change is an ancestor
   of the actual exporter revision.
4. **Ordinary HF execution is not a bitwise Core oracle.** The
   [exported implementation](https://github.com/allenai/OLMo-core/blob/bdfdcbfd092b631d9c7214bacb115a5040210612/src/olmo_core/nn/moe/v2/hf/modeling_olmo3moe.py#L355)
   documents different GEMM layouts and accumulation order, and supplies
   `OLMO_HF_MOE_CORE_REFERENCE=1` for a controlled conversion check. It reconstructs
   Core's permutation, packed/grouped GEMMs and unpermutation using the exported
   HF weights. It also matches packed shared-expert and first dense-block GEMMs.
   The [strict converter](https://github.com/allenai/OLMo-core/blob/bdfdcbfd092b631d9c7214bacb115a5040210612/src/examples/olmo_ddp/hero_hf_convert.py#L128)
   additionally matches attention backends and forces math SDPA. This is a
   conversion oracle, not the normal standalone HF implementation.
5. **These SFT exports did not originally pass a forward-parity gate.** Their
   [export policy](https://github.com/allenai/OLMo-core/blob/444025b6de8013025c351d9c0015b760ef866217/src/examples/olmo_ddp/olmoe3_hero_4t_eval_policy.py)
   explicitly waived numerical parity and retained structural/finiteness checks.
   That historical waiver does not waive our current RL qualification gate.

| Operation | Core eager training / intended HF semantics | Original SGLang fused experts |
| --- | --- | --- |
| Router | BF16 stored weights; FP32 projection, scores and combine weights | Same on identical inputs in prior measured cases |
| SwiGLU | Round SiLU to BF16, then multiply and round | Fused SiLU/multiply with one final rounding |
| Expert down projection | BF16 output before routing weight | Routing weight applied to FP32 GEMM accumulator |
| Expert combination | FP32 weighting and accumulation, final BF16 cast | BF16 weighted route outputs, then reduction |

Matching these choices means matching the trained finite-precision computation;
a fused expression with fewer roundings can be closer to real arithmetic while
being further from the training policy.

## Direct experiment

The diagnostic uses the actual pinned Core factory and imported checkpoint
weights, without replacing Core operators with HF implementations. It compares
ordinary HF eager attention, HF SDPA, and HF with Core expert layouts plus SDPA;
Core no-gradient scoring and gradient-enabled training forwards; and SGLang
Triton/torch-native attention. The gradient forward performs no backward pass
or optimizer update. A one-rank NCCL group supplies Core router load-balancing
collectives; auxiliary loss weights are zero.

Five prefixes comprise the previous synthetic lengths 16, 81 and 83 plus a math
question and a Python coding question formatted with the checkpoint's chat
template. Serving generates four greedy tokens per prefix. Core teacher-forces
each generated sequence, allowing comparison of the actual returned behavior
log probabilities with the current trainer's scores on those same tokens.
This is distinct from comparing top-20 alternatives on the first token.

The optional serving prototype retains SGLang's Triton expert GEMMs and uses
its unweighted `no_combine` output. It adds BF16 SiLU intermediate rounding,
then FP32 routing-weight multiplication/reduction and one final cast. A second
variant also uses the eager FP32 RMSNorm expression for embedding, final and
block normalization. Original and modified variants run within the same engine
with identical weights; a discarded priming request installs diagnostic hooks.
This is a TP1, unquantized BF16 diagnostic, not a production implementation.

All requests fit within one prefill chunk (1024); graphs, radix caching and
scheduler overlap are disabled. Four-token generation includes cached decode,
but does not qualify variable batching, chunk boundaries, graph replay, TP/EP,
long contexts, backward propagation or a training update. Original SFT used
64 GPUs and Flash4 attention; this test targets the current MILES Core policy
with Torch attention, not bitwise reproduction of that historical distributed
training execution. The earlier exhaustive tensor audit verified the expected
vocabulary trim from 100352 native padded rows to 100278 tokenizer rows. The
current Core adapter uses that exported vocabulary; this comparison does not
measure the original padded-vocabulary softmax denominator.


## Measured results

Both checkpoints completed all four baseline/prototype arms; the experiment
exited zero. HF with Core expert layouts plus SDPA produces **identical raw
last-token logits and identical selected-token log probabilities** to Core
scoring in every case. Core's gradient-enabled forward is also exact against
scoring. Layer replay on lengths 16 and 81 is exact at every captured boundary
when HF uses Core expert layouts. There are five original prefixes per arm and
five extended sequences; repeated backend/variant generations produce duplicate
extended sequences, not additional independent coverage.

Ordinary HF first differs from Core inside the layer-1 feedforward computation
on both traced prefix lengths and both checkpoints, with identical block input
and attention output. Its first post-feedforward-norm output differs by at most
0.00390625 (EMO) and 0.0009765625 (non-EMO across these two lengths). Matching
expert layouts removes this and all later measured drift. This extends the
previous HF/SGLang isolation: even after preserving the same arithmetic
rounding boundaries, different expert execution layouts remain consequential.

Maximum absolute log-probability error on **Core's top 20 tokens**, over the same
five original prefixes:

| Execution compared with Core scoring | EMO ancestry | Non-EMO ancestry |
| --- | ---: | ---: |
| Core gradient-enabled forward | 0 | 0 |
| HF, Core expert layouts + SDPA | 0 | 0 |
| Ordinary HF, eager attention | 1.278627 | 0.889887 |
| Ordinary HF, SDPA | 1.167817 | 0.479027 |
| Original SGLang, Triton attention | 1.277320 | 0.624209 |
| Original SGLang, torch-native attention | 0.622634 | 0.655460 |

These are not the preceding report's serving-top-20 metric. Large alternative-token
log-probability errors do not mean equally large errors on likely generated tokens.
For original Triton serving, maximum prefill KL(Core || serving) is 0.002252 and
0.015560 nats, respectively. On the two natural prompts specifically, maxima are
0.000472 and 0.003500. This small sample does not estimate a deployment distribution.

For the actual four generated tokens, maximum absolute Core/behavior log-probability
differences are 0.095289 (EMO) and 0.173309 (non-EMO) for either original attention
backend. Across both arms/backends, `exp(logp_Core - logp_serving)` ranges from
0.886375 to 1.189234. For the natural prompts alone, the range is 0.886375–1.080817.
Thus token agreement conceals a measurable policy mismatch, while the much larger
alternative-token maxima should not be presented as generated-token errors.

## Does a small serving arithmetic change solve it?

The unchanged torch-native serving baseline repeats exactly in the prototype
engine. Every variant generates the same four tokens for every prompt, allowing
like-for-like comparisons. The changes preserve the checkpoint weights and use
SGLang's expert GEMMs rather than invoking HF expert modules.

| Variant, torch-native attention | EMO top-20 max | Non-EMO top-20 max | EMO generated-token max | Non-EMO generated-token max |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.622634 | 0.655460 | 0.095289 | 0.173309 |
| Core-style activation/combine rounding | 1.590799 | 0.561851 | 0.116357 | 0.092594 |
| Same + eager FP32 norms | 0.589767 | 0.531002 | 0.139222 | 0.063781 |

The combined prototype reduces the natural-prompt generated-token maxima from
0.077717 to 0.027778 for EMO, and from 0.120615 to 0.038312 for non-EMO. It does
not consistently improve the synthetic cases: EMO's length-16 generated-token
maximum worsens. Nor does it pass the existing 0.1 alternative-token gate.
Changing attention backend alone also helps some cases and hurts others.

These interventions support the arithmetic diagnosis, but **do not establish a
sufficient production fix**. Small residual GEMM/reduction/norm/attention
perturbations can change later routed computation; correcting one operation
need not improve final probabilities monotonically. This run does not isolate
each remaining prototype residual, and the hook implementation is unsuitable
for graph capture or production scheduling.

## Recommendation

1. **Keep the checkpoint weights and BF16 router representation.** Exhaustive
   weight conversion passed, the actual SFT forward used BF16 weights, and the
   HF Core-layout reference now exactly matches the current Core forward.
   There is no evidence here of a tensor-export or architecture bug. Ordinary
   HF is useful and semantically faithful interchange, but is not a strict
   numerical oracle for this training policy.
2. **Make actual Core scoring/training forward the serving qualification target.**
   Retain the HF Core-layout reference as a conversion-control test. Record
   full-distribution KL and actual behavior-token log probabilities/importance
   ratios alongside alternative-token maxima. Keep Core scoring/gradient-forward
   agreement as a separate invariant. Do not redefine the old gate as passed.
3. **Develop an explicit Olmo/Core-compatible serving mode.** Preserve BF16
   SiLU and down-projection boundaries and FP32 combine weights/reduction, then
   align expert packing/GEMM layouts and remaining norm/attention execution
   against Core using layerwise controls. The measured partial prototype is
   evidence for this work, not a patch to promote unchanged. Optimize only after
   establishing a reliable numerical reference and quantifying the tradeoff.
4. **Accept only measured residual off-policy error deliberately.** Bitwise
   agreement across every GPU, batching pattern and kernel is not a prerequisite
   in principle; silently assuming these backends implement the same probability
   policy is unjustified. Broader representative rollouts, likelihood ratios,
   KL and training stability must justify any tolerance or off-policy correction.
   The current sample does not justify either declaring the mismatch harmless or
   predicting training failure. Keep hero RL unqualified pending the broader
   cached-decode/chunking/graphs/hardware checks and a Core/serving training smoke.

No production arithmetic, checkpoint, dependency pin or acceptance threshold was
changed. Seven focused diagnostic tests, Ruff checks and the documentation build
pass. The compact [machine-readable results](hero-core-fidelity-20260923.json)
retain per-prefix values; full reports, source and logits are in the experiment's
immutable result dataset.

## Provenance and harness corrections

The final matrix and arithmetic probe run in
[experiment 01M3827ZJPS1RQD67BNHCXMCCH](https://beaker.org/ex/01M3827ZJPS1RQD67BNHCXMCCH),
source `29203e8af62e3c57b37bc507ca583b1c8d928c9e`, on one H100 in Jupiter with
128 GiB host memory, 32 GiB shared memory, a 30-minute minimum runtime and a
60-minute timeout. The immutable image is `01M37Q3A7H80X3RWHRXNESKV29`;
application and dependency pins match the preceding numerical isolation report.
The spec embeds committed diagnostic sources; the committed-image wrapper
submits it. The inspected spec, sources, runtime lock and reports are retained.
The result dataset is `01M3827ZJYNV9CVKJA3EYY4KZQ`.

Earlier attempts are not numerical evidence:

- `01M3812NCXQKH08EM0NZ39CA9R`: queued on a full pinned node; stopped before execution.
- `01M3815WVN7FXGWYN8XXDPEKXG`: Transformers chat-template return shape required
  explicit `return_dict=False`; synthetic serving cases ran, then the harness stopped.
- `01M381K9VQXBBPPNB222GEPQRJ`: Core scoring ran, but the training-forward check
  required assigning the initialized one-rank group to routers explicitly.
- `01M3824R08EEFZDEW1YJWD7A7N`: stopped early after source inspection identified
  the dense block's null router; the correction uses Core's `routed_blocks()`
  iterator, matching its data-parallel setup.

The completed diagnostic subprocesses emit an unrelated Python multiprocessing
resource-tracker `KeyError` at shutdown, after writing reports; each exits zero.
It is not a model-forward exception.
