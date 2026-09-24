# Core-compatible serving: checkpoint history and workload measurement

This measurement compares actual OLMo-core scoring with SGLang behavior probabilities, before any optimizer update. The optional `OLMO_SGLANG_CORE_COMPAT=1` mode is off by default. See [operating instructions](../core-compatible-serving.md).

## Final workload results

All three arms completed successfully. The new mode nearly eliminates **full-prefix** error on the four shared continuations, but only modestly improves probabilities returned during **cached generation**. Keep it off by default: this sample does not justify a 6.5–6.6× generation-time cost for routine RL rollouts. It is useful for controlled fidelity diagnostics; improving recurrent decode agreement and graph support remains separate work.

All error columns are absolute log-probability differences in nats against actual Core. “Cached” compares each mode’s own sampled tokens with Core scores for those same tokens. “Fixed prefix” scores the same four continuations in every checkpoint/mode. The fixed-prefix cohort is a controlled token comparison; cached cohorts can differ between modes. Timing is aggregate generated tokens/second and median wall time for one batch of four 512-token responses (2,048 output tokens), after warm-up.

| Checkpoint | Serving variant | Cached mean error | Cached p99 | Ratio outside ±20% | Fixed-prefix mean error | Tokens/s | Seconds/batch |
|---|---|---:|---:|---:|---:|---:|---:|
| Hero non-EMO base | Default eager (initial run) | 0.03706 | 0.21258 | 1.28% | 0.03476 | 167.0 | 12.17 |
| Hero non-EMO base | Core layout/norm/rounding, before KDA fix (initial) | 0.03672 | 0.20139 | 1.05% | 0.02885 | 142.7 | 14.34 |
| Hero non-EMO base | Default + decode graphs (final) | 0.03810 | 0.22820 | 1.54% | 0.03596 | 998.5 | 2.05 |
| Hero non-EMO base | Core-compatible, including KDA fix (final) | 0.03635 | 0.20844 | 1.11% | 2.33e-07 | 151.7 | 13.48 |
| Hero EMO SFT | Default eager (initial run) | 0.04231 | 0.32679 | 3.79% | 0.04258 | 165.3 | 12.36 |
| Hero EMO SFT | Core layout/norm/rounding, before KDA fix (initial) | 0.04121 | 0.32708 | 3.74% | 0.03773 | 146.8 | 13.91 |
| Hero EMO SFT | Default + decode graphs (final) | 0.04269 | 0.33250 | 3.84% | 0.04303 | 993.7 | 2.06 |
| Hero EMO SFT | Core-compatible, including KDA fix (final) | 0.03938 | 0.31373 | 3.28% | 4.38e-07 | 153.7 | 13.32 |
| Hero non-EMO SFT | Default eager (initial run) | 0.03499 | 0.27612 | 2.55% | 0.04004 | 169.9 | 12.10 |
| Hero non-EMO SFT | Core layout/norm/rounding, before KDA fix (initial) | 0.03280 | 0.25058 | 2.17% | 0.03703 | 141.8 | 14.38 |
| Hero non-EMO SFT | Default + decode graphs (final) | 0.03445 | 0.26494 | 2.28% | 0.04150 | 1002.3 | 2.04 |
| Hero non-EMO SFT | Core-compatible, including KDA fix (final) | 0.03339 | 0.27150 | 2.28% | 4.26e-07 | 151.1 | 13.55 |

Final Core-compatible full-prefix **maximum** errors are 1.66e-6 (base), 1.43e-6 (EMO), and 1.49e-6 (non-EMO). Cached-generation maxima remain 0.55735, 1.30042 and 0.67723 respectively. Non-EMO’s cached p99 slightly worsens and its ±20% outlier fraction is unchanged. Full-prefix fidelity must not be presented as an exact cached-rollout result.

The final graph-enabled baseline and final compatibility mode use the same corrected immutable image and H100 host. The initial eager/prototype rows are retained to show the sequence of changes and approximate the graph cost. Small default-result differences between the initial and final processes are visible; do not attribute them to graph capture alone. Relative to the earlier eager baseline, compatibility lowers throughput by about 7–11%; most of the 6.5–6.6× wall-time increase versus the final ordinary baseline comes from forgoing decode graphs. These are bounded single-host timings, not an end-to-end training slowdown estimate or a multi-seed quality result.

Final qualification passed **71 targeted GPU tests**, the cache/prefix/chunk/weight-update exercise, and the three workload arms. The serving portable suite passed 87 tests with 10 GPU-only skips; the benchmark’s two unit tests passed in the final image. Ruff and documentation build passed. No backward pass or optimizer update ran. [Compact results and provenance](core-compatible-serving-20260923.json).

## Was the mismatch new?

It predates the hero SFT checkpoints. The [earlier latent-KDA Dolci-Think SFT measurement](core-native-routes-20260911.md) already found a mean absolute Core/SGLang log-probability difference of 0.02008 when rescoring retained rollout tokens and a maximum of 0.69929 over 31,905 active response tokens. This was an eager-prefill rescoring comparison, not a comparison against the original cached-decode behavior probabilities. Its prompts, runtime and capture method differ from this study, so those numbers do not establish a checkpoint-size trend.

Here, the hero non-EMO base checkpoint and the two hero SFT checkpoints share the architecture and serving runtime. Scoring the **same four continuations** (2,048 tokens per checkpoint) with default serving gives:

| Checkpoint | Mean absolute log-probability error | 99th percentile | Core/serving ratio outside [0.8, 1.2] |
|---|---:|---:|---:|
| Non-EMO base, step 75500 | 0.03596 | 0.23748 | 2.29% |
| EMO SFT, step 5402 | 0.04303 | 0.33660 | 4.15% |
| Non-EMO SFT, step 5402 | 0.04150 | 0.28136 | 2.93% |

Thus these SFT checkpoints expose somewhat larger average discrepancies and heavier tails on this fixed cohort. Independently sampled continuations do **not** show uniformly larger means: base 0.03810, EMO 0.04269, non-EMO 0.03445. Token selection matters. These are a matched hero-base/SFT comparison, not a controlled comparison against the earlier latent-KDA model, and the non-EMO base is not the EMO checkpoint's direct parent.

The raw base/SFT configs differ only in maximum-position metadata (8192 versus 65536) and absent versus empty RoPE metadata. These models do not use RoPE; all arms use context length 4096 here. The evidence points to existing arithmetic differences interacting with changed weights and activations. It does not identify a particular learned router margin or parameter as the cause of the larger SFT error.

## Checkpoint names and earlier corrective variants

The historical `s002-olmo3moe-instruct-sft-resume-to1000-fused-20260727-hf` is a separate 31-layer, hidden-2048, 64-expert model. It is not the 20-layer latent-KDA Dolci-Think checkpoint whose export is named `olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf`. That latent-KDA model has approximately 18.5B stored parameters and 1.2–1.3B active per token. The hero checkpoints in this study have approximately 12.5B stored and 0.794B active parameters. The earlier “1.2b” name denotes active scale, not total size. A trustworthy s002 error/throughput measurement was not recovered in this investigation; do not relabel the latent-KDA numbers as s002 results. See the [historical-match notes](learning-comparisons-20260911.md#light-sft-historical-match).

The [earlier hero Core/HF diagnostic](hero-core-fidelity-20260923.md) used five short prefixes (three synthetic and two natural), with four greedy generated tokens per prefix. These are **maximum** errors against actual Core, not the workload means above. Runtime was not benchmarked for these diagnostic hooks.

| Execution variant | EMO: max generated-token error, all five | Non-EMO: max, all five | EMO: max, natural two only | Non-EMO: max, natural two only | Throughput |
|---|---:|---:|---:|---:|---|
| Original SGLang, Torch-native full attention | 0.09529 | 0.17331 | 0.07772 | 0.12062 | Not measured |
| Add Core-style activation/expert-combine rounding | 0.11636 | 0.09259 | Not summarized | Not summarized | Not measured |
| Also use eager FP32 RMS norms | 0.13922 | 0.06378 | 0.02778 | 0.03831 | Not measured |

This is the source of the previously quoted **0.078 → 0.028** and **0.121 → 0.038** improvements. They are valid improvements on the natural prompts; the larger synthetic-case EMO error explains why the full five-case result was described as mixed. These partial variants retained SGLang expert GEMMs and did not implement the later Core weight layouts or KDA dispatch change.

A separate **next-token distribution** comparison on those five prefixes used Core's top 20 alternatives:

| Execution compared with actual Core | EMO: max top-20 error | Non-EMO: max top-20 error | Throughput |
|---|---:|---:|---|
| Core gradient-enabled versus Core scoring | 0 | 0 | Not measured |
| HF with `OLMO_HF_MOE_CORE_REFERENCE=1` and SDPA | 0 | 0 | Not measured |
| Ordinary HF, eager attention | 1.27863 | 0.88989 | Not measured |
| Ordinary HF, SDPA | 1.16782 | 0.47903 | Not measured |
| Original SGLang, Triton full attention | 1.27732 | 0.62421 | Not measured |
| Original SGLang, Torch-native full attention | 0.62263 | 0.65546 | Not measured |

The HF flag changes the reference implementation; it does not turn on a serving mode. Replacing serving operations with ordinary HF operations also produced exact HF/serving agreement on three synthetic prefixes, but that is agreement with HF, not a proof of agreement with Core. The [HF/serving substitution report](hero-numerics-20260923.md) records that separate experiment.

## Where the arithmetic diverges

A 673-token real math trajectory exposes the first discrepancy in **the first KDA attention output, before the first router**, despite identical inputs. The initial output difference is small (maximum 0.0009765625 in BF16), then grows through later layers. Routers can amplify upstream differences through discrete expert selection, but FP32 router computation from BF16 stored weights is not the first source in this trace.

After matching expert rounding/layouts, RMS normalization and full attention, this long-prefix comparison still had mean absolute log-probability error about 0.0468 and maximum 1.1262. Splitting the KDA Q/K/V projection, matching convolution activation expressions, and changing its output normalization did not remove it.

The remaining cause was KDA kernel execution: Core dispatches each sequence in its fixed-shape FLA path using `[K,V]` state orientation; serving used packed variable-length dispatch and `[V,K]` orientation. Matching **both** reduced the long-prefix error to a mean of approximately 3.5e-7 and maximum 1.91e-6. Changing orientation alone or dispatch alone did not suffice; replacing a zero initial state with `None` alone had no effect. The adapter now converts states at the cache boundary and retains nonzero prefix states. Cached recurrent decode remains a separate numerical path, so full-prefix agreement does not imply exact rollout agreement.

The earlier tensor audit found the exported BF16 weights identical to the native tensors. This investigation supports an optional serving arithmetic change, rather than an HF export weight correction. The numerical reference is current Core with Torch full attention; it does not reproduce the historical distributed SFT run's Flash4 attention implementation.

## Workload and provenance

Sixteen frozen prompts come from the real prepared RL basket, four each from math, code, IFEval and general tasks. Input lengths range from 52 to 627 tokens. Serving generates 512 tokens per request at temperature 1, top-p 1, batch size 4, ignoring EOS for equal work. SFT arms have two repetitions (16,384 generated tokens each); the base arm has one (8,192). Every retained rollout is teacher-forced through actual Core. Four additional shared continuations separate numerical effects from different sampled tokens.

Generation timing excludes model load, warm-up and Core scoring. Tests use one H100 on `jupiter-cs-aus-219.reviz.ai2.in`, TP1, BF16, no overlap scheduling, no radix cache in the throughput benchmark, and a separate cache/update qualification with caching enabled. This is a small serving workload sample, not a full RL optimizer run or a learning-quality evaluation.

The workload source is `/weka/oe-training-default/robertb/open-instruct/data/full-sft-basket-20260914/train.jsonl`, SHA256 `fde6da774f735ea8d3720598f85ecbd613d5fcf0533f85dd5d8997fedbe93805`. Prompt text and raw token/log-probability artifacts remain in the Beaker results. The benchmark and boundary trace programs are committed under `scripts/miles/`.

The final packaged image is `01M38F6YS8W767GHG7WSW5EX1W`, application `4d9d46a95`, serving `c33ed68dad3a2c2b36281e800b6beee1145e19ad`, OLMo-core `e505356353aa7ce1f6ff83e24d6eb945f463714e`, and MILES `cd0cbe5cc08de85128ed2b56db0c76e6681ef50b`. Docker image ID: `sha256:1625f297560de7f92b04cba719c250d6e2f3566e7198e370ec59ffd41846ee91`.

Runs:

1. Initial workload, before the KDA dispatch correction: [Beaker](https://beaker.org/ex/01M38C88N08MK2NFE66QE0HMKQ).
2. Initial packaged-image short-prefix validation: [Beaker](https://beaker.org/ex/01M38CFMW1V69XEDEX25BQX0WX).
3. Long-prefix boundary trace: [Beaker](https://beaker.org/ex/01M38DD4KGNP13W96K54R9CNZW).
4. KDA component ablations: [Beaker](https://beaker.org/ex/01M38DXETTKPEPZ4VYJS3SKR2M).
5. KDA dispatch/state ablations: [Beaker](https://beaker.org/ex/01M38EBN5WKVMT502P30SD8PB5).
6. Final packaged-image qualification and workload: [Beaker](https://beaker.org/ex/01M38FG3MMENC3J3FVEYVT198S).
