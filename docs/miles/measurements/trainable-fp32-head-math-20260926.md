# Trainable FP32-output head: short math comparison

The opt-in trainable FP32-output head reduced the observed active-token
training–inference log-probability gap by **34.0%** over warmed updates. Trainer
step time rose **1.0%**, accepted response-token throughput fell **0.9%**, and
rank-zero peak allocation rose **1.87 GiB**. This short math screen showed **no
learning-quality benefit**. The option remains off by default.

The [frozen-weight precision study](selective-precision-20260926.md) motivated
the implementation. This comparison uses independently generated training
rollouts, so its numerical percentages also reflect different tokens and policy
trajectories. It does not establish long-run RL quality.

## Implementation and provenance

Set `trainer.fp32_lm_head=true` in a structured MILES run, or
`core.fp32_lm_head=true` in a low-level run. The option enables both Core's
`LMHeadConfig.fp32_output` and the existing SGLang FP32-output head. Default is
false; an explicit conflicting serving setting is rejected.

The CUDA forward keeps BF16 GEMM operands and FP32 output. The custom first-order
backward rounds the incoming gradient to the operand dtype and uses ordinary
low-precision gradient GEMMs. Parameter storage, checkpoint keys and publication
mapping are unchanged. This preserves logits before BF16 output rounding; it is
not full-FP32 training. See [supported boundaries](../core.md#fp32-output-vocabulary-head).

| Source | Identity |
|---|---|
| Open Instruct image source | `392ea5f0451b` |
| OLMo-core | `3c2ad5989f88f8cfb04c983b4243e1a502e4ea1d`, branch `robertb/miles-fp32-head` |
| MILES | `9874d6f589f37b070f15489585325fa32dd5c638` |
| Olmo SGLang | `5514bf5885e690f7cfcd9cfb08e03111fbd70a78` |
| Immutable Beaker image | `01M3E7THRAXQ77QR6S2W94AS6Q` |
| Head qualification | [Beaker](https://beaker.org/ex/01M3E8A8SY17M8Z42JQRVWC7SK) |
| Paired math comparison | [Beaker](https://beaker.org/ex/01M3E8ACHW80E116HFJXPKJPS5) |

Both experiments completed with exit code zero. The paired job executed for
81.1 minutes on four B300 GPUs, approximately 5.40 GPU-hours excluding queue and
node health checks. Its result dataset is `01M3E8ACJ3MN9R5HE32731SHTD`.
The [machine-readable results](trainable-fp32-head-math-20260926.json) retain
per-update summaries, hashes, paired statistics, source pins and benchmark timings.

## GPU qualification and isolated cost

The qualification finished with exit code zero on a B300: 14 Core tests and
145 adapter/configuration tests passed. New tests cover noncontiguous inputs,
bias, autocast, reference forward/gradients, an SGD weight update, frozen-weight
input gradients and compiled full-graph training. Tensor-parallel and the
optional Liger fused-linear loss tests were excluded; those modes are explicitly
unsupported by the new option. An earlier attempt passed the new tests but
failed an unrelated existing fused-loss test because Liger was not installed.

`scripts/miles/benchmark_trainable_head.py` measures the head, default cross entropy
and backward, with width 1,024 and vocabulary 100,278, on PyTorch `2.13.0+cu130`.
Each case has five warmup iterations and 20 measured iterations. Parameters and
their gradients remain BF16 in both cases; output is FP32 only in the enabled case.

| Tokens | BF16 median ms | FP32-output median ms | Difference | Additional peak allocation |
|---|---:|---:|---:|---:|
| 128 | 1.552 | 0.809 | −47.8% | 24.5 MiB |
| 512 | 1.877 | 2.090 | +11.4% | 98.1 MiB |
| 2,048 | 6.280 | 5.793 | −7.7% | 392.0 MiB |

The 128-token timings vary substantially (FP32 case 0.572–2.016 ms), so that
apparent speedup is not reliable. This is one sequential, eager head microbenchmark,
not end-to-end inference or training throughput. Peak allocation includes the
benchmark's retained previous result. The extra logit tensor storage is real;
FP32 output doubles that tensor's bytes. Removing the BF16-to-FP32 logit conversion
before cross entropy can offset some cost, but these timings do not isolate it.

## Math protocol

Both arms start afresh from the non-EMO hero SFT checkpoint at
`/weka/olmo-3p5-checkpoints/scratch/olmo35-fixedtok-sft-20260921/olmo35-fixedtok-sft-20260921-4t-non-emo-dolci-think/emo/step5402/hf`.
The BF16 arm precedes the FP32-output arm in the same four-GPU allocation.
The configuration comparison permits only run identity/output paths and the head
option to differ. Both configurations passed `plan` and `validate`; the rendered
spec has one single-node four-GPU task, one-hour minimum and two-hour timeout.

| Control | Value |
|---|---|
| Training horizon | 32 optimizer updates per arm |
| Data | 1,024 training and 128 disjoint held-out GSM8K questions; seed 17 |
| Dataset revision | `ai2-adapt-dev/rlvr_gsm8k_zs` at `93ffaae6cd2acb8f821f6d4712651320a889b1b9` |
| Evaluation | Same 128 questions initially and finally; greedy, one completion |
| Response/context cap | 4,096 / 6,144 tokens; prompt cap 2,048 |
| Batch | 16 prompt groups × four responses, 64 responses per update |
| Allocation | Two trainer GPUs with EP2, two TP1 inference engines |
| Trainer | Packing, dynamic rows, no activation recomputation, FlashAttention 4, router replay |
| Async | Refresh, maximum policy lag six, TIS, sample backfill |
| Optimizer | Constant LR 1e−6, Adam β=(0.9, 0.95), ε=1e−8, gradient clip one |
| Other objectives | No reference KL, entropy bonus, router auxiliary/z loss or weight decay |
| Serving | Full decode CUDA graphs through batch 32; eager prefill; radix cache |
| Artifact cost | Offline W&B; no checkpoints or HF exports; complete rollout capture |

Zero-standard-deviation groups are filtered. The existing `exclude_truncated`
postprocessor masks capped responses and computes group baselines from finished
siblings, while preserving raw reward metrics. Consequently raw reward can count
an answer embedded in unfinished reasoning; correct-and-finished counts and the
fraction of useful advantages must also be reported. The held-out questions are
a split of the selected source pool, not the official GSM8K test set.

## Observed numerical difference and cost

The warmed window is optimizer updates **9–32**, inclusive. Gaps are mean absolute
differences in sampled-token log probabilities, weighted by active token count.
These active tokens exclude truncated responses. The first collection is scored
before an optimizer update; later collections also include asynchronous policy
age. The two arms generated different rollouts even at unchanged weights.

| Measurement | BF16 head | FP32-output head |
|---|---:|---:|
| First collection mean absolute gap | 0.015017 | 0.007735 |
| Warmed mean absolute gap | 0.018564 | 0.012255 |
| Mean Core optimizer-step time | 10.242 s | 10.346 s |
| Model tokens / Core trainer second | 21,170 | 21,199 |
| Mean driver cycle | 38.948 s | 39.754 s |
| Accepted response tokens / second | 5,403 | 5,354 |
| Unmasked response tokens / second | 2,064 | 2,010 |
| Rank-zero peak allocated memory | 134.68 GiB | 136.55 GiB |
| Mean consumed sample age, updates | 1.089 | 1.047 |
| Maximum consumed sample age, updates | 3 | 2 |
| Active-token TIS clipping | 0.00104% | 0.00120% |

The initial gap was 48.5% lower; the warmed gap was 34.0% lower. Different sampled
tokens, masked responses, learned weights and ages prevent attributing those
exact percentages solely to arithmetic precision. The prior frozen-token study
provides the controlled component evidence. Neither arm discarded stale groups
in this warmed window, and both remained well below the configured lag-six bound.

Trainer work is essentially unchanged after normalization by model-token count.
Mean driver training stages, including orchestration around the Core step, were
11.35 versus 11.43 seconds. Generation wait averaged 24.91 versus 25.81 seconds;
publication averaged 2.68 versus 2.50 seconds. Accepted throughput includes reward
filtering and generation variability. This single fixed-order comparison cannot
resolve a sub-percent inference-kernel cost. Both arms retained full decode CUDA
graphs and the same serving capacity settings.

Both arms started with cold Triton cache fingerprints. The first full training
calls took 518.5 and 484.8 seconds, including standalone scoring and kernel
compilation; these are excluded from warmed timing. Worker stack samples showed
FlashAttention and KDA compilation during the baseline's cold first call.

## Held-out math outcome

Every entry below has the same denominator of 128 questions. "Correct and
finished" requires both verifier success and a completed response.

| Outcome | BF16 before → after | FP32-output before → after |
|---|---:|---:|
| Raw correct answers | 72 → 78 | 73 → 78 |
| Correct and finished | 59 → 60 | 57 → 57 |
| Capped responses | 65 → 64 | 65 → 69 |
| Mean response tokens | 2,797 → 2,761 | 2,782 → 2,844 |

Final raw accuracy tied at 60.94%. Final correct-and-finished accuracy was 46.88%
versus 44.53%, with FP32 already starting two questions lower. The final paired
finished-answer comparison had 16 FP32-only successes and 19 BF16-only successes:
difference −2.34 percentage points, paired bootstrap 95% interval over questions
[−11.72, +7.03], exact McNemar p=0.736. The interval resamples these questions;
it does not measure variation across training seeds.

There is no convincing quality gain in this run. The cap left about half the
greedy responses unfinished, and raw-reward gains mostly came from unfinished
answers. Raw mixed-reward filtering precedes truncation exclusion: removing the
capped siblings can leave a batch with all-zero advantages. This happened on
BF16 updates 12 and 22, and FP32 updates 10, 14, 21, 25 and 30. All gradients
were finite; the other 30 and 27 updates had nonzero gradient norms. Adam still
executed all 32 steps in each arm; zero-gradient steps can still move weights
through momentum. One BF16 preclip gradient norm exceeded one; none did in the FP32 arm.
Different batches limit interpretation of that difference.

## Completion and reproducibility

The retained audit verifies optimizer steps 1–32 on both trainer ranks, policy
publications 0–32, both complete workflow manifests, and all driver stages passing.
All four evaluation panels contain exactly the same 128 unique prompts, with
policy version zero initially and 32 finally. Direct GSM8K verification matches
every stored evaluation reward. Prepared training, evaluation and verifier hashes
match across arms; the 1,024 training and 128 held-out source rows are disjoint.

The Beaker result contains both submitted configurations, logs, training contracts,
timings, flow records, data manifests and `eval-audit.json`. Full rollout dumps
remain under the two WEKA run roots. The CPU evaluation audit source is retained
as `audit-evals.py` in the result. To reproduce the aggregate analysis:

```bash
beaker dataset fetch 01M3E8ACJ3MN9R5HE32731SHTD --output /tmp/head-study/comparison-results
python docs/miles/measurements/trainable-fp32-head-math-20260926/analyze.py /tmp/head-study
```

Local validation additionally passed 154 targeted configuration/documentation
tests, targeted Ruff/format checks and the type check against the pinned Core
source. The generated reference and MkDocs build were checked. The new option's
GPU tests and this non-EMO EP2 run establish the exercised path; normalized heads,
tensor-parallel wrapping, fused-linear loss and higher-order CUDA gradients remain
outside its supported scope.
