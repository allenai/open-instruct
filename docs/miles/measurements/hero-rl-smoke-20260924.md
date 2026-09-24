# Hero SFT end-to-end MILES smoke — September 24, 2026

Both corrected-tokenizer hero SFT checkpoints completed four MILES optimizer
updates on H100 with automatic fused rounding, nonzero policy gradients,
changed weights, exact live weight-transfer checks and clean shutdown. This
qualifies this short **barrier-publication mechanics configuration**, not learning
quality, long responses, mixed-policy refresh, checkpoint/resume or active EMO
routing. EMO/non-EMO denotes pretraining ancestry; both SFT policies use ordinary
routing. The earlier numerical studies remain relevant to residual probability
mismatch; passing this smoke does not imply exact Core/SGLang parity.

Runs:

1. EMO SFT: [Beaker](https://beaker.org/ex/01M38Z03FB011TG8S858H466TV).
2. Non-EMO SFT: [Beaker](https://beaker.org/ex/01M38Z04TZNZRT7ZVAECQJMN8Y).

The [machine-readable record](hero-rl-smoke-20260924.json) retains submitted
configurations, exact source pins, checkpoint paths, result datasets, job/node
identities, all four ranks' optimizer contracts, probability profiles, publication
records, timing and cache outcomes. Beaker results include `run.log`, the submitted
run, completed `workflow.json`, per-rank contracts and driver/publication logs.
Rollout tensors and offline W&B files remain under the recorded WEKA output roots.

## Configuration and runtime

- Immutable image: **`01M38YYQFVP5EBXRG9D83RW3CQ`**; application
  `d5b60f2ebbd2c086f3c4710036e0c4ce83723001`, Core
  `e505356353aa7ce1f6ff83e24d6eb945f463714e`, MILES
  `cd0cbe5cc08de85128ed2b56db0c76e6681ef50b`, olmo-sglang
  `5514bf5` (full revision in the JSON runtime lock).
- One Jupiter node / five H100 GPUs per run: four EP4 trainer ranks plus one
  disaggregated TP1 serving engine. One Beaker task, minimum runtime one hour,
  timeout two hours, both model and output WEKA mounts. No multi-node launch.
- The paired full 12.5B 4T Dolci Think SFT exports at step5402 from the
  [checkpoint audit](hero-sft-20260923.md), with their own tokenizer/templates.
- BF16 serving, ordinary SGLang attention/KDA, full decode graphs, eager prefill,
  Triton attention, radix cache disabled. The compatibility environment flag was
  **unset**; both logs confirm `Olmo Core compatibility mode: rounding`.
- Core Torch attention, activation recomputation, microbatch one, no packing,
  router auxiliary coefficients zero, learning rate 1e-6. Four prompts × two
  responses per update; response cap 256, context cap 1024, seed 17.
- Identical GSM8K prompt data in both runs, with the existing synthetic mixed-reward
  fixture. Zero-variance filtering disabled so mechanics batches cannot starve.
  All 32 training responses per run reached the cap: 8,192 active response tokens.
  Synthetic held-out evaluation ran before training and after updates two/four;
  its rewards are **not GSM8K accuracy**.
- Barrier publication, offline W&B, rollout capture enabled. No checkpoint saving
  or HF export. `diagnostic_interval=1` and `check_weight_update_equal=true`
  exercise a snapshot/reset/republish/exact-compare audit after every update.
  The existing mean-absolute log-probability guard remained **0.05**.

Both configurations were copied from `small.toml` into ignored
`runs/hero-rl-smoke-20260924/`, then adjusted for full-checkpoint memory. `plan`
and `validate` passed, the rendered immutable specs were inspected before
submission, and both jobs used the committed-image wrapper. Retained metadata
confirms five GPUs in one task per run. Launcher/topology tests: 21 passed;
portable compatibility tests including the dtype regression: 54 passed, 34 CUDA
cases skipped locally. The actual H100 runs passed the GPU attention preflight.

## Results

| Check | EMO SFT | Non-EMO SFT |
| --- | ---: | ---: |
| Job exit / workflow state | 0 / complete | 0 / complete |
| Optimizer updates, on every trainer rank | 4 | 4 |
| Training samples / active response tokens | 32 / 8,192 | 32 / 8,192 |
| Exact weight checks: initial plus four trained policies | 5 passed | 5 passed |
| Published policy versions | 0–4 | 0–4 |
| Mean absolute trainer/behavior log-probability gap, update 1 | 0.02959 | 0.02699 |
| Update 2 | 0.03183 | 0.02751 |
| Update 3 | 0.03039 | 0.02745 |
| Update 4 | 0.03465 | 0.02677 |
| Largest individual token gap over these updates | 0.73391 | 1.19860 |
| Total gradient norm range | 0.7717–1.2334 | 0.8283–1.7151 |

Each rank recorded finite, nonzero dense/expert/router gradients with no missing
parameter gradients, and nonzero sampled parameter changes in those categories
on every update. Parameter-change probes sample at most 256 entries per named
parameter; they do not inspect every element. The independent publication checks
cover the serving weight tensors exactly, including reset before re-publication.
Each transfer carried 23,441 tensors / 24,992,380,160 bytes in 24 flattened buckets.
Nine publications per run comprise initial version zero plus two transfers for
each trained version, the second belonging to the diagnostic round trip.
Subsequent training batches recorded behavior versions 1, 2 and 3, establishing
that generation continued under newly published weights. Final evaluation used
version four. Every recorded driver stage passed.

The scalar policy loss can be zero at unchanged weights with centered GRPO
advantages. The nonzero gradients and parameter changes above establish actual
optimization. Auxiliary losses were zero, so they cannot explain those gradients.

The numerical guard is a **mean**, despite its historical
`max_train_rollout_logprob_abs_diff` name. Individual token gaps remain much
larger, as the table shows. These are Core scores versus recorded cached-generation
behavior probabilities on the same tokens, not separately sampled responses.
No guard was relaxed and no claim of learning benefit follows from these values.

## Timing and limits

| Stage | EMO | Non-EMO |
| --- | ---: | ---: |
| Job execution, excluding queue/image pull | 9m46s | 9m44s |
| Serving startup | 172.9 s | 172.9 s |
| Trainer startup | 34.7 s | 34.4 s |
| First training stage, including cold scoring/backward compilation | 171.7 s | 171.5 s |
| Median training stage, updates 2–4 | 1.43 s | 1.47 s |
| Median normal trained-weight transfer | 1.33 s | 1.35 s |
| Median publication stage including the full diagnostic round trip | 21.53 s | 19.67 s |

These short runs deliberately pay repeated snapshot/reset/equality-check costs.
The roughly 20-second publication stage is diagnostic overhead, not the normal
transfer cost. Cold startup/compilation dominates wall time. Use the
[separate fused-rounding benchmark](fused-rounding-20260923.md) for serving
throughput comparisons; this is not a throughput or learning study.

Non-EMO published compiler-cache entries successfully. EMO's best-effort cache
publication rejected conflicting KDA autotune JSON entries from the concurrent
runs; training and shutdown still succeeded. This limits cache-reuse evidence,
not the recorded optimizer/publication results. No cache-reuse success is claimed
for EMO.

## Default-selection correction found by the smoke

The initial attempts ([EMO](https://beaker.org/ex/01M38YDYZJAPW38TRRAKX7Z69E),
[non-EMO](https://beaker.org/ex/01M38YFGHBNGZSDNP9YQ5X4SK9)) were deliberately
stopped before training after the EMO log selected ordinary mode. The selector
checked `ServerArgs.dtype`, which MILES leaves as `auto`; SGLang had already
resolved the model to BF16. Serving commit `5514bf5` passes the loader's actual
construction dtype into selection and validation. Explicit opt-out still wins,
and FP16/FP32 resolved models remain outside the default profile. The arithmetic
kernels were unchanged. Regression cases cover auto dtype and explicit overrides;
the successful retries above confirm automatic rounding in actual launches.

That fix is pushed to olmo-sglang `main`, and Open Instruct's primary branch pins
it. Earlier default-selection image `01M38YCEFHDKRR7X166XYSJW9D` contains the bug;
use the corrected image above or rebuild the current runtime lock.

## Continuing from here

The next learning experiment should use real rewards and an appropriate response
budget, retaining behavior probabilities and checking their tails as well as
means. Mixed-policy refresh needs its own end-to-end exercise with actual
cross-version responses; these barrier runs do not establish it. Longer contexts,
other hardware/topologies and save/resume also remain separate qualification.
See [serving modes](../core-compatible-serving.md) for explicit ordinary, fused,
tensor-control and full-reference options and their numerical tradeoffs.
