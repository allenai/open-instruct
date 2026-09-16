# Dense GSM8K, full test: MILES/Core versus original Open Instruct after 200 updates

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

The same-checkpoint, same-data, same-recipe dense pair from the
[learning-confidence campaign](learning-confidence-20260914/README.md) scored on
the official GSM8K test split (1,319 questions) on one serving stack. The
original framework's final policy improved; the MILES/Core final policy did not.
The two finals differ by 64 questions in a paired comparison. This is the first
completed same-model framework comparison in the campaign and it does not show
preservation of learning.

Evaluation job: [01M2NDNN6X6S5WW9CJ0KS36PQH](https://beaker.org/ex/01M2NDNN6X6S5WW9CJ0KS36PQH),
eight B300 SGLang TP1 engines, image `01M2F1RKZFZVJYAS0XQGEC3SEJ`,
`scripts/miles/gsm8k_test_eval.py` with the serving context raised to 34,816 and
`--max-new-tokens 32768` to match training ([spec](gsm8k-dense-test-20260916/beaker-spec.yaml)).
All three checkpoints were rendered and served with the Core run's prepared
tokenizer and chat template, checked against every prepared held-out row, so
prompt bytes are identical across checkpoints. Greedy plus four temperature-1
samples per question. Per-question correctness, finish reason and token counts:
[per-question.csv](gsm8k-dense-test-20260916/per-question.csv);
[summary.json](gsm8k-dense-test-20260916/summary.json).

| Checkpoint | Source | Greedy correct /1319 | Truncated at 32K | Mean tokens | Sampled pass@1 | pass@4 |
|---|---|---:|---:|---:|---:|---:|
| Start | Olmo 3 Think-SFT `6ff857587e…` (Core run `prepared/hf`) | 1130 (85.67%) | 8.7% | 4,597 | 0.917 | 0.961 |
| Core, update 200 | [01M2KEPRMD3NGSJEV9GM7VR0HF](https://beaker.org/ex/01M2KEPRMD3NGSJEV9GM7VR0HF) `export-hf` (r2→r3→r4 chain) | 1114 (84.46%) | 9.7% | 4,904 | 0.919 | 0.963 |
| Original, update 200 | [01M2K1SFCHC33S3R4JF9PWAF0F](https://beaker.org/ex/01M2K1SFCHC33S3R4JF9PWAF0F) public export | 1178 (89.31%) | 8.2% | 4,463 | 0.941 | 0.969 |

Paired greedy comparison, exact two-sided McNemar on the discordant questions:

| Pair | Only first correct | Only second correct | Net | p |
|---|---:|---:|---:|---:|
| Start → Core 200 | 79 | 63 | −16 | 0.21 |
| Start → Original 200 | 57 | 105 | +48 | 0.0002 |
| Core 200 → Original 200 | 50 | 114 | +64 | 6e−7 |

The change is concentrated in the truncation tail of greedy decoding: 56 of
Core's 79 losses are responses that now hit the cap, and 59 of the original's
105 gains are questions that were capped at the start and now finish. Sampled
responses never hit the cap on any checkpoint, and the sampled pass@1 tells the
same story with the tail removed: the original gained 2.5 points, Core 0.2.

## What matched and what did not

Both arms: Olmo 3 Think-SFT, the same 6,000 prepared training rows with
byte-identical prompt tokens ([input parity](learning-confidence-20260914/original-input-parity.json)),
16 prompts × 4 samples per update, one optimizer step per collection
(`num_mini_batches=1`, `num_epochs=1`; Core `optimizer_steps_per_collection=1`),
synchronous (original `async_steps=0`, `inflight_updates=False`; Core
`publication_mode=barrier`, `max_policy_lag=0`, no TIS), learning rate 1e-6
constant, no warmup, Adam β 0.9/0.95, no weight decay, no KL, centred advantages
without standard-deviation normalisation, clipping 0.2/0.28, temperature 1,
32,768-token responses in a 34,816 context, seed 17, 200 updates.

Training reward was indistinguishable throughout (Core 0.92–0.93 per 25-update
block, original 0.90–0.95; both at zero truncation and 2–5% all-wrong groups),
which is why this difference was invisible until the held-out test.

Known differences, from the campaign's own inventory:

- **Loss reduction.** The original reduces the clipped policy loss as a token
  mean over the batch's response tokens; Core's `calculate_per_token_loss` was
  off, so it averages per response and then over responses. Under a token mean
  a long response carries proportionally more gradient. On this task the
  failures are long: the tail that the original learned to finish is exactly
  what a token-weighted objective penalises hardest.
- Serving: vLLM eager (original) versus SGLang with decode CUDA graphs (Core);
  trainer/rollout log-probability gaps were similar (0.01 mean on both).
- Trainer: DeepSpeed ZeRO-3 with gradient checkpointing versus OLMo-core with
  sequence packing to 34,816 tokens and the standalone scoring pass.
- Verifier: the historical GSM8K verifier in the original image versus the
  current `GSM8KVerifier`; both are last-number matches.
- The original was preempted after update 145 and resumed from its update-125
  checkpoint, so its updates 126–145 ran twice; the resumed pass is the one in
  the final model.

## What this does and does not establish

The [overfit diagnostic and its controls](overfit-20260915.md) show Core's
update path is sound: reward rises on a fixed set, reverses under a negated
reward, and is flat at a null learning rate. This result therefore points at the
objective or the numerics, not at a dead gradient. The single-knob experiment
that separates the leading hypothesis from the rest is the Core GSM8K arm rerun
with `calculate_per_token_loss = true` and nothing else changed. If it recovers
the original's gain, the framework difference is a reduction convention and the
MILES default should follow the original. If it does not, the serving and
trainer numerics are next.

One seed per arm; the McNemar tests are paired within seed and do not estimate
between-seed variance. The start checkpoint was served once for both arms, so
cross-engine greedy noise (one to four questions on 32 to 512 rows in earlier
records) does not enter the head-to-head.
