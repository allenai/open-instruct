# Characterizing retained-response policy refresh

The user has authorized implementation, debugging and thorough characterization,
including an outcome in which the approach is not beneficial. Do not equate a
successful serving probe with correct training or a useful learning tradeoff.
Keep this work isolated on `robertb/miles-policy-refresh` and its MILES source
branch `robertb/policy-refresh`.

## Evidence already available

- Local tiny KDA/full-attention/latent-MoE tests, radix off/on, unchanged and
  changed weights, greedy and temperature-1 continuations.
- Two-GPU tiny and full-SFT serving probes completed; original token/logprob
  retention and exact version boundaries audited on 16 interrupted responses
  per matrix.
- The full checkpoint's direct 37 GB transfer takes about 1.8–2.2 seconds.
  Short refresh is inexpensive; a long-prefix case took 9.5 seconds to restart
  and finished its batch slower than the drain control. Profile and repeat it
  before attributing that delay to compilation or treating it as steady state.
- Natural top-k order differs in the unchanged-weight short control as well as
  changed-weight cases. Report expert-set overlap alongside positional matches;
  actual trainer replay must be verified separately.

## Stage 1: actual trainer contract

Run EP2 plus two TP1 engines with the existing SFT checkpoint and frozen GSM8K
preparation. Four optimizer updates, saves/evals at two and four, then a fresh
process resumes for update five. Use temperature 1, no sampling truncation,
radix enabled, graphs disabled, TIS, retained responses and replay diagnostics.

The refresh run is `01M2CSXN60DP5DX0EHFTTKHNP5`, committed overlay `8885f4d2c` on
immutable image `01M2CJG5RQQ93GEYNYAS7ASCQJ`. It was queued when this plan was
written. Do not infer success from this run identity.

Require all of the following:

- At least one mixed-policy response actually reaches an optimizer step.
- Original per-token behavior probabilities and normal prefix loss masks survive
  sample conversion, DP partitioning, and training. No group is consumed twice.
- Every behavior span is within the lag limit at consumption; a recent suffix
  cannot relabel an old prefix.
- Trainer scoring and gradient/recomputation replay the returned route table
  with zero intervention mismatches. Metadata boundaries align with the next-
  token scoring convention.
- At least one group has nonzero policy advantage; optimizer clocks advance on
  both ranks and saved state resumes at the next update.
- Shared-engine eval, saturated-buffer draining, save, resume and final shutdown
  complete. Partial requests are regenerated after process restart using the
  existing cursor contract, not serialized as reusable inference state.
- The independent retained-artifact audit passes. This is still not proof of
  equal learning quality or exact next-step numerical resume equivalence.

Run the current barrier mode with matching model, sampling, resources, objective
and diagnostics. That path cancels/retries unfinished requests at publication;
call it the **barrier baseline**, not a drain-only control. The serving probe's
finish-before-publication control answers the separate drain comparison.

## Stage 2: performance and freshness

After correctness, use longer matched runs with expensive replay diagnostics off
in both arms, sparse saves, matched evaluation cadence and warmed caches. Keep
both initial/cold costs and warm intervals. Start with roughly 20 updates per arm
before deciding whether a 100-update learning comparison is justified.

Report:

- Wall time to a fixed number of optimizer updates, completed groups and useful
  trained tokens; time to first update separately.
- Trainer scoring/backward, publication pause/transfer/load/flush, generation
  wait, prefill/restart, checkpoint and evaluation time. Summed asynchronous
  component times are not additive wall time; use timestamps for overlap.
- Generated versus trained versus retried/discarded tokens and groups, queue
  occupancy and oldest-token lag, fraction sampled at the consuming version.
- Number of refreshes per response, retained prefix lengths and cumulative
  re-prefill tokens. Identify long-request starvation or a refresh feedback loop.
- p50/p90/p99 restart and group latency; avoid reporting only cheap short cases.
- Short/long prefixes, repeat refreshes, cold/warm kernels, radix off/on. Keep
  same-weight updates as a control for cache rebuild and numerical variation.

## Stage 3: policy drift and learning

Retain original behavior scores, token-version spans, latest replay version,
rewards and generations. The trainer logs historical-prefix and latest-forward
ratio/clipping diagnostics separately. A latest-forward span can still be stale
relative to the trainer; report both clocks.

Measure absolute log-ratios, ratio tails/clipping, age and fraction of historical
prefix tokens, entropy/response length, cap frequency, reward and held-out GSM8K
accuracy. Inspect paired generations on the same held-out questions at update
zero and later snapshots. Analyze group-relative reward variance, because mixed
policy siblings may alter the advantage distribution.

For numerical mismatch, compare forwards on exactly the same token path and
weights/routes; greedy continuations that diverge after a near tie cannot be
compared tokenwise as if they were the same sequence. Policy drift and inference-
trainer numerical mismatch are different measurements. Small local ratios do
not establish that whole-prefix/state-distribution effects are absent.

Sampling order and asynchronous completion selection can differ between arms.
Record effective configs and data identities; do not claim exact batch parity.
If learning diverges, add focused ablations (refresh frequency/lag, replay,
prefix masking as an investigation rather than the default) and a second seed
before attributing a cause. Route changes alone are not proof of harmful drift.

## Scope and decision

The first implementation deliberately retains the existing PPO/TIS estimator,
FIFO completed-group queue and conservative oldest-version lag rule. It does not
add cohort delivery, host-backed snapshots, persistent mixed-request resume,
multi-turn tools, speculative decoding or engine replacement.

The outcome may be: useful as-is, useful only above certain request lengths or
concurrency, requires batching/cache improvements, or not beneficial. Report
that outcome with observed limits and retained artifacts; do not promote a
production default based on a single short run.
