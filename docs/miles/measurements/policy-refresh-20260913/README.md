# Live request policy refresh prototype

Isolated branch: `robertb/miles-policy-refresh`, based on the experimental
`robertb/miles-engine-drain` branch at `33944193a`. This is a serving/publication
probe, not an enabled training configuration. No primary branch, default, or
ongoing colleague experiment is changed.

## Mechanism

Use the pinned SGLang `pause_generation(mode="retract")`, publish the next
weights directly from a GPU through the existing flattened NCCL loader, flush
all cached state, and `continue_generation`. Pending requests retain their
prompt and generated token IDs. Their next forward prefills the entire retained
prefix under the newly published policy. KDA recurrent and convolution state,
as well as full-attention KV state, must be rebuilt.

The pinned SGLang scheduler already supports these primitives, including
policy-version spans within one response and cache flush while paused with
requests waiting. The prototype does not introduce a new scheduler or CPU
weight snapshot. The two GPUs are one controlled GPU weight source and one TP1
inference engine. There is no optimizer and no claim about end-to-end RL speed
or learning quality.

## Probability and router bookkeeping

A retained prefix is a valid path under the new policy. Re-prefill supplies new
conditional state and can score retained tokens under that policy, but does not
change the distribution that originally sampled them. Small updates may make
this a small effect; measure it rather than silently overwriting behavior data.

Keep these distinct:

- Original sampled token IDs and their original behavior log-probabilities.
- Actual behavior-policy version spans, including the exact interruption offset.
- Original routing decisions for the retained prefix.
- Recomputed current-policy prefix log-probabilities and routes.
- Newly sampled suffix log-probabilities and routes under the current policy.

Stock retraction preserves output log-probabilities but clears `routed_experts`.
A probe-only hook captures routes **before** request cache slots are released.
Final route capture can then reflect the recomputed prefix. Passing that mixed
record into the existing strict single-policy router-replay path without an
explicit contract would mislabel the old routes. This prototype does not do so.

## Measurements and checks

Run four simultaneous requests through a short-prefix drain control, a
short-prefix refresh, and a 1024-token-prefix refresh. Retain raw responses,
pre-retraction metadata/routes, and empty-cache reference continuations.

Measure:

- Waiting for drain versus pausing/retracting.
- Full direct GPU weight transfer and cache flush separately.
- Time from resume to the first newly generated token, including re-prefill.
- Completed batch wall time and the fraction of output tokens actually sampled
  after the publication boundary.
- Prefix token and behavior-logprob preservation, version spans, and fresh-prefill
  token/logprob comparisons.
- Current-policy rescoring minus original behavior scores for retained tokens.

Instrumented pause includes bounded route copying to CPU for the audit; record
that time separately. This is request metadata, not a CPU model copy.

The controlled weight perturbation is a diagnostic fixture, not an actual RL
update. Greedy requests make continuation comparisons easier but are not a
measurement of stochastic training-policy bias. Warm-up precedes measured cases.
Decode CUDA graphs are initially disabled to isolate lifecycle correctness.

## Launch

Launch only committed source through the usual wrapper:

```bash
MILES_EXISTING_IMAGE=01M2CJG5RQQ93GEYNYAS7ASCQJ \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_policy_refresh_probe.sh tiny
```

Use `sft` instead of `tiny` for the existing SFT HF checkpoint. Add `--radix`
after the mode for a separate radix-enabled qualification. GPU placement is
Holmes, urgent, `ai2/open-instruct-dev`, minimum runtime one hour for the full model. The disposable tiny fixture uses
a 10-minute minimum / 20-minute timeout to allow short backfill. Each run uses
its own Beaker result dataset. Injected probe sources and their SHA-256 hashes
are recorded in `provenance.json`; runtime image and source overlays are fixed.

## Progress

- Four focused local tests passed: pre-release capture, scope cleanup on an
  exception, isolated GPU placement/provenance, invalid mode rejection.
- Initial queued attempt `01M2CN14KDG9WS9TMG3MDSAT6P` was stopped before GPU
  placement to correct the diagnostic capture ordering.
- Tiny attempt `01M2CN77CSHAAXXDT5JE3NWY6B` submitted; no result claimed yet.

## Remaining integration work if measurements justify it

Preserve partially built prompt groups in the MILES producer; add token-level
version/lag handling instead of the existing single-policy response contract;
choose explicitly whether router replay uses historical behavior routes or
current-policy rescored routes, and keep scoring consistent with that choice.
Qualify multiple refreshes, queued versus active siblings, cache-enabled
operation, evaluation, failure/timeout and shutdown. Then run a small actual RL
comparison with retained behavior data and measured importance ratios. A
serving-only success cannot establish the RL estimator's behavior.

### First local GPU result

The tiny hero-shaped hybrid (KDA + full attention + latent MoE) completed the
three cases on the local RTX 4090 using same-GPU IPC for the 95,892-byte fixture.
All eight interrupted responses preserved their sampled prefix tokens and
original output log-probabilities, and their two version spans matched the
exact observed pause offsets. All 128 checked greedy continuation tokens
matched fresh-prefill references; maximum compared log-probability error was
0.010671. Prefix route agreement with fresh-prefill references was near, but
not exactly, 100%; see `local-initial-transition-audit.json`. This is a BF16
batch-versus-single-request comparison, not bit-exact routing qualification.

The short case waited 0.474 s for draining versus 0.0121 s from interruption to
weights ready with refresh. About 84.8% of its tokens were sampled after the
switch. These tiny-model IPC timings are not a real-model NCCL speed claim.

The first long case exposed measurement overhead: repeated streaming prompt
scores delayed the requested interruption until token 452/512. Its client-side
first-token timing also included JSON processing. The next revision skips
prompt scores on streaming requests and records first-new-token time inside the
scheduler. The old measurements are retained as an explicit initial result,
not presented as clean long-prefix re-prefill performance.

Debugging also found the pinned SGLang metadata expression
`x if x < vocab_size - 1 else 0` in `_process_input_token_logprobs`: it relabels
the last valid vocabulary token as zero. The numeric score is unaffected by
that expression. The probe reports and permits exactly this known label case;
any other label mismatch fails. No serving computation was patched to hide it.

The diagnostic hook initially used obsolete direct `Req` log-probability
fields; the pinned runtime stores these under `req.logprob`. This was fixed
locally before any Beaker probe obtained GPUs. Queued attempts
`01M2CN77CSHAAXXDT5JE3NWY6B`, `01M2CNKAN2NBSYAQ4M8CDZSY6B`, and
`01M2CNG15KWH0ZKABKYZNR6SPA` were stopped before execution. They supply no GPU
validation evidence.

### Key/value perturbation and radix qualification

Revision `b5b9f6a58` additionally perturbs **KDA keys and values** (not just
queries), scaling alternate input columns by 1 +/- 1/64. This deliberately
changes the recurrence and tests rebuilding state. It is a controlled fixture,
not an RL optimizer update. An unchanged-weight refresh supplies a numerical
control. The six-case matrix now contains short/long drain controls,
unchanged-weight refresh, short/long changed-weight refresh and temperature-1
sampled refresh. Complete same-token-path teacher forcing avoids interpreting
logprobs from different greedy continuations as numerical error.

Both local radix-disabled and radix-enabled matrices completed. Each audited
16 interrupted responses with exact token-prefix/original rollout-score
preservation and version boundaries. Original routes and freshly recomputed
routes are retained separately. CUDA decode graphs were disabled.

| Local tiny case (radix disabled) | Boundary to weights ready | Scheduler resume to next token | Tokens generated after switch |
| --- | ---: | ---: | ---: |
| Short drain | 0.494 s | n/a | n/a |
| Short refresh, changed K/V | 0.0094 s | 0.0070 s | 86.3% |
| Long drain | 0.791 s | n/a | n/a |
| Long refresh, changed K/V | 0.0107 s | 0.0085 s | 73.6% |
| Sampled refresh | 0.0092 s | 0.0073 s | 86.9% |

The first unchanged-weight refresh paid a cold-path pause/resume penalty
(0.239 s until weights ready, 0.242 s resume-to-token); do not omit that cost
when considering cold runs. Short control total batch time also includes cold
batch compilation. Long total batch time was essentially unchanged (1.0747 s
versus 1.0752 s); refresh changes **when the new policy becomes available**, not
necessarily total generation throughput. Tiny 95,892-byte same-GPU IPC results
cannot establish a 37 GB trainer-to-engine transfer time.

Mean absolute same-path suffix logprob differences were 0.0057–0.0070 for
changed-weight cases versus 0.0063 in the unchanged-weight control. Individual
outliers reached 0.191 nats. This supports comparable numerical behavior, not
bit-exactness or proof that every discrepancy is harmless. New-prefix route
agreement with fresh prefill was 99.7–100% in the changed-weight cases; agreement
with historical routes was lower (93.1–96.5%), confirming the audit distinguishes
recomputed routes from original decisions.

For the temperature-1 sampled case, 268 retained tokens had a mean absolute
rescore-minus-original logprob difference of 0.0171 nats. Per-token probability
ratios ranged from 0.846 to 1.133. These are fixture measurements, not estimates
of actual RL-step drift. Greedy cases use original raw model logprobs for
numerical checks; they are not stochastic-policy sampling-bias measurements.

Retained compact results: `local-kv-summary.json`,
`local-kv-transition-audit.json`, `local-kv-prefix-ratios.json`, and
`local-radix-summary.json` / `local-radix-transition-audit.json`.
Full token, route, reference and teacher-forced records remain in the local
`/tmp/policy-refresh-local-kv` and `/tmp/policy-refresh-local-radix` directories.

Revised two-GPU Beaker probes submitted with exactly the committed sources:

1. Tiny hybrid: [Beaker](https://beaker.org/ex/01M2CPRSZDTNTX8N5B5GR2DH6H).
2. Existing SFT checkpoint: [Beaker](https://beaker.org/ex/01M2CPRTVEQ9BDRCDVRMXSV2DF).

At this writing both await Holmes placement. No cross-GPU result or full-model
speedup is claimed. Five focused CPU tests, explicit script Ruff checks, and
repository `make style` / `make quality` pass. These are prototype/training
probes, not the repository GPU pytest certification.


## Two-GPU serving qualification completed

Both isolated jobs completed with exit 0: tiny
[01M2CPRSZDTNTX8N5B5GR2DH6H](https://beaker.org/ex/01M2CPRSZDTNTX8N5B5GR2DH6H)
and full SFT
[01M2CPRTVEQ9BDRCDVRMXSV2DF](https://beaker.org/ex/01M2CPRTVEQ9BDRCDVRMXSV2DF).
Each six-case matrix retained and audited 16 interrupted responses. Original
prefix tokens, behavior logprobs and exact policy boundaries all passed.
Compact summaries, route audits and source provenance are retained alongside
this document as `beaker-{tiny,sft}-*.json`.

Full SFT results (37,028,386,304 bytes, 35 NCCL buckets):

| Case | Boundary to weights ready | Resume to next token | Request batch wall time |
| --- | ---: | ---: | ---: |
| Drain short | 14.834 s | n/a | 22.652 s |
| Refresh unchanged | 1.783 s | 0.196 s | 16.599 s |
| Refresh changed, short | 1.881 s | 0.147 s | 16.685 s |
| Drain long | 43.244 s | n/a | 52.606 s |
| Refresh changed, long | 2.192 s | 9.472 s | 60.658 s |
| Refresh sampled, T=1 | 1.873 s | 0.164 s | 41.849 s |

These are individual cases, not warm steady-state throughput estimates. The
long case demonstrates the real tradeoff: much earlier publication, but more
prefill work and a slower completed batch. The 9.472-second restart is observed;
its compute/compilation breakdown has not been profiled.

Expert routes are not generally bit-exact across batching/prefill executions.
The unchanged short control had 74.2% positional top-k agreement but 97.4%
expert-set overlap with a fresh reference prefill. The changed short case had
74.1% positional agreement and 97.1% expert-set overlap; the long case matched
exactly. The unchanged control matters when interpreting these differences.
The next qualification must check actual trainer replay, rather than inferring
trainer equivalence from natural-route agreement alone.

No optimizer ran in either serving probe. These results do not establish RL
quality or complete the training integration acceptance gate.
