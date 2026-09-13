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
