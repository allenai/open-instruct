# Request for review: pause, publish, re-prefill, continue

## Objective

Avoid waiting for all long-running RL generations to finish before publishing
new policy weights. Retain partially generated responses and the prompt groups
being assembled. Prefer a fast direct GPU publication path over new host-backed
weight snapshots or cohort-to-cohort delivery machinery.

## Mechanism being prototyped

1. Pause inference at a safe scheduler boundary.
2. Retract active requests to the waiting queue, retaining prompt and output IDs.
3. Transfer a consistent new policy directly from trainer GPUs to inference GPUs
   using the existing flattened/bucketed weight-update protocol. Trainer updates
   wait during the transfer so the source tensors cannot change under readers.
4. Flush full-attention KV, KDA recurrent/convolution and reusable prefix state.
5. Re-prefill every unfinished request's retained prefix under the new weights.
6. Continue sampling, keeping the existing response and group identities.
   The trainer can run the next update during re-prefill and generation.

KDA state is updated recursively per token, not recomputed from the whole prefix
at every decode step. Re-prefill reconstructs that history using the new weights.
The retained prefix therefore gives current-policy state for subsequent sampling.
The inference fleet still shares a publication barrier, but waits for a scheduler
boundary rather than completion of every request.

## Statistical question to review

The user's intuition: a retained prefix remains a valid, reachable path under
the new policy; a small policy update simply makes some of its token draws
slightly more or less likely. Most of the rest of the response can now be
sampled under the current policy, and small deviations may be inconsequential.

We agree this is a plausible practical tradeoff. The distinction to preserve is
sampling distribution versus target likelihood. If q sampled a token at 0.50
and new p assigns 0.49, rescoring supplies p but does not change how often that
token appeared in the collected samples. Keeping q and p allows an importance
ratio p/q = 0.98; replacing q with p would conceal the difference. This is a
bookkeeping and empirical-estimator question, not an argument that the path is
invalid or must be discarded.

After re-prefill, newly generated tokens are sampled from the new policy
conditioned on the retained prefix. Prefix contexts themselves have historical
sampling provenance. Repeated switches could accumulate old prefix spans even
if most new tokens use recent weights.

Please assess practical significance, estimator choices and measurable failure
modes without assuming that tiny departures from on-policy sampling make the
approach unusable.

## Information retained

- Original token IDs and per-token rollout scores; at temperature 1 with the
  untruncated sampler these are the behavior log-probabilities.
- Exact sampling-policy version spans within the response.
- Original expert assignments before retraction.
- Recomputed current-policy prefix scores and routes, retained separately.
- Current-policy suffix scores and routes.

SGLang's stock retraction clears its route capture and re-prefill replaces those
rows. Therefore its final route payload must not be interpreted as historical
behavior routing for old tokens. The prototype records original routes before
cache-slot release. Choosing which routes the RL scorer/replay forward should
use remains an explicit trainer-integration decision.

## Implementation and evidence

Isolated worktree `/home/robert/proj/open-instruct/.worktrees/miles-policy-refresh`,
branch `robertb/miles-policy-refresh`; based on engine-drain branch at `33944193a`.
Primary working branches and their defaults are untouched.

Probe: `scripts/miles/policy_refresh_probe.py`.
Probe-only scheduler hooks: `scripts/miles/policy_refresh_hooks.py`.
Offline audit: `scripts/miles/analyze_policy_refresh.py`.
Beaker launcher: `scripts/miles/launch_policy_refresh_probe.py`.
Tests: `tests/miles/test_policy_refresh_probe.py`.

Existing pinned SGLang provides pause/retract, paused cache flush, version spans
and continue. No new serving algorithm has been implemented. The diagnostic
hooks record behavior before cache release and measure resume-to-first-token
inside the scheduler. The probe is **not yet a MILES RL-loop integration**.

Two local six-case matrices (radix disabled/enabled) completed on an RTX 4090
with a tiny KDA + full-attention + latent-MoE hero-shaped fixture. They each
validated 16 interrupted responses with exact prefix/rollout-score retention
and policy-span boundaries. Tests include an unchanged-weight refresh,
explicit KDA key/value perturbations, short/long prefixes, and temperature-1
sampling. Same-token-path scoring errors were comparable in mean to the
unchanged-weight control, with individual BF16/MoE outliers; no claim of bit
exactness. Warm refresh made weights available much earlier than draining,
while long-case total generation time stayed essentially flat.

Full measurements and caveats are in
`docs/miles/measurements/policy-refresh-20260913/README.md`.
Tiny fixture weights are only 95,892 bytes; local GPU IPC is not evidence for
37 GB NCCL transfer performance. No RL quality comparison has run.

Two-GPU probes, each one GPU source + one TP1 engine, submitted on Holmes:

- Tiny: https://beaker.org/ex/01M2CPRSZDTNTX8N5B5GR2DH6H
- Existing SFT checkpoint: https://beaker.org/ex/01M2CPRTVEQ9BDRCDVRMXSV2DF

Both were queued when this handoff was written; inspect current Beaker status
before claiming results. They use immutable image
`01M2CJG5RQQ93GEYNYAS7ASCQJ` and injected committed probe sources recorded with
SHA-256 provenance. The source GPU holds a controlled perturbed checkpoint;
there is no optimizer in these experiments.

## Review questions

1. Is retaining original behavior probabilities plus current rescoring enough
   for a sensible existing GRPO/TIS estimator? Which approximations remain?
2. Should old prefix tokens contribute to loss, be importance-corrected, or be
   masked while training on the newly sampled suffix? What measurements decide?
3. How should router replay handle old-prefix historical routes versus the
   routes that rebuilt current-policy recurrent state? Avoid conflating these.
4. How should per-token lag and repeated refreshes interact with group-relative
   advantages, completed-group queues and maximum staleness?
5. What bounds prevent repeated re-prefill from starving long requests, and how
   do we qualify queued siblings, admission, failures, eval and shutdown?
6. What minimal actual-RL comparison would establish that this is useful, after
   serving correctness and full-model timings are available?
