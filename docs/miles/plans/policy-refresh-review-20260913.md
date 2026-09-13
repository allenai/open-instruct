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
cache-slot release. The intended trainer contract is to replay the latest rebuilt prefix routes
and subsequent decode routes in both the trainer scoring and gradient passes.
Historical routes are diagnostic provenance, not replay input for that refreshed
forward. This choice still needs end-to-end trainer qualification.

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


## Direction after review

Proceed with retaining prefix tokens in the ordinary response loss. Do not add
prefix masking merely because the response crossed a policy update. Preserve
the original per-token rollout log-probabilities, including each historical
span, and use the existing MILES token-level PPO/TIS machinery. Recomputed
prefix scores must never overwrite the behavior denominator.

The referenced `open_instruct/grpo_fast.py` belongs to the separate vLLM backend.
The actual MILES implementation was checked in the runtime worktree at
`bc582bc5c680b2138d50cda141322155207a34fa`:

- `miles/backends/training_utils/loss_hub/losses.py` selects either original
  rollout probabilities or detached trainer scores as the PPO denominator.
- With `use_rollout_logprobs=false, use_tis=true`, PPO compares the gradient
  forward with the detached trainer scoring pass; `vanilla_tis_function` in
  `loss_hub/corrections.py` multiplies by the clipped exponent of
  trainer-scored minus original rollout log-probability.
- Alternatively `use_rollout_logprobs=true` uses the rollout denominator
  directly. The wrapper rejects enabling this together with TIS; these are
  alternative configurations, not two corrections to stack indiscriminately.
- Before clipping, the two ratios in the trainer-scored/TIS configuration
  multiply to the gradient-forward probability divided by the original
  behavior probability. Clipping the factors separately is not identical to
  clipping that product. Preserve the existing implementation for the trial.

This fits the existing token-local surrogate, not an exact unbiased correction
of the entire response or group distribution. Historical prefix visitation,
changed future continuations/rewards, and group-relative advantage construction
remain approximations to evaluate. Refresh reduces the age of newly sampled
suffix tokens; it does not guarantee a smaller realized importance ratio at
every token. None of this is a reason to reject the experiment.

For replay, use the newest re-prefill routes for the prefix and the corresponding
decode routes for the suffix. Core already uses the same replay context for
its scoring and gradient passes (`open_instruct/miles/actor.py`). Confirm this
on a refreshed sample, preserving next-token alignment. Compare trainer scores
against fresh inference scores from that same rebuilt forward to assess
numerical mismatch. Compare against original behavior scores separately to
measure policy drift; those are no longer interchangeable diagnostics.

The main remaining changes are transport and scheduling contracts:

- Preserve exact token-span version boundaries through the SGLang response,
  MILES sample conversion, batching, serialization and resume. A list of turn
  versions is not enough. The inspected session merge path currently appends
  a scalar `weight_version`, so serving support alone does not finish this.
- Make staleness rules explicit for historical prefixes and repeated refreshes.
  The existing async buffer computes age from the group's oldest version;
  relabeling the whole response with its latest version would hide old tokens.
- Keep group identities and normal loss masks; qualify refreshed routes and
  ratio/clipping diagnostics separately for historical and fresh token spans.
- Judge speed with full-model publication, re-prefill and completed-group
  throughput, then compare actual RL learning. The queued probes contain no
  optimizer and cannot settle the learning question.
