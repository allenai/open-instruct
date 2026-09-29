# Forced-exit stopping guidance

This is an experimental, **opt-in training objective**. Ordinary configurations
have `core.forced_exit_positions=0` and use the existing GRPO rollout/loss path.
No maintained example enables it. Inference from the trained checkpoint needs no
special stopping controller: the model generates its own closing delimiter.
Disabling this objective stops further stopping supervision; it does not undo
what a checkpoint has already learned.

## Enable

Add these settings to a compatible synchronous run in ignored `runs/`:

```toml
[core]
publication_mode = "barrier"
scoring_pass_required = true
router_aux_loss_weight = 0.0
router_z_loss_weight = 0.0
reward_zero_truncated = true
reward_final_answer_only = true
forced_exit_positions = 5
forced_exit_parents = 1
forced_exit_trials = 3
forced_exit_answer_tokens = 1024
forced_exit_coefficient = 0.1
forced_exit_probe_interval = 0 # Optional: 8 captures labeled hidden states every 8 updates

[miles]
calculate_per_token_loss = false
rollout_function_path = "open_instruct.miles.rollout.forced_exits.ForcedExitRollout"
loss_type = "custom_loss"
custom_loss_function_path = "miles.backends.core_utils.stopping.policy_loss"
```

This is an overlay, not a complete run configuration. Keep at least two natural
responses per prompt and the normal GRPO group-centered reward calculation. The
pilot used four natural responses, five paragraph cuts on one uniformly chosen
parent, and three answers per cut. Both arms disabled
`training.filter_zero_std_groups` to retain all prompt groups; match this choice
between control and treatment. The answer trials are labels and never enter the
natural GRPO group or its behavior-policy importance correction.

A separate clipped objective trains the probability of the **full closing tag**,
including multi-token delimiters. Each nonzero cut advantage requires an extra
teacher-forced prefix forward/backward context; the method has real generation
and training cost. Zero-advantage cuts skip this auxiliary computation.

Run `python -m open_instruct.miles plan runs/YOUR_RUN.toml` and `validate` before
launching. Validation rejects async/refresh publication, conflicting hooks,
incompatible reward postprocessors and unsupported objective settings. Readiness
capture requires context parallelism one and a supported unpadded layout.

## Comparative stop/continue experiment

`core.forced_exit_mode="comparative"` selects an experimental replacement for
the trace-relative labels above. It requires `forced_exit_positions=2`,
`forced_exit_guidance="first_token"`, synchronous barrier publication, binary
rewards, no zero-standard-deviation filtering, and the rollout hook
`open_instruct.miles.rollout.comparative_exits.ComparativeExitRollout`.
The existing custom stopping loss hook remains in place. The matching MILES
runtime must include conditional branch training; older images are incompatible.

The rollout samples fresh stop and continue groups at the same prefix, under
the same policy and total response cap. Continue suppresses tokens containing
`</` for exactly one generated token, then restores normal generation. This
includes alternate merged closing-token spellings and also suppresses unrelated
closing markup for that one token. The suppressed-choice token is masked from
branch training; no `Wait` text is inserted.

The closing advantage is mean stop reward minus mean continue reward, without
normalization or significance filtering. An optional `forced_exit_tie_bonus`
adds a fixed bonus only when observed successes tie exactly, stop accuracy meets
`forced_exit_tie_min_accuracy`, and mean total remaining tokens favor stopping.
This empirical tie is not proof of equal true accuracy. The default bonus is zero.

Both action groups also train their sampled completions with separately centered
rewards. The inherited prefix and forced delimiter are masked; continue's first
constrained token is also masked. The remaining continuation tokens, including
thinking, are trained. Constant-reward action groups have zero answer advantage.
The native prompt GRPO groups and main batch denominator are unchanged. Auxiliary
branch loss averages over each action group, cuts and selected parents; its
coefficient is independent of the closing-guidance coefficient. TIS uses sampled
token behavior scores only, and routed expert replay is retained for completions.

Parent selection mixes uniform and capped thinking-length weights. A separate
stop-only screen selects one candidate, and another is random. Fresh scoring
samples use distinct seeds. Candidate paragraph locations currently follow the
parent's paragraph quantiles. A hard per-update parent limit bounds ordinary
probing; uniform natural-close audits occur separately, once per represented
task on each audit round. These audits are independent of outcomes and may skip
parents that never closed.

The rate halves every configured interval to a floor. A one-sided normal screen
over independent audited parent differences can double it when continuing is
better; old risk evidence suspends tapering but only a fresh audit doubles it.
The audit history expires by update age. This approximate screen controls
acquisition, never the sign of training advantages. Controller state and cut
labels are saved by rollout index under `forced-exits-v2/`, so resumed collections
read the preceding collection's state. Missing history falls back to the scheduled
rate and an empty audit window.

Generated-token/request-time metrics include screens and both branches. Request
seconds overlap and are not GPU-seconds. There is no calibrated GPU-time cost
controller yet: sample counts and the per-update cap must be reviewed against
measured throughput before a learning launch. CPU tests do not qualify the GPU
generation, replay, or distributed auxiliary schedule.

## Disable

Set `core.forced_exit_positions=0` (or remove it), and remove these three
`[miles]` overrides so their normal defaults apply:

- `rollout_function_path`
- `loss_type`
- `custom_loss_function_path`

Leftover forced-exit hooks with zero positions are rejected explicitly. Do not
set the coefficient to zero as an off switch: it must be positive while enabled.
`forced_exit_probe_interval=0` disables hidden-state capture only, not stopping
supervision. You can retain the two reward gates independently for a matched
control; their defaults are false, and changing them changes reward semantics.

## Evidence and limitations

A single matched-update Math/GSM8K pilot completed training and unforced
evaluation. Its [archived qualification record](https://github.com/allenai/open-instruct/blob/8ae0e322f89d5f8dd00ef407c59f577e8e739cad/docs/miles/forced-exit-stopping.md#evidence-and-limitations)
contains the image, runs and measurements. This is not a replicated or
matched-compute result. Conditional readiness and mechanism claims require
further audits; separate sampling/entropy tooling is not required for training.

An exit's success is correctness **within its answer budget**: truncated answers
always score zero. Monitor `forced_exit/truncation_rate`; a high rate can obscure
readiness. This pilot has no position-decaying reward and does not independently
sample continuation value at every cut. Three training answer trials per cut
are noisy labels, not a calibrated answer-entropy measurement.

Use an application image containing this implementation. Local configuration and
source changes do not update an older Beaker image. The integration adds clearer
disabled-hook validation; build from the integrated branch for that guard.
