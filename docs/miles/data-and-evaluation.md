# Data, rewards and evaluation

The structured workflow prepares data in the GPU job where mounts and tokenizer
are available. It records preparation under the run root; do not place reference
assistant solutions into training prompts. See [workflow](workflow.md) for paths.

## Selecting data

Choose exactly one mode under `[data]`:

| Mode | Contract |
|---|---|
| `[[data.tasks]]` | Named tasks with train/eval counts and optional prompt wrapper |
| `rl_manifest` | Adopt an immutable prepared olmo-miles-compatible manifest and supported verifier bindings |
| `prompt_data` | Prepared JSONL; supply reward_config and optional alternating dataset-name/eval-path entries |
| `recipe` | Rejected: the olmo-miles named recipe catalog is not ported |

Current named tasks are `gsm8k`, `math`, legacy `ifeval`, and generated
`multiplication`. Dataset IDs/revisions live in `open_instruct/miles/run_data.py`.
The [multitask example](https://github.com/allenai/open-instruct/blob/fe4d9f2bdc994adb35f839718d86e420d8481e12/configs/miles/examples/medium.toml) mixes
GSM8K and math; it is not the complete published Olmo 3 mixture.
Use manifest adoption and the [mixed-task qualification](measurements/mixture-qualification.md)
for broader data. Unsupported task/verifier contracts fail rather than silently
substituting a different judge or reward.

The trusted verifier registry names factories and configurations; individual
samples select registered names, targets and weights. See
[prepared fixtures](https://github.com/allenai/open-instruct/blob/fe4d9f2bdc994adb35f839718d86e420d8481e12/configs/miles/verifiers.json). Code verification may
need an externally provisioned service. GPU judges have an explicit preparation,
service and rubric contract in [managed judges](managed-judges.md); launch does not
automatically supply every Olmo 3 external service.

`prompt_wrapper` supports the task-specific contracts in the preparation code.
Review rendered prompts and answer format after changing it. A familiar task
name or equal split count does not prove identical questions or reward semantics.

## Online group filtering

MILES/Core training defaults to online reward-group filtering:

```toml
[training]
filter_zero_std_groups = true
```

The low-level equivalent is `core.filter_zero_std_groups`. The compiler resolves
this to `miles.dynamic_sampling_filter_path =
"miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"`;
`plan` records the effective native path. The native filter keeps a complete prompt
group only when the standard deviation of its selected reward scores exceeds
`1e-8`. All-0, all-1 and other constant-reward groups are discarded before trainer
packing/scoring. Mixed groups retain every response, including zero-reward ones.
This requires more than one training response per prompt. Evaluation is unfiltered
and may still use one response per prompt.

Generation continues until the configured number of accepted groups fills the
collection; global batch size counts retained responses. This works with barrier,
engine-drain and mixed-policy refresh publication. Async filtering occurs before
completed-buffer admission, after checking group provenance; staleness is still
checked at consumption. Filtered groups are dropped, even when
`async_unused_samples_handler="retry"`, and retired from the checkpoint retry
ledger. Their prompts remain eligible on later dataset passes.

Filtering saves trainer work but still pays generation and verification costs.
A low acceptance rate increases collection time; if every group is rejected,
collection waits for useful data and cannot advance the policy. The tiny `dev`
and `small` mechanics examples therefore explicitly set this switch to false.
Monitor `rollout/dynamic_filter/drop_zero_std_0.0`,
`rollout/dynamic_filter/drop_zero_std_1.0` and the corresponding counters for other
constant rewards, together with trainer wait time and accepted task proportions.
These are per-report drop counts, not accuracy estimates. Use unfiltered held-out
evaluation to assess learning.

Zero group-relative advantage removes the reward-driven policy-gradient signal,
but the discarded responses would still affect loss normalization and any enabled
KL, entropy or router auxiliary objectives. Treat enabling filtering as a recipe
change and compare learning at matched wall time or generated-token budgets.
Do not change filtering in-place and call the run a resume.

Offline preprocessing remains compatible and can reduce online rejection rates.
The existing [correctness filter](../../scripts/data/rlvr/filter_existing_dataset_correctness.py)
scores saved completions and retains prompts within inclusive average-score bounds.
Its default `[0, 1]` bounds retain all prompts; use bounds strictly inside that range
to exclude all-wrong/all-correct prompts for binary rewards. This is a snapshot of
an earlier policy, so newly sampled groups can still have constant rewards after
preprocessing. Online filtering continues to check the current groups. For graded
rewards, average-score filtering and zero-variance filtering are different tests.

Advanced custom native filters require `core.filter_zero_std_groups=false` to
avoid conflicting with the built-in selection. They remain unsupported with
refresh or engine-drain publication; those modes allow only the built-in filter.

## Held-out evaluation and comparison

Specify eval_count per task, or explicit held-out files. The structured defaults
use greedy evaluation, one response per question, initial evaluation and periodic
evaluation every 20 collections when held-out data exists. Examples can override
these values; inspect plan. Eval shares rollout engine admission and cache limits.

Retain question IDs, rendered prompt/token IDs, generations, rewards, lengths,
cap hits, policy versions and source/config provenance. Compare update zero on
the same checkpoint before interpreting learning differences. For matched studies
use the same immutable held-out set, template, sampling, response limit and reward
implementation. Examine paired answer changes rather than just aggregate scores.

Rollout dumps are trusted tensor artifacts on WEKA; do not load arbitrary external
pickle files. The small Beaker reports are an index into the larger retained
artifacts, not a replacement for per-sample auditing. See [operations](operations.md)
and [comparison evidence](measurements/gsm8k-results-20260911.md).


## Code-service failure policy

Code rewards default to Open Instruct's zero-reward continuation behavior.
After existing HTTP retries are exhausted, a transport/gateway failure or
invalid service reply receives zero reward, logs a warning, and leaves the
rollout running. This is a fallback reward, not proof that the generated code
failed its tests. Existing HTTP rejection handling is unchanged.

Set `OI_MILES_CODE_FAILURE_POLICY = "raise"` in `[launch.env]` for strict
service-failure handling, or set `failure_policy` to `"zero"` or `"raise"` in
the code verifier's trusted JSON config. An explicit verifier setting takes
precedence over the environment. Configuration errors and cancellation still
propagate; known-answer preparation canaries use strict mode.

Per-sample diagnostics distinguish `service_error`, `rejected`, and successful
grading. W&B records `rollout/code_verifier/service_errors` and
`rollout/code_verifier/service_error_fraction`, with status breakdowns alongside
the existing rejection counters. The fraction covers code verifier calls in
the consumed collection; it is not the fraction of HTTP attempts or all
samples generated. Retain these metrics when comparing learning curves.

### Symbolic math timeouts

Symbolic math runs in a bounded subprocess pool. An individual request exceeding
45 seconds kills and replaces that worker and defaults to reward zero. The sample
retains `verifier_diagnostics` with `kind=math`, `status=timeout` and elapsed time;
`rollout/math_verifier/timeouts` and `timeout_fraction` count these separately from
ordinary incorrect answers. Set `OI_MILES_MATH_TIMEOUT_POLICY=raise` for strict
qualification. Configuration errors, unexpected worker failures and cancellation
still propagate. A timeout-zero is an ungraded sample, not evidence of a wrong
mathematical answer; include the timeout rate when interpreting benchmark scores.
