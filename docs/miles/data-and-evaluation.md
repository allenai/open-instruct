# Data, rewards and evaluation

The structured workflow prepares data in the GPU job where mounts and tokenizer
are available. It records preparation under the run root; do not place reference
assistant solutions into training prompts. See [workflow](workflow.md) for paths.

## Selecting data

Choose exactly one mode under `[data]`:

| Mode | Contract |
|---|---|
| `[[data.sources]]` | Separate source datasets with explicit allocation fractions summing to 1 (Core GRPO) |
| `[[data.tasks]]` | Named tasks with train/eval counts and optional prompt wrapper |
| `rl_manifest` | Adopt an immutable prepared manifest and supported verifier bindings |
| `prompt_data` | Prepared JSONL; supply reward_config and optional alternating dataset-name/eval-path entries |
| `recipe` | Rejected: named recipes are not supported |

Current named tasks are `gsm8k`, `gsm8k-less-noise`, `gsm8k_original`, `math`, legacy `ifeval`, and
generated `multiplication`, and `synlogic-sudoku`. Dataset IDs/revisions live in `open_instruct/miles/datasets/run_data.py`.

`gsm8k` uses the established `ai2-adapt-dev/rlvr_gsm8k_zs` train split.
`gsm8k_original` remains a compatibility alias for the same pinned dataset,
retaining its historical sampling seed convention.

The maintained dev example uses `gsm8k-less-noise`, which selects
[`techarb/gsm8k-cleaner`](https://huggingface.co/datasets/techarb/gsm8k-cleaner), an
**informally cleaned variant that has not been rigorously validated**. It has 39
corrected targets and omits 93 problems judged ambiguous or unreliable, leaving
7,380 rows. Each prepared row records `metadata.original_row`, its index in the
original dataset.

For fresh preparations that previously selected the cleaned data as `gsm8k`, use
`task = "gsm8k-less-noise"`. It retains that variant's sampling seed convention;
task names and prepared sample IDs identify it explicitly. Equal seeds across the
original and cleaned datasets do not select the same problems. Existing prepared runs retain their immutable
data on resume; keep their original configuration rather than renaming the task
inside an existing preparation contract.

Fresh preparations select training rows first, independently of `eval_count`,
then select held-out rows from the remainder. This can change selections compared
with older preparations that sampled both partitions together. Existing prepared
runs reuse their recorded data unchanged. For named source datasets, the `train`
component in `prepared_sample_id` identifies the source split, including when the
row is assigned to this run's held-out partition; it is not a training assignment.

The [single-node example](../../configs/miles/examples/node.toml) takes prepared
math, instruction-following and code JSONL with a verifier registry, without
judge-dependent tasks; its placeholder paths do not identify the complete published Olmo 3 mixture.
Use manifest adoption when starting from dataset manifests. Unsupported task/verifier contracts fail
rather than silently substituting a different judge or reward.

The trusted verifier registry names factories and configurations; individual
samples select registered names, targets and weights. Preparation writes the
resolved registry to `prepared/data/verifiers.json`. Code verification may
need an externally provisioned service. GPU judges have an explicit preparation,
service and rubric contract in [managed judges](managed-judges.md); launch does not
automatically supply every Olmo 3 external service.

`prompt_wrapper` supports the task-specific contracts in the preparation code.
Review rendered prompts and answer format after changing it. A familiar task
name or equal split count does not prove identical questions or reward semantics.

## Fixed allocation from separate source datasets

Replace the run's existing `[data]` section with this example to allocate 90% of
fresh draws to a prepared Dolci Think dataset and 10% to Sudoku:

```toml
[data]
seed = 17
shuffle = true

[[data.sources]]
name = "dolci"
fraction = 0.90
prompt_data = "/weka/your-path/dolci/train.jsonl"
reward_config = "/weka/your-path/dolci/verifiers.json"

[[data.sources]]
name = "sudoku"
fraction = 0.10
tasks = [{ task = "synlogic-sudoku", train_count = 800, eval_count = 90 }]
```

Dolci stays intact as one source. The paths must refer to existing rendered data
and a trusted verifier registry, with the same named judge bindings and code
services the original run needs. This example does not provision those services.
The Sudoku source itself requires neither. Remove `eval_count` if using the
background evaluation path; shared-engine held-out evaluation and background
evaluation are mutually exclusive.

Each source accepts one existing preparation mode: named `tasks`, prepared
`prompt_data` with `reward_config`, or `rl_manifest`. Arbitrary raw Hugging Face
schemas still need an adapter to render prompts and bind verifier targets.
Source names are unique, fractions are positive and must sum to 1 (roundoff
tolerance 1e-9), and seed/shuffle are shared at the outer data level. Nested
mixtures and per-source seed/shuffle overrides are rejected. Changing the mix
requires a new run root, as with other immutable preparation changes.

Preparation writes separate `prepared/data/source-<name>.jsonl` files, a merged
`verifiers.json`, and `mixture.json` containing source fractions, row counts and
SHA-256 hashes. It checks prompt limits and train/held-out overlap across **all**
sources and rejects conflicting verifier definitions. It does not deduplicate
training sources against each other. The runtime retains a loaded dataset per
source; it does not create an interleaved training file or stream file cursors.

For every fresh prompt, the provider draws a source from the configured fractions
and walks that source's seeded shuffled order without replacement. When a source
is exhausted it independently reshuffles and cycles. Small sources can therefore
repeat many times during a run. Fractions describe allocation in expectation,
not exact batch quotas; reward filtering and backfill can change the accepted
training mix. Buffered retries do not make a new draw. A one-source mixture at
fraction 1 preserves the shuffle provider's sequence, with added attribution.

Mixtures reject the global frozen exclusion table (`[selection]` /
`core.selection_table`): replacing excluded prompts with draws from the whole
mixture would change the admitted source fractions. Per-source exclusion files
that restrict each source's eligible pool are not implemented yet.

The source RNG and every source cursor are checkpointed with pending groups;
resume validates the manifest identity. Metadata preserves original provenance
and adds `mixture_source` and `mixture_source_epoch`. When inference records are
enabled, those fields also appear on scored group records. This fixed provider
ignores outcome feedback. Recording remains best effort, as with shuffle: an
unavailable store or a failed record/attribution build is logged and counted,
without stopping sampling or training. ReMind retains its strict feedback checks
because its selections depend on outcomes. Combining mixtures with ReMind is not implemented.
Existing configurations without `data.sources` keep the default shuffle path.

### SynLogic Sudoku

`task = "synlogic-sudoku"` selects the 890 Sudoku rows of
[`MiniMaxAI/SynLogic`](https://huggingface.co/datasets/MiniMaxAI/SynLogic),
configuration `hard`, split `train`, pinned to
`bb4297b82b9c28ef39249f48386cefd5e856618a`. The
[pinned dataset card](https://huggingface.co/datasets/MiniMaxAI/SynLogic/blob/bb4297b82b9c28ef39249f48386cefd5e856618a/README.md)
declares MIT. `datasets.load_dataset` downloads through the normal Hugging Face
cache (`HF_DATASETS_CACHE` / `HF_HOME`, or the user's default cache); run-specific
rendered artifacts live under `output.root/prepared/data`. The initial download
is the full hard training split, roughly 53 MB, before filtering to Sudoku.

This is an explicit rendering adaptation: the original instructions are replaced
with English Sudoku rules, unchanged givens, and a request for a JSON 9×9 integer
grid inside one `<answer>...</answer>` block. No reference solution enters the
prompt or verifier target. Preparation checks every source reference against the
rules and givens, preserves the original row, UUID, difficulty and revision, and
records `prompt_rendering = "synlogic-sudoku-english-v1"`. The 800/90 example
partitions these 890 training rows deterministically; it does **not** use or
claim the upstream validation split.

`SudokuVerifier` gives 1 for a complete grid satisfying all givens, rows, columns,
and 3×3 boxes, and 0 otherwise. It accepts any valid completion, not only the
published reference, and never executes generated code. Missing/multiple answer
blocks, invalid JSON, booleans/floats, partial grids and rule violations score 0.
It is an independent rule checker, not a port of SynLogic's upstream scorer.

## Prompt provider interface

MILES still requests prompt groups through `get_samples(n)`. Open Instruct's
`DashboardDrainingRolloutDataSource` delegates to `ProviderRolloutDataSource`,
which takes buffered retries first and asks `provider.next_samples(n)` for the
remaining fresh prompts. The adapter makes K independent copies of each prompt
and assigns the existing group/response IDs. Generation and reward computation
keep their existing inputs.

The default `ShuffledPromptProvider` walks the same loaded dataset with MILES'
seeded shuffle and cursor. No configuration or preparation changes are needed.
It also handles requests crossing multiple complete epochs. A provider implements
`next_samples`, `feedback`, `state_dict`, and `load_state_dict`, and declares
`requires_feedback`; the provider itself
does not require an epoch or source-mixture abstraction. Alternate implementations
can be supplied through the data source's Python `provider=` argument. The built-in
choices are `core.prompt_provider = "shuffle"` (default), `"remind"`, and `"mixture"`.
Structured `data.sources` selects `"mixture"` automatically.

Provider state is saved atomically alongside the existing pending-group ledger.
The default provider accepts legacy shuffle checkpoints and preserves their
cursor fields. Provider-aware checkpoints also check dataset size, seed, and
shuffle mode; prepared artifact identity remains the preparation layer's job.
Retries and resumed pending groups do not request a new prompt from the provider.

`feedback(records)` receives inference records under the same lock as selection
and checkpoints. The shuffle provider ignores feedback. With ReMind enabled, the
recorder delivers every scored group before its best-effort disk queue, including
all-1 and all-0 groups dropped by the unchanged reward filter. A lost disk row does
not lose feedback; an unavailable recorder fails the run rather than silently
turning off adaptation.

### ReMind review provider

The second implementation illustrates outcome-dependent selection without
changing dataset preparation, generation, rewards, or training. It wraps the
shuffled stream with a FIFO review queue, following the prompt-review mechanism in
[ReMind](https://arxiv.org/html/2606.03087v1#S3.SS3). Enable it in a fully async Core
run with inference records:

```toml
[core]
prompt_provider = "remind"
remind_review_fraction = 0.02
remind_enqueue_probability = 0.25

[records]
enabled = true
root = "/weka/your-workspace/inference-records"
```

| Event | Provider action |
|---|---|
| Fresh group is verified all-1 | Enqueue its dataset row with probability 0.25, unless already queued or under review. |
| Review slot | Pop the oldest queued row and generate fresh responses. An empty queue falls back to the shuffled stream. |
| Review is verified all-1 | Remove the row from review tracking. |
| Review regresses, including all-0 | Return the row to the queue tail. |
| Review outcome is unknown or invalid | Keep it eligible for another review; do not infer mastery or regression. |

This example uses a **continuous allocation budget**, one review slot per 50 new
prompt admissions at the default fraction. It is independent of how requests are
chunked into `get_samples(n)` calls. It does **not** reproduce the paper's default
10%-every-fifth-update bursts, or guarantee 2% of accepted training groups, tokens,
or wall-clock time. Async backfill and policy lag still apply. Unused review slots
are not accumulated. Ordinary prompts retain shuffled traversal; review deliberately
repeats a prompt before the next epoch.

Only complete groups with verified binary rewards establish mastery. Graded or
unknown rewards cannot add prompts to the queue. Intentional truncation-to-zero
counts as a failure when that reward policy is enabled. Aborted groups retain their
selection identity for retries. A group acknowledged without a recorded outcome
releases its pending selection and returns an unfinished review to the queue. Reviewed rows and their targets come from the same
loaded dataset; no response trajectories are replayed.

The checkpoint contains the base cursor, review queue, pending selections, random
state, and allocation counter, alongside pristine pending rollout groups. Duplicate
feedback is ignored after the first terminal observation. Feedback takes effect on
subsequent selections in completion order; identical seeds do not remove async
arrival-order differences. Records include selection IDs and a review flag for
measuring the realized allocation. Frozen exclusion tables are currently rejected
with ReMind to avoid silently consuming queued reviews without outcomes. This is a
CPU-tested interface example, not a GPU-validated reproduction of ReMind's results.

<!-- Preserve the former section anchor for existing links. -->
<a id="large-training-mixture"></a>

## Three-node training mixture

`three-node.toml` targets the full `train` split of
[`allenai/Dolci-Think-RL-7B`](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B),
pinned to revision `0fb6466d31ef3a9dd16985ef635e6429e05a6491`.
This is the dataset selected by the checked-in
[Olmo 3 7B Think RL recipe](../../scripts/train/olmo3/7b_think_rl.sh).
The [dataset card](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B/blob/0fb6466d31ef3a9dd16985ef635e6429e05a6491/README.md)
reports these source-group counts:

| Source group | Rows |
| --- | ---: |
| Math | 30,180 |
| Instruction following | 29,813 |
| Code | 21,385 |
| General | 20,636 |
| Total | 102,014 |

Prepare every source row once, preserving the published proportions, source
identities, verifier names and targets. Do not substitute a balanced four-domain
basket, an RLZero mixture or a smoke-test subset. Render and tokenize prompts with
the selected policy checkpoint; the source's saved token IDs belong to its original
tokenizer. Preserve the full question for judged examples in `metadata.judge_query`.
The trusted verifier registry must support math, code/code_stdio and instruction
following; general-quality and general-quality_ref use the template's named bindings.
Code verification requires an externally provisioned service.

The `dolci-think-rl-7b` paths in the template are **prepared-artifact placeholders**,
not an automatic Hugging Face download or proof of provenance. Before launch,
record the source ID/revision and verify row counts, source-group counts and retained
identities against the pinned source. Record any deliberate exclusions as a changed
mixture; do not silently filter rows to satisfy the policy's prompt-length limit.
Use the existing prepared-JSONL or immutable-manifest input contract. Background
benchmark evaluation is configured separately and does not change this training mix.

## Online group filtering

MILES/Core training defaults to online reward-group filtering:

```toml
[training]
filter_zero_std_groups = true
```

The low-level equivalent is `core.filter_zero_std_groups`. By default the compiler resolves
this to `miles.dynamic_sampling_filter_path =
"miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"`;
`plan` records the effective native path. The native filter keeps a complete prompt
group only when the standard deviation of its selected reward scores exceeds
`1e-8`. All-0, all-1 and other constant-reward groups are discarded before trainer
packing/scoring. Mixed groups retain every response, including zero-reward ones.
This requires more than one training response per prompt. Evaluation is unfiltered
and may still use one response per prompt.

With `custom_reward_post_process_path="open_instruct.miles.rewards.truncation.exclude_truncated"`,
the compiler instead selects the built-in finished-response filter. It ignores
truncated responses when checking reward variation and drops groups with fewer
than two finished responses. A reward earned by unfinished text cannot keep an
otherwise constant-reward group in the training batch.

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
mechanics example therefore explicitly sets this switch to false.
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
artifacts, not a replacement for per-sample auditing. See [operations](operations.md).

## Reward-service failure policies

MILES inherits the existing Open Instruct GRPO verifiers' default of assigning
zero reward when judge or code-service requests fail after retries. This keeps
training running through transient failures, but a fallback zero is an ungraded
response. In a group with successfully graded responses, it can affect relative
advantages as though the response were incorrect. Failures correlated with
response content or length can therefore bias learning. Use the strict policies
below when failed grading should stop the run.

Service-error and timeout metrics cover the consumed collection, after group
filtering. Dropped groups are absent from those fractions, so the metrics can
understate failures. With zero-variance filtering enabled, a total outage can
prevent a collection from filling and produce no new collection metrics at all.
Inspect individual failure warnings as well; a low or missing error fraction
does not establish service health.

### Code-service failure policy

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

### General judge failures

Judge transport failures and invalid or incomplete replies default to zero
reward after retries, with a warning and per-sample `judge_error` diagnostics.
Set `OI_MILES_JUDGE_FAILURE_POLICY="raise"` in `[launch.env]` to stop on those
failures. Detected context overflow and configuration errors raise regardless
of that fallback policy. The three-node example explicitly selects `"zero"` for judge
failures; both node and three-node select it for code-service failures. Change those
settings to opt into strict handling. Managed judge health-check failures separately stop
training; the fallback applies to grading requests.

### Symbolic math timeouts

Symbolic math runs in a bounded subprocess pool. An individual request exceeding
45 seconds kills and replaces that worker and defaults to reward zero. The sample
retains `verifier_diagnostics` with `kind=math`, `status=timeout` and elapsed time;
`rollout/math_verifier/timeouts` and `timeout_fraction` count these separately from
ordinary incorrect answers. Set `OI_MILES_MATH_TIMEOUT_POLICY=raise` for strict
handling. Configuration errors, unexpected worker failures and cancellation
still propagate. A timeout-zero is an ungraded sample, not proof of a wrong
mathematical answer; include the timeout rate when interpreting benchmark scores.

## Independent background evaluation

For best-effort external olmo-eval jobs and checkpoint-axis W&B curves, see
[background evaluation](background-evaluation.md). Selecting this mode replaces
shared-engine evaluation; failed or skipped jobs leave gaps without blocking training.
