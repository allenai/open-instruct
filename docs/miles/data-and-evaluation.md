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

## Screening instruction-following training data

MILES preparation rejects known unsuitable targets assigned to the built-in
`IFEvalVerifier` or `ManifestIFVerifier`, including custom registry names for
those factories. The screen applies to every training source and rechecks
immutable prepared data on resume. It does not remove rows automatically or
change held-out datasets. A rejected historical preparation needs an explicitly
cleaned input and a new run root.

These training-data safeguards complement the checker corrections in
[fix-ifevalg-checkers (#2124)](https://github.com/allenai/open-instruct/pull/2124) and depend on its
lowercase-counting semantics: `N` caps repetitions of each lowercase word, so
positive limits, including two and three, are admitted.

The screen rejects malformed constraint bundles; missing, noninteger or
nonpositive `N` for lowercase and total-letter counting; invalid total-letter
relations; and total-letter `less than` limits with `N <= 3`. Such tiny total-letter
limits encourage answers with almost no letters. It also rejects missing or
invalid paragraph counts/indices and first words, including indices beyond the
declared paragraph count and first words containing whitespace or punctuation
that the checker truncates (`. , ? ! ' "`). These rules apply only to training admission and do
not change checkers, held-out data or scoring. They are not a complete quality or
relevance check; admitting a row does not certify it. Explicit response-language
constraints remain supported.

Audit a prepared JSONL file before changing the training mixture:

```bash
python scripts/data/filter_ifeval_constraints.py \
  --input /path/to/train.jsonl --report /path/to/if-audit.json
```

To create a cleaned copy, use fresh report/output paths:

```bash
python scripts/data/filter_ifeval_constraints.py \
  --input /path/to/train.jsonl --output /path/to/train-clean.jsonl \
  --report /path/to/if-filter-report.json
```

The tool supports MILES `metadata.verifiers` and legacy `dataset`/`ground_truth`
rows, preserves each retained row's bytes and order, and records hashes, reason
counts and dropped row identities. For custom IF registry names, supply
`--verifier-name NAME` for each trusted IF name, including `ifeval` if both occur.
Legacy verifier names are matched case-insensitively, as in their reward router;
MILES names are matched exactly. The tool trusts these names without reading a
reward registry: do not select names routed to `IFEvalVerifierOld`. Existing files
are never overwritten; report/output filesystems must support hard links.
Legacy Core/vLLM and DeepSpeed/vLLM data
preparation can use the same tool before tokenization; their launchers do not
apply this MILES admission screen automatically.

The composition checker (`count:counting_composition`) already accepts the
instructed `* * *` dividers and ignores punctuation-only word tokens; its rows
are retained by this screen. English-request filtering alone does not remove
these constraint problems, and neither this screen nor constraint scoring checks
semantic relevance. Record changed dataset hashes and mixture counts when
adopting cleaned inputs for a new experiment; keep historical run inputs intact.

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
The mixture loader requires pre-rendered text: runtime `apply_chat_template`
must be false and `multimodal_keys` must be unset for consistent source loading.

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

### Low-success history and length-triggered prompt replay

A group with zero or one successful response and several truncations may become
more productive as the policy changes its answer delivery or subject-matter
ability. The opt-in prompt-replay layer records that history and
can regenerate failed prompts after recent response length falls. It wraps the
ordinary shuffle or mixture provider; it reuses the original prompt and verifier
targets, never old completions. Existing configurations remain unchanged.

Add these options to an existing fully async run:

```toml
[core]
prompt_replay_enabled = true
prompt_replay_fraction = 0.10
prompt_replay_length_drop = 0.20
prompt_replay_window = 256
prompt_replay_max_successes = 1
prompt_replay_truncation = "any"
prompt_replay_answer_probe = true

[records]
enabled = true
root = "/weka/your-workspace/inference-records"
```

| Option | Meaning |
|---|---|
| `prompt_replay_fraction` | Fraction of proposed prompt draws reserved for eligible retries. Zero enables tracking only. An empty eligible pool falls back to fresh draws; unused slots do not accumulate. |
| `prompt_replay_length_drop` | Relative reduction since the failed attempt: 0.20 requires the recent mean to fall to at most 80% of its failure-time value. |
| `prompt_replay_window` | Number of valid fresh groups in each source's rolling window. Require this many additional fresh groups after a failure before retrying. |
| `prompt_replay_max_successes` | Maximum successful responses in a complete, valid binary-reward group. Default 1 includes 0/K and 1/K; set 0 for all-zero only. Graded nonbinary groups are tracked but excluded from replay eligibility. |
| `prompt_replay_truncation` | `any` requires at least one capped response; `half` requires at least half; `all` requires every response capped; `none` imposes no truncation requirement. Default `any`. |
| `prompt_replay_answer_probe` | Record a bounded literal-reference match for scalar `math` targets, including unfinished text. Unsupported targets are unknown. This is a diagnostic, **not verified correctness**, and never changes a reward. |

The length statistic is mean generated tokens per response across the recent
fresh groups **within the same source**, including intentional truncation zeroes.
The failure baseline is that rolling mean when feedback arrives, rather than the
failed prompt's own length. For example, a 20% threshold with a 10,000-token
failure baseline permits a retry once a full fresh window has elapsed and the
source mean is at most 8,000. Replay responses do not update this baseline. Sparse
sources therefore take longer to become eligible. Shortening is a heuristic for
when to retry, not evidence that the answer is now within reach.

Complete all-zero groups are counted separately as no truncations, some
truncations, or all truncations. A capped response deliberately assigned zero by
`reward_zero_truncated` is valid evidence under that reward policy. Verifier
errors, unknown outcomes, incomplete groups and aborted generations do not count
as task failures. Historical IF diagnostics with the successful `ifeval: scored`
status are recognized despite older recorder validity bookkeeping. Truncation
membership does not establish that truncation caused an otherwise correct answer
to fail.

Eligible prompts are visited in first-observation order, without concurrent
replays of the same prompt. A valid group outside the selected criteria retires the
candidate; later qualifying fresh observations can enroll it again. A one-success
group can already train through the normal zero-variance filter and also become a
future replay candidate. A qualifying replay waits for another
full fresh window and a further relative length drop. An invalid replay retains
the candidate but also resets its cooldown. The first scored generation counts;
staleness regeneration is not counted twice.

History, length windows, pending reviews and base-provider cursor are checkpointed
with the rollout ledger. Resume requires the same replay configuration and prompt
catalog. Enabling replay on an old run requires an explicit new continuation;
it cannot reconstruct unrecorded history from a legacy cursor. Frozen exclusions,
ReMind, synchronous rollouts and nonglobal datasets are currently rejected.
Like ReMind, this feature requires working recorder feedback and fails if the
recorder cannot construct outcomes; disk queue drops remain separately reported.

`rollout/prompt_replay/*` metrics report category counts, candidates, replay outcomes
and literal-match counts (including zero-group and truncated-response subsets).
Group records carry stable prompt identity, a replay flag, outcome classification,
and the baseline/current source lengths. Prompt identity hashes all rendered turns
and verifier targets, scoped by source. It is not semantic deduplication across
sources. Both online feedback order and async policy lag affect future selections.

Replay fractions refer to proposed groups, not accepted groups, tokens, or compute.
The base mixture controls fresh draws; replays can change the overall source mix.
Inspect source attribution and accepted-group/token costs when comparing runs.
Even a fraction of 1 falls back to fresh draws when no candidate is eligible: this
is not a finite, frozen-pool replay experiment.

For a compact historical inventory, stream explicitly chosen record attempts:

```bash
python -m open_instruct.miles records replay-pool \
  --manifest /records/run/manifest-ATTEMPT.json \
  --manifest /records/resume/manifest-RESUME.json \
  --max-policy-version 400 \
  --max-successes 1 --truncation any \
  --attempt-ranges attempt-ranges.json \
  --output zero-reward-history
```

The optional ranges file maps attempt IDs to inclusive policy-version bounds,
for example `{"ATTEMPT": [0, 100], "RESUME": [100, 400]}`. Select these from the
actual checkpoint ancestry to exclude abandoned recovery branches. The CPU reader
exports `summary.json`, the selected `replay-prompts.jsonl`, all ever-zero IDs in
`zero-prompts.jsonl`, and `low-success-prompts.jsonl` for comparing alternate
truncation thresholds locally. They contain per-source counts, stable IDs, success
and truncation histograms, and each prompt's latest valid outcome. The latter file
keeps both 0/K and 1/K observations, irrespective of the chosen pool threshold.
A prompt is included if it **ever** qualified; later outcomes remain visible so
you can choose a latest-outcome rule when preparing a historical experiment.
These runs can have K=4 or K=8: do not label a 1/4 observation as 1/8. The reader does not
copy response text or create a training dataset. Old records without answer probes
remain unknown; this command does not retrospectively verify their text. Missing
records and verifier errors are reported separately from valid zeroes.

A 0/K-to-mixed transition alone does not show improvement: ordinary resampling can
produce it at a frozen policy. Compare replay productivity and downstream evaluations
against matched continuation controls before adopting this experimental sampler.

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
45 seconds kills and replaces that worker and returns reward zero. The sample
retains `verifier_diagnostics` with `kind=math`, `status=timeout` and elapsed time;
`rollout/math_verifier/timeouts` and `timeout_fraction` count these separately from
ordinary incorrect answers. Validation timeouts always return zero with diagnostics;
there is no timeout policy setting. Configuration errors, unexpected worker
failures and cancellation still propagate. A timeout-zero is an ungraded sample, not proof of a wrong
mathematical answer; include the timeout rate when interpreting benchmark scores.

## Independent background evaluation

For best-effort external olmo-eval jobs and checkpoint-axis W&B curves, see
[background evaluation](background-evaluation.md). Selecting this mode replaces
shared-engine evaluation; failed or skipped jobs leave gaps without blocking training.

## Qualified local source catalogs

`python -m open_instruct.miles.datasets.source_catalog` resolves reviewed source
names to ordinary `[[data.sources]]` blocks. It checks the prepared prompt and
verifier-registry SHA-256 digests, rejects unqualified names, and requires the
requested fractions to sum to one. The catalog and reward registries are trusted
configuration, just like a run file; do not accept them from unreviewed datasets.

```bash
python -m open_instruct.miles.datasets.source_catalog /path/to/qualified/catalog.json \
  --source c094-deepmath-103k-math=1.0 --output /tmp/deepmath-data.toml
```

Use the resulting `[data]` section in a run file, or combine its named source
blocks with existing prepared sources and set the final fractions. Moving the
whole catalog directory preserves its relative bundle paths. This does not
change mixture sampling or introduce another run-file data mode.

A qualified **slice** is not a claim that every row or subtask of its parent
Hugging Face dataset is ready. Read the catalog's row counts, strata, exclusions,
and qualification receipts. The preparation catalog initializes `gpu_exercised`
to false; the later GPU audit writes `source-gpu-receipts.json` under the coverage
run directory without modifying that immutable catalog. Use those separate
receipts for completed GPU coverage. The local qualification tools in
`scripts/data/rl_readiness/` check positive, wrong, empty, and capped responses
through the real reward registry. CPU qualification exercises native data
preparation, mixture-provider source attribution, cursor restoration, and
mixed/all-zero group filtering. Their input
export must come from a reviewed adapter with immutable source provenance and
existing hygiene receipts.

The optional `deterministic_verifiers` factories cover explicit MCQA answer lines,
formatting-only checks, Countdown arithmetic, named knight/knave assignments,
structural numeric answers, exact text, and typed JSON answers. MCQA uses a
prepared `Answer: LETTER` contract; it must not silently replace a source's
original extraction rules. The JSON-schema factory grades schema conformance,
not factual content. ReasoningGym uses the pinned optional dependency listed in
`scripts/data/rl_readiness/requirements.txt`, with explicit family coverage.
No factory executes code supplied in a dataset or fetches external schema URLs.
The schema subset rejects `pattern` and `patternProperties` because the schema
library uses unbounded Python regex matching. This conservative recursive check
also rejects those names inside property maps or instance literals. Such schemas
must be deferred during preparation, not silently have constraints removed.

H169's explicit `mcqa_final_choice` target kind also accepts a bare choice letter,
an `Answer: LETTER` final answer or a boxed letter. Multiple explicit answers and
trailing contradictory text score zero. Preparation must select this contract
explicitly; existing `mcqa` targets retain their answer-line extraction.

Run the readiness tools as modules from the repository root, with their optional
dependencies installed in the pinned MILES environment:

```bash
python -m scripts.data.rl_readiness.qualify --export /path/to/export.jsonl --output /path/to/new-catalog --hf /path/to/model
python -m scripts.data.rl_readiness.local_coverage --catalog /path/to/new-catalog --fixture /path/to/toy-model --output /path/to/coverage
python -m scripts.data.rl_readiness.audit --catalog /path/to/new-catalog --run /path/to/coverage
```

Direct file-path invocation is not the supported entry point. To package these
tools, build the readiness image with the **repository root** as its context.
Set `GH_TOKEN` to a GitHub token with read access to the private code-execution
client repository; BuildKit mounts it only during dependency installation:

```bash
docker build -f scripts/data/rl_readiness/Dockerfile \
  --build-arg BASE_IMAGE=YOUR_MATCHING_MILES_IMAGE \
  --secret id=github_token,env=GH_TOKEN -t rl-readiness .
```

The image copies the readiness and RLVR preparation packages under
`/opt/core-rl/scripts/data/` and checks the readiness and source-preparation
entry points with `--help` during the build.

Formatting matching shares a one-second monotonic deadline across the response,
with each regex call additionally capped at 100 ms or the remaining time,
whichever is shorter. The deadline is checked between lines, markers, matches and
patterns, including the quoted-pair fast path. These are in-process checks, not
process isolation: linear string operations and target validation are not
forcibly interrupted. A timeout returns
fallback zero with `verifier_diagnostics.status="timeout"`, the original error,
and elapsed time. It is an ungraded response, not evidence of an incorrect answer.
There is no timeout policy setting. Invalid targets and unexpected failures
still raise. Failure capture reserves two slots per
worker/save directory: one for a raised registered-call exception and one for a
returned `status="timeout"` diagnostic from any verifier, including math and
formatting. Each records the full prompt, response, scored response and target
under `<save>/verifier_failures/`. A fallback timeout cannot consume the exception
slot, and absent error text is recorded with a descriptive placeholder.
Minimum-line rules return success as soon as enough lines match; irrelevant
trailing lines cannot invalidate that predicate. Qualification rejects timeouts
even though runtime validation timeouts return zero.

ReasoningGym families inheriting its default substring scorer use stripped exact
matching instead. At the pinned version these are `base_conversion`, `leg_counting`
and `simple_equations`; gold-containing wrong answers are qualification controls.
Task-specific scorers retain their semantics, and multiple explicit final answers
are rejected. A single boxed answer nested inside answer tags is one answer.
Diagnostics identify the corrected protocol as `exact_default_v2`;
earlier screen rewards using the default substring scorer need rescoring before
comparison with this protocol.

The catalog CLI emits seed 17 and shuffling enabled. The bounded qualification
pass uses shuffling disabled to check source order and cursor restoration; it is
not evidence for a particular shuffled training sequence.

The local GPU coverage tool uses one actual prompt per qualified named source,
four short completions, and a tiny model. It loads plain prompt data and verifies
that each row's existing sample/source labels survive rollout; it does not exercise
mixture-provider runtime attribution on GPU. That provider attribution is checked
separately during CPU qualification. It disables zero-variance filtering
because the random model generally earns zero; the separate controls test the
real filter. Router auxiliary losses can produce optimizer updates even with
zero task rewards. This is a training-machinery check, not evidence of learning,
endpoint transfer, or a recommendation to use four samples in a real run.


## Dataset integration

Use [source preparation](source-preparation.md) to convert
reviewed dataset exports into named training sources, preserving provenance and
reporting unsupported rows. The [reward verifier guide](dataset-verifiers.md)
describes their runtime contracts. CPU qualification checks the complete prepared
pool and reviewed reward controls before a named source enters a MILES mixture;
it does not require a separate inference study or establish downstream benefit.
## Named pools with training-exposure control

A fixed source mixture controls proposed draws; filtering and queue expiry can
make the data actually used for training quite different. Named pools instead
target uniform exposure in **trained prompt-groups**, and can replay low-success
truncated prompts within the selected pool. Dataset rows supply a top-level
`pool_id`; preparation preserves their source and verifier and maps the label to
a configured pool name. This is opt-in and leaves existing shuffle and fixed
mixture configurations unchanged.

Exposure counts groups, not draws or tokens: a prompt with eight responses counts
as one unit only after the optimizer update using it succeeds. If Math delivers a
trained group on 25% of draws and IF on 50%, equal exposure needs approximately
two Math draws per IF draw. The controller estimates that delivery rate, favors
pools behind their accumulated target, and subtracts the expected contribution
of outstanding work. Adjustments are bounded; token totals are reported separately.

```toml
[data.pools]
math = { label = "math" }
if = { label = "instruction_following" }
code = { label = "code" }
science = { label = "science_knowledge" }
logic = { label = "logic_reasoning" }
general = { label = "general" }

[[data.sources]]
name = "labeled_data"
prompt_data = "/data/labeled-trainer-rows.jsonl.gz"
reward_config = "/data/verifiers.json"
# Add additional labeled sources in the same way. Omit fraction in pool mode.

[async]
fully_async = true

[records]
enabled = true
root = "/run/inference-records"

[core]
prepared_batch_overlap = false
prompt_replay_enabled = true
prompt_replay_max_successes = 0
prompt_replay_truncation = "any"
prompt_replay_fraction = 0.1
prompt_replay_length_drop = 0.1
prompt_replay_window = 256
```

This is a fragment to apply to an otherwise valid MILES run, not a complete
launch configuration. Prepared inputs must already have the standard rendered
`input`, `label`, and verifier `metadata` fields. Dolci's prompt-inventory records
still require target resolution and rendering. Compressed JSONL is supported.
There must be exactly one optimizer update per collection. Sample count is
unchanged by the sampler; an eight-response recipe remains eight-response.

| Contract | Behavior |
|---|---|
| Membership | Exactly one configured label per training row. Missing, unknown, duplicate label definitions, and conflicting runtime annotations fail preparation. |
| Source fractions | Rejected in pool mode, rather than silently ignored. Fresh rows within a pool are uniformly shuffled across its source membership. |
| Finite fresh data | Each row is drawn once; there are no automatic epochs. Empty pools are skipped. |
| Exposure credit | One group per completed optimizer update that used it. Scored, queued, filtered, or stale groups earn no credit. |
| Reservations | Outstanding draws reserve estimated delivery, reducing over-correction during asynchronous generation. |
| Controller | Delivery estimate `(trained + 1)/(terminal + 2)`, floored at 0.05; target deficit corrected over a 64-group horizon. Proposal weights are bounded to `[0.25, 5]`, limiting draw ratios to 20:1. |
| Uniform target | Each trained group's unit target is divided over the pools active when it was drawn. Inactive pools accrue no new target from later draws. Exact equality is not promised after pools empty or for sources with effectively zero delivery. |
| Replay | Pool selection precedes replay. The replay fraction is a cap on that pool's proposed draws. Replay regenerates responses and uses the original verifier. |
| Replay trigger | Existing PR 2187 history, with fresh-response length windows scoped to source within pool. Invalid verifier results are separate from valid zeros. |
| Exhaustion | A pool without fresh rows or a budget-eligible replay is inactive. Once all pools and outstanding generation drain, an incomplete final collection is not trained; the last completed checkpoint is saved when configured. |
| Stale work | Pool mode defaults to `async_unused_samples_handler="drop"`; explicit retry is rejected. Drops release reservations. |
| Recovery | Manifest identity, cursors, RNG, targets, delivery counters, reservations, and replay histories are checkpointed with pending prompts. |

The first implementation supports fully asynchronous disaggregated serving,
without prepared-batch overlap. The application requires the companion MILES
runtime's `core_commit_exposure` callback and finite-pool termination support,
included in this branch's runtime source pin. Older images fail explicitly.
Build an image from this branch before launching this mode; the existing image
alone is insufficient. Full distributed GPU validation remains pending.

Inference records retain `pool_id`, `sampling_pool`, `mixture_source`, and
`pool_selection_id`. Per-update `pool-exposure-<attempt>.jsonl` receipts live in
`records.root` and contain trained/target/proposed/pending counts, generated and
trained response tokens, fresh/replay training counts, and replay diagnostics.
Unknown token counts are recorded separately. Receipts identify the resume point;
if an attempt fails before checkpointing, reconcile its tail against the restored
checkpoint instead of adding counts from different attempts. Captured response
tokens are not loss-mask tokens or GPU cost.

An empty pool does not force another epoch or a replay. When fresh data is gone,
replay must still satisfy its budget and shortening trigger; otherwise training
can finish before the requested update count. Different pool labels do not change
verifiers, reward semantics, truncation policy, or the zero-variance filter.
