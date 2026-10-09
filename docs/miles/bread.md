# OpenR1 and BREAD continuation experiments

This is an opt-in experimental synchronous rollout path. It is not a reproduction
of the BREAD paper: it searches forward over paragraph sections and trains on the
first mixed-reward probe. The default MILES rollout and existing data recipes are unchanged.
CPU tests cover data isolation, search, admission and evaluation behavior. GPU
mechanics checks and unassisted evaluation are required to assess learning.

## Prepare OpenR1

Use an immutable commit from the Hugging Face dataset repository:

```bash
uv run python -m open_instruct.miles.datasets.openr1 \
  --revision <40-character-dataset-commit> \
  --config default --train-count 1024 --eval-count 128 \
  --output /path/to/openr1-prepared
```

The converter streams the train split and selects eligible unique problems by
seeded UUID hash, retaining only the requested count in memory. Training examples
remain fixed when only the held-out count changes. It retains final-answer questions with a complete Math-Verify-approved
R1 generation, using the shortest eligible attempt. It excludes proof targets,
obvious diagram references, and references without paragraph boundaries. These
are conservative filters, not a guarantee of correctness or absence of images.
The `openr1_math` registry entry uses Math-Verify 0.9.0, with raw answer targets
wrapped in LaTeX boxes and anchored prediction extraction following OpenR1's
reward recipe. Symbolic dependencies are pinned separately and run in bounded
verifier subprocesses. Existing `math` grading is unchanged. Audit reference
agreement and wrong-answer rejection before a learning run: source flags alone
do not guarantee runtime agreement. Math-Verify can interpret comma-separated
answers as sets, losing order and multiplicity; agreement is not a correctness
certificate for sequence questions.

`manifest.json` records revision, configuration, seed, filter counts and output
hashes. Each training row contains a question-only user message, a reward target,
source UUID, verifier metadata and `metadata.bread.sections`. Teacher text is
never rendered into the initial student prompt. Held-out rows omit reference
metadata entirely. Exact whitespace-normalized duplicate questions and repeated
UUIDs are removed before partitioning. Check external benchmark overlap separately.
The manifest uses the existing adoption schema, including its legacy reward-path
identifier; adoption resolves that identifier to the application's verifier registry.

OpenR1 and NuminaMath-1.5 declare Apache-2.0. Preserve applicable upstream license
and attribution notices and document filtering/conversion when distributing a
prepared dataset. No dataset files are checked into the source repository.

## Configure a run

Use an ordinary supported synchronous MILES configuration, with its policy,
resources and output path chosen separately. Select the prepared manifest:

```toml
[data]
rl_manifest = "/path/to/openr1-prepared/manifest.json"

[core]
publication_mode = "barrier"
filter_zero_std_groups = true
reward_zero_truncated = true
bread_max_prefixes = 9
bread_max_groups_per_batch = 1024
bread_concurrent_groups = 64

[miles]
fully_async = false
rollout_function_path = "open_instruct.miles.rollout.bread_runtime.BreadRolloutFn"
```

Configure at least two samples per prompt. Collection continuously backfills completed
prompt groups up to `bread_concurrent_groups`, rather than waiting for whole waves.
When enough eligible groups exist, stop new admissions and drain outstanding work.
Select eligible groups in input order, so an earlier slow BREAD search can replace
an already completed later group. Overflow generation is counted and discarded.
The default concurrency is one group; size it to the inference fleet.

This still uses synchronous weight publication, with fixed policy weights during
each collection. Fully asynchronous publication, partial rollout and custom
generation/postprocessing hooks are rejected. Deterministic serving is rejected
because repeated per-sample seeds undermine independent within-group sampling.
Timing observations support a later async admission policy; this version does not
implement a separate BREAD work queue or policy-lag scheduler.

## Search and training semantics

1. Generate and grade an unassisted group.
2. Groups whose binary rewards are all zero enter prefix search. Length-capped
   responses receive zero under `reward_zero_truncated=true` and remain eligible.
3. Reveal cumulative reference sections from the beginning, restarting this search
   on each failure. Prefixes never include the entire reference. Stop before a
   section containing an explicit boxed answer or answer tag.
4. Stop at the first probe containing a success. If its rewards are mixed, train
   directly on that probe group. If all responses succeed, discard the group; do
   not mix failures from another prefix into it or sample another fresh group.
5. Admit mixed-reward groups, including capped zero-reward responses. A finite
   collection budget prevents an unproductive dataset from hanging indefinitely.

`reward_zero_truncated=true` is required. Truncated responses remain in the loss;
this path rejects custom reward postprocessors, including `exclude_truncated`.
There is no soft length penalty in this experiment. Binary correctness with a hard
zero at the response cap is shared by ordinary groups and prefix continuations.

The reference prefix is appended to the rendered assistant generation prompt.
It is prompt context, with no generated-response tokens or behavior log probabilities
attributed to the expert text. The student continues it; scoring and policy loss
retain that context. Ordinary on-policy continuation GRPO needs no Guide-style
cross-context importance correction. The response budget stays fixed across probes;
a prefix that would exceed context capacity is skipped instead of shortening the
student's budget. Actual aborted or failed generation requests trigger recovery;
they are distinct from terminal responses that reached the generation limit.

## Progress evidence

`bread/groups.jsonl` contains append-only schema-2 events:

- `zero_group_ready`: identity, policy version, readiness time, per-response rewards
  and completion statuses, and failure category (`all_truncated`,
  `mixed_truncation`, or `finished_wrong`).
- `group_completed`: initial responses, each probe's section count, prefix hash and
  character length, generated tokens, rewards/statuses and timings; eligibility
  and search outcome.
- `batch_admission`: the final selection, plus arrival-to-batch and
  completion-to-batch delays. Use these events for per-batch aggregation, not all
  three event types together.

Interrupted attempts can replay identities. These are attempt evidence, not a
unique-example or committed-update ledger. Admitted rollout samples also retain
collection metadata and policy version for later timing analysis.

Metrics separate initial, probe, all-generated and admitted responses, reporting
truncation, accuracy, accuracy among finished responses, and token use. Search and
rescue counts are split by failure category. `bread/admitted_bread_fraction`
reports BREAD's share of training groups; overflow work remains in generation
cost. BREAD and ordinary group timing distributions include mean, median, p95 and
maximum collection duration and admitted waiting time. Per-response timing splits
client semaphore admission wait from the generation call; the latter includes
server queueing and service, not pure decoding. Reward-centered values are split
by finished-correct, finished-wrong and truncated responses; they describe the
advantage signal without claiming to decompose the actual gradient.

Use unassisted held-out accuracy as the primary learning measure. Compare against
ordinary GRPO from the same checkpoint, questions, filtering and truncation rules,
at matched generated-token budgets (including probes) and report wall time too.
Rescue rate and earlier branching are mechanism diagnostics, not substitutes for
unassisted generalization. Compare branching only on a fixed question panel or
conditioned on identity. This version does not persist anchors across visits or
screen questions using short-cap rollouts.

## Evaluation panels

Set `core.bread_eval_panel_size=128` and `training.eval_interval=5` for a fixed
prefix of each prepared held-out dataset every five updates. Initial and final
evaluation cover the full held-out dataset and report the same panel separately.
The driver also forces full evaluation on a graceful time-budget stop. Reserve
allocation time for this evaluation and checkpoint/export; a hard process kill
cannot guarantee final coverage. Shared-engine evaluation is required.

Optionally set `core.bread_eval_long_response_len=32768` for an additional initial
evaluation on that same panel at a larger cap. Engine context must fit the longer
response plus the question. Training response and packing caps remain separate.
These are matched questions and sampling parameters, not paired random trajectories.
`bread/eval-responses.jsonl` records per-response identity, status, reward and length;
metrics include accuracy among finished responses alongside native accuracy,
truncation and length. Interpret finished-only accuracy cautiously because easier
questions may be more likely to finish.
