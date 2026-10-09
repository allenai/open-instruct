# OpenR1 and BREAD continuation experiments

This is an opt-in experimental synchronous rollout path. It is not a reproduction
of the BREAD paper: it searches forward over paragraph sections and resamples the
selected prefix. The default MILES rollout and existing data recipes are unchanged.
CPU tests cover data isolation and search behavior; GPU qualification and checkpoint
selection are still required before interpreting training results.

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
The source's Math-Verify flags do not establish agreement with our runtime math
verifier; check that agreement before a learning run.

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
bread_max_prefixes = 9
bread_max_groups_per_batch = 1024
bread_concurrent_groups = 64

[miles]
fully_async = false
rollout_function_path = "open_instruct.miles.rollout.bread_runtime.BreadRolloutFn"
```

This fragment does not specify a checkpoint or authorize a launch. Configure at
least two samples per prompt. The implementation collects bounded waves of prompt
groups, with both groups and their samples generated concurrently. Each wave completes
before admitting results in input order, avoiding selection by completion speed.
The default concurrency is one group; size it to the inference fleet. Async publication,
partial rollout and custom generation/postprocessing hooks are rejected. Existing
synchronous publication keeps policy weights fixed through probes and fresh samples.
Deterministic serving is also rejected because its repeated per-sample seeds would
make the selected-prefix resampling identical to the probe.

## Search and training semantics

1. Generate and grade an unassisted group.
2. Only finished groups whose binary rewards are all zero enter prefix search.
3. Reveal cumulative reference sections from the beginning, restarting this search
   on each failure. Prefixes never include the entire reference. Stop before a
   section containing an explicit boxed answer or answer tag.
4. After the first probe with a success, generate a fresh group at that prefix.
   An all-success probe gets the same fresh-group check; it is not combined with
   failures from another prefix to manufacture reward variance.
5. Admit only finished mixed-reward groups. If the fresh group is constant, drop
   it and acquire another question. A finite collection budget prevents an
   unproductive dataset from hanging the run indefinitely.

The reference prefix is appended to the rendered assistant generation prompt.
It is prompt context, with no generated-response tokens or behavior log probabilities
attributed to the expert text. The student continues it; scoring and policy loss
retain that context. Ordinary on-policy continuation GRPO needs no Guide-style
cross-context importance correction. The response budget stays fixed across probes;
a prefix that would exceed context capacity is skipped instead of shortening the
student's budget. Incomplete groups never masquerade as zero-reward search targets.

## Progress evidence

`bread/groups.jsonl` contains append-only group events: sample identity, rollout
and policy version, initial rewards, each probe's section count/prefix character
count and hash, rewards and generated tokens, fresh-group outcomes, elapsed time,
and final admission.
Attempts replayed after an interrupted collection can produce multiple events;
these are attempt evidence, not a unique-example or committed-update ledger.

The rollout metrics report seen, zero-reward, rescued and dropped groups, probe
counts, and total generated tokens including discarded attempts. Compare branch
lengths only on a fixed problem panel or conditioned on problem identity; a changing
mixture of hard problems can otherwise make apparent curriculum progress misleading.

Use unassisted held-out accuracy as the primary learning measure. Compare against
ordinary GRPO from the same checkpoint and data at matched generated-token and
wall-time budgets. Rescue rate and earlier branching are mechanism diagnostics,
not substitutes for unassisted generalization. This first version records progress
but does not persist a per-problem anchor across visits.
