# Configuration reference

This is the maintained help for the MILES + OLMo-core wrapper. Start from a
[structured example](../../configs/miles/examples/README.md), then inspect `plan`.
The tables below are generated; edit reference-help.json or source definitions
and run `python -m scripts.miles.generate_docs`.

## Interface and precedence

`python -m open_instruct.miles {plan,validate,train,run,status} run.toml` accepts
repeatable `--set SECTION.KEY=TOML_VALUE` and `--debug`. Structured files require
schema_version=1 and model/data/output sections. Low-level files contain [core]
and [miles] and support only plan/validate/train. Structured validate is CPU-safe;
low-level validate invokes the installed native parser. Neither certifies GPU fit.

```bash
python -m open_instruct.miles plan configs/miles/examples/medium.toml \
  --set training.num_rollouts=20 \
  --set 'tracking.wandb_group="my-comparison"'
```

Repeated assignments to the same override key use the last value. Structured
aliases and explicit Core/native options targeting one resolved value must agree;
conflicting values fail. Native aliases cannot be supplied twice under different
spellings. There is no implicit environment expansion or configuration inheritance.
Booleans are unquoted true/false; strings need TOML quotes protected by shell quotes.
Lists and inline tables are encoded according to the pinned native parser.

Python callers may use RunSpec.load(path).compile() or
RunConfig(CoreConfig(...), miles_options). The latter assumes prepared inputs and
has no launch/data workflow. User-input failures raise InputError (a ValueError
subclass); the CLI prints a field-oriented error and exits 2. --debug includes the
traceback. Preparation checks that need data/model files run where those are mounted.

## Which section to edit

| Section | Purpose |
|---|---|
| model / conversion / output | Input identity, preparation and final artifacts |
| data | Tasks, immutable manifests or prepared rows and reward configuration |
| launch | Allocation, mounts, secrets and scheduling |
| training | Collection count, evaluation and saving cadence |
| trainer | Trainer node/GPU geometry, EP, microbatching and recomputation |
| inference | Engine topology, sampling geometry, lengths and SGLang admission |
| optimizer | Learning rate, Adam, clipping, entropy and KL |
| async | Buffering, lag and importance correction |
| tracking | W&B and retained reporting |
| runtime / compiler_cache | Adapter execution and compiler reuse |
| judges / rubrics / judging | Named service identity and verifier bindings |
| core / miles | Explicit backend controls and advanced native escape hatch |

## Defaults and interactions

There are three different sources of values: raw CoreConfig/parser defaults,
structured workflow defaults, and explicit example recipe choices. The generated
Core and native tables identify raw defaults; `plan` shows the effective structured
configuration and leaves unspecified native runtime choices unresolved.

Structured runs default to 8 prompts × 8 responses, global batch equal to the
collection, 100 collections, microbatch one, LR 1e-6 with constant schedule,
Adam betas 0.9/0.95 and epsilon 1e-8, weight decay zero, gradient clip 1, PPO clip
0.2/0.28, no GRPO standard-deviation normalization, KL/entropy coefficients zero.
Online constant-reward group filtering defaults to true through
`training.filter_zero_std_groups` (also `core.filter_zero_std_groups`). It uses the
native MILES dynamic filter and replenishes accepted groups to fill the training
batch. This is independent of GRPO standard-deviation normalization and offline
dataset preprocessing. Set it to false for a tiny mechanics check that cannot yet
produce mixed rewards. See [online filtering](data-and-evaluation.md#online-group-filtering).
Do not confuse PPO clipping with TIS clipping or reward normalization.

Context defaults to 6144 tokens during structured compilation. max_context_length
sets Core, SGLang and rollout context together; response must be smaller to leave
prompt space. max_total_tokens defaults to at least 524288 and at least engine
admission × context. This is a requested pool capacity, not measured GPU allocation.

Synchronous colocation is the implicit default; async requires explicit resident
disaggregation. Structured async enables TIS unless rollout log probabilities are
explicitly the anchor. TIS and use_rollout_logprobs cannot both be enabled. Async
derives the omitted producer budget from engine count and request capacity; see
[async queues and discard metrics](async-pipeline.md). It uses buffer factor 2,
retry, group submissions, and requires an explicit valid Core
lag allowance. The maintained async example selects lag one. KL > 0 enables the
reference pass with the prepared starting model unless ref_load is supplied.

Saving defaults to the final collection; save_checkpoints=false disables cadence
and conflicts with explicit save_interval. Held-out data enables initial and
periodic evaluation (interval 20, greedy, one response), using the training response
cap unless overridden. Tracking stays disabled unless explicitly enabled or a
non-disabled wandb_mode is selected. Example offline tracking is an explicit choice.

Core's raw packing default is false and row_specialization is static. The async
example enables packing and dynamic specialization. Replay is independent:
use_rollout_routing_replay requires use_miles_router; structured compilation supplies
the latter when omitted. The Megatron use_routing_replay option is not the Core switch.

For conditional constraints see [run controls](run-controls.md), [packing](sequence-packing.md),
[topology](topology.md), and [managed judges](managed-judges.md). For all native
flags, choices and source help see the [native appendix](native-options.md).

## Rollout capture

Capture defaults off. Set a group sampling rate in the run file:

```toml
[output]
root = "/path/to/run"
rollout_sample_rate = 0.1
```

This saves approximately 10% of prompt groups, with all sibling responses kept
or omitted together. Selection is deterministic from the rollout seed, update,
dataset and group identity, independent of training randomness. `0` skips capture
entirely; `1` or higher saves every group. Negative/non-finite rates are invalid.
The same selection controls `.pt`, dashboard Parquet and trajectory sidecars for
training and shared evaluation. Training batches and aggregate metrics remain
complete. Captures still serialize synchronously when enabled; a fractional rate
reduces the work but does not make it asynchronous. Use rate `1` for full-batch
replay diagnostics; a sampled capture is not a complete training batch.

This replaces the structured `save_debug_rollout_data` path setting. The wrapper
owns the path under `output.root/rollouts/`. Native MILES callers supply both
`--save-debug-rollout-data PATH` and `--rollout-sample-rate RATE`.
