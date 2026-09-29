# MILES GRPO

Use MILES GRPO for reinforcement learning through Open Instruct, with OLMo-core
training and SGLang inference. This guide covers setup, the pinned runtime
and a first run. [Support and boundaries](feature-parity.md) remain specific to the
model and topology; retain each run's image, configuration and checkpoint identity.

## Run MILES GRPO

1. Check out the intended Open Instruct revision, then follow the
   [laptop/session setup](launching.md). Sibling development worktrees are unnecessary.
2. Copy [small.toml](../../configs/miles/examples/small.toml) to ignored
   `runs/my-grpo.toml`. Set model and output paths for a two-GPU disaggregated
   mechanics check. Use [medium.toml](../../configs/miles/examples/medium.toml)
   for mixed-workload training, after preparing its policy, data and judge assets.
3. Run `plan` and `validate`, then set `MILES_EXISTING_IMAGE` to the immutable
   image ID and invoke `python -m open_instruct.miles run /path/to/run.toml`.
4. Retain the launch receipt, submitted TOML and [completion artifacts](operations.md).
   Report a symptom with the experiment ID, image, model and configuration.

The submitting host needs Python 3.12 and the Beaker CLI. Training runs use a
separate image built from the selected application revision and
[`runtime.lock.json`](../../runtime/miles/runtime.lock.json). Building that image
currently requires access to private AI2 source repositories and the pinned
binary base; this integration is not a self-contained public training install.
See [image setup](launching.md#laptop-choose-or-build-an-image).

The [support matrix](feature-parity.md) and [measurements](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/index.md)
distinguish short lifecycle checks, learning-path evidence and longer experiments.
Use those recorded boundaries when choosing a model, topology or recipe.

## Router auxiliary losses in the examples

The four maintained examples set both router balancing and z-loss coefficients
to zero. This matches the current RL investigation recipe; low-level API defaults
remain unchanged. Re-enable either coefficient explicitly for an auxiliary-loss
experiment. Mixed-workload quality and throughput evidence remains in progress.

## Experimental expert-aware packing

Replay-informed expert-aware packing is **experimental and disabled by default**
in all maintained examples. Opt in with `trainer.expert_balanced_packing=true`
only on a supported topology. Correctness qualification has passed for the tested
configurations; planning overhead can offset training savings, and a learning
benefit has not been established. See [requirements, controls and measured
scope](sequence-packing.md#replay-informed-expert-aware-packing-experimental).

## Online filtering default

Learning runs enable `training.filter_zero_std_groups=true`: constant-reward prompt
groups are dropped and generation replenishes the accepted training batch. Offline
correctness preprocessing can still be applied. The tiny `dev`/`small` mechanics
examples explicitly disable filtering to permit all-zero-reward checks. See
[data and evaluation](data-and-evaluation.md#online-group-filtering) for metrics,
batch semantics and the opt-out.

This default and its refresh/engine-drain support require an application image
built from the updated source, including async retry-ledger retirement. The older
images preceding the filtering qualification do not include this change. The
[small filtering qualification](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/online-filtering-20260919.md) passed
with barrier and refresh publication. Local source changes are not overlaid onto
an existing application image.

## Excluding truncated responses

By default a response that reaches `max_response_length` is scored like any other:
the verifier reads the unfinished text, and an answer mentioned mid-reasoning can earn
reward. To leave truncated responses out of training (overlong filtering), set:

```toml
[miles]
custom_reward_post_process_path = "open_instruct.miles.rewards.truncation.exclude_truncated"
```

Truncated responses then get zero advantage and a zeroed loss mask, and each group's
baseline uses only its finished responses. Raw reward metrics are unchanged. With
response-averaged loss, excluded responses still count in the batch denominator.

## Runtime and qualification

Build an application image from the source revision you intend to run, using the
[locked runtime](architecture.md#runtime-sources-and-images). Reusing an image
runs its original code; local Python edits are not submitted with the TOML.

Historical [qualification records](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/index.md)
record their own model, image, topology and limits. They do not qualify subsequent
source changes or every maintained template. Use the [validation procedure](architecture.md#local-development)
for current changes, then exercise the selected model and topology.

## Publication modes

| Mode | Behavior and use |
|---|---|
| `core.publication_mode="refresh"` | Medium and large templates. Preserve generated tokens and their original behavior log probabilities across publication; rebuild serving state under the new weights and continue decoding. A response can span multiple policy versions. |
| `barrier` | Low-level default, used by dev and small mechanics examples. Does not continue mixed-policy responses across publication. |
| `engine_drain` | Independent engine drain: finish requests on their admitted version before swapping weights. This mode is distinct from mixed-policy refresh; see [engine drain](engine-drain.md). |

The [generated recipe tables](configuration.md#example-recipes) show each
starter's allocation, batch geometry, publication mode and lag limit. Read
[refresh requirements](async-pipeline.md#mixed-policy-refresh) and
[policy lag and TIS](async-pipeline.md#policy-lag-and-tis) before changing the
async recipe.

## Documentation and agent entry point

Agents discover this workflow through AGENTS.md; no
special prompt is needed. Follow the [documentation index](index.md) for model
support, configuration, data, parallelism and operations.
