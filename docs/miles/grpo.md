# MILES GRPO

MILES GRPO is the recommended path for reinforcement learning with supported
Olmo MoE models through Open Instruct, with OLMo-core training and SGLang inference.
Existing [Core/vLLM and DeepSpeed/vLLM workflows](../algorithms/grpo.md) remain
available for dense-model training. This guide covers setup, the pinned runtime
and a first run. See [support and limits](feature-parity.md) for what is available
and [development defaults](development-defaults.md) for starting settings.

## Run MILES GRPO

1. Check [model and topology support](models-and-checkpoints.md), check out the
   intended Open Instruct revision, then follow the
   [laptop/session setup](launching.md). Sibling development worktrees are unnecessary.
2. Copy [small.toml](../../configs/miles/examples/small.toml) to ignored
   `runs/my-grpo.toml`. Set a run name, fresh output path and compatible tiny
   checkpoint for a two-GPU disaggregated mechanics check. Keep barrier publication
   and offline W&B for this first exercise unless requested otherwise.
   Use [medium.toml](../../configs/miles/examples/medium.toml)
   for mixed-workload training, after preparing its policy, data and judge assets.
3. Check Beaker resource access and supply credentials for the selected inputs
   and tracking mode. Run `plan` and `validate`, then inspect the rendered Beaker
   spec against the [distributed scheduling contract](launching.md#distributed-scheduling-contract);
   those commands alone do not check that contract. Set `MILES_EXISTING_IMAGE`
   to the immutable image ID and invoke
   `python -m open_instruct.miles run /path/to/run.toml` through the committed-image
   wrapper. Verify replica grouping and placement after submission.
4. Retain the launch receipt, submitted TOML and [completion artifacts](operations.md).
   Follow the run to completion and report the experiment link, image, model,
   configuration and validation outcomes.

The submitting host needs Python 3.12 and the Beaker CLI. Training runs use a
separate image built from the selected application revision and
[`runtime.lock.json`](../../runtime/miles/runtime.lock.json). Building that image
currently requires access to private AI2 source repositories and the pinned
binary base; this integration is not a self-contained public training install.
See [image setup](launching.md#laptop-choose-or-build-an-image).
The submission host does not need the local CUDA training stack. Example model
and data paths are placeholders that must be replaced for the selected run.

## Configuration conventions

The maintained examples are `dev.toml`, `small.toml`, `medium.toml` and `large.toml`
under `configs/miles/examples/`. Keep these templates unchanged for individual
runs. Copy one to Git-ignored `runs/` for experiments, sweeps and debugging.
Add a maintained example only for an agreed, distinct user-facing purpose;
otherwise update an existing tier. Tests must not depend on untracked run files.
Keep historical provenance in measurement reports and immutable run artifacts.

For disposable runs without checkpoint/resume testing or saved-weight needs, set
`training.save_checkpoints=false` and `output.export_hf=false`, and remove any
explicit `save_interval`. Rollout capture defaults off; set
`output.rollout_sample_rate` when whole-group captures are needed for analysis
(`1` keeps every group).

New asynchronous runs start with `async.max_weight_staleness=6`, compiled to
`core.max_policy_lag=6`. Set the latter explicitly in low-level `[core]`/`[miles]`
configurations. Preserve user overrides and historical reproduction settings;
synchronous mechanics runs retain zero lag. Six is a starting point, not an
optimum for every workload. See [policy lag and TIS](async-pipeline.md#policy-lag-and-tis)
and [development defaults](development-defaults.md).

## Router auxiliary losses in the examples

The four maintained examples set both router balancing and z-loss coefficients
to zero; low-level API defaults remain nonzero. Re-enable either coefficient
explicitly to train with auxiliary losses. See
[router auxiliary objectives](core.md#router-auxiliary-objectives) for the controls.

## Experimental expert-aware packing

Replay-informed expert-aware packing is **experimental and disabled by default**
in all maintained examples. Opt in with `trainer.expert_balanced_packing=true`
only on a supported topology. Planning overhead can offset training savings;
measure end-to-end update time before adopting it. See [requirements and
controls](sequence-packing.md#replay-informed-expert-aware-packing-experimental).

## Online filtering default

Learning runs enable `training.filter_zero_std_groups=true`: constant-reward prompt
groups are dropped and generation replenishes the accepted training batch. Offline
correctness preprocessing can still be applied. The tiny `dev`/`small` mechanics
examples explicitly disable filtering to permit all-zero-reward checks. See
[data and evaluation](data-and-evaluation.md#online-group-filtering) for metrics,
batch semantics and the opt-out.

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
When zero-variance group filtering is enabled, it also considers only finished
responses: groups with fewer than two finished responses or no variation in
their rewards are dropped before occupying training batch slots.

## Runtime images

Build an application image from the source revision you intend to run, using the
[locked runtime](architecture.md#runtime-sources-and-images). Reusing an image
runs its original code; local Python edits are not submitted with the TOML.
Use the [validation procedure](architecture.md#local-development) for source
changes, then try the selected model and topology on a small run.

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
