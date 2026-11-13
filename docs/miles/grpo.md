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
2. Copy [dev.toml](../../configs/miles/examples/dev.toml) to ignored
   `runs/my-grpo.toml`. Set a run name, fresh output path and compatible tiny
   checkpoint for a single-GPU colocated mechanics check. Keep barrier publication
   and offline W&B for this first exercise unless requested otherwise.
   Use [node.toml](../../configs/miles/examples/node.toml)
   for single-node math, instruction-following and code training, after preparing
   its policy, data and code-verifier service. It allocates no judge GPU.
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

The maintained examples are `dev.toml`, `node.toml` and `three-node.toml`
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

The three maintained examples set both router balancing and z-loss coefficients
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
correctness preprocessing can still be applied. The tiny `dev` mechanics
example explicitly disables filtering to permit all-zero-reward checks. See
[data and evaluation](data-and-evaluation.md#online-group-filtering) for metrics,
batch semantics and the opt-out.

## Truncated responses

The demo configs are the baseline for configuration choices unless a run explicitly
states otherwise. GRPO uses **per-response loss weighting** and **zero reward for
length-truncated responses**. The Core reward default also scores truncations zero,
so omitting the field does not silently change the objective. The examples make
these choices explicit:

```toml
[core]
reward_zero_truncated = true

[miles]
calculate_per_token_loss = false
```

This works with every verifier, managed judges included. It cannot be combined
with a `custom_reward_post_process_path` other than the default ungraded-response
exclusion described below. Final-answer selection and explicit verifier exceptions
are described in [Grading inputs and changes](grading.md).

## Ungraded responses

A verifier or judge request can fail for reasons unrelated to the response: a
code-service outage, a judge transport error, an unparseable verdict. That
response has no grade. Scoring it zero would place a correct answer below its
siblings' baseline and push the policy away from it; dropping its prompt group
would waste every graded sibling. By default (`core.exclude_ungraded = true`) the
compiler installs
`custom_reward_post_process_path = "open_instruct.miles.rewards.exclusion.exclude_ungraded"`:
the ungraded response gets zero advantage and a zeroed loss mask, its group's
baseline uses only the graded responses, and the paired group filter ignores it
when checking reward variation (groups with fewer than two graded responses are
dropped). The failed response keeps `metadata.reward_omission` with the verifier
name and error, and W&B reports `rollout/verification/ungraded_samples` and
`ungraded_fraction` per collection. With response-averaged loss, excluded responses
still count in the batch denominator, so a high failure fraction reduces the
policy-gradient signal per step. Groups with no trainable responses are
always rejected, even with `filter_zero_std_groups=false`; that setting keeps
valid all-zero grades, not empty groups. The collector requests replacement groups.
A run where every group is rejected makes no optimizer progress.

Set `core.exclude_ungraded = false` to return to omitting the whole prompt group
(`rollout/verification/omitted_records`). The compiler also leaves the hook out
under `core.expert_balanced_packing`, which supports no reward post-process, and
under `miles.partial_rollout`, since the hook supports single-segment rollouts
only. Custom multi-segment generators must explicitly opt out as well. A custom
`miles.custom_convert_samples_to_train_data_path` bypasses reward postprocessing:
it retains group omission, and explicitly combining it with a managed exclusion
hook is rejected. Math verifier timeouts and code-service rejections (4xx, or a 500 that
persists through its retries) are not ungraded: they keep their zero reward, as
documented under [reward-service failure policies](data-and-evaluation.md#reward-service-failure-policies).

For an explicitly documented overlong-filtering experiment that excludes truncated
responses from training instead, override the baseline with:

```toml
[core]
reward_zero_truncated = false

[miles]
custom_reward_post_process_path = "open_instruct.miles.rewards.truncation.exclude_truncated"
```

Truncated responses then get zero advantage and a zeroed loss mask, and each group's
baseline uses only its finished, graded responses (this hook also excludes
ungraded responses). Raw reward metrics are unchanged. With response-averaged
loss, excluded responses still count in the batch denominator. When zero-variance
group filtering is enabled, it also considers only finished, graded responses:
groups with fewer than two of them or no variation in their rewards are dropped
before occupying training batch slots.

## Final-answer-only rewards

Set `core.reward_final_answer_only=true` to grade only text after the last literal
`</think>` by default. A missing closing marker, whitespace-only final answer or
reopened thinking receives each response-required verifier's floor without calling
it. Set `core.reward_reasoning_mode="none"` for an explicitly non-thinking run.
This applies to text/code verifiers and named judges; named judges receive the
selected region intact, including answer tags and surrounding prose.
The full response and its token IDs remain available for training and recording.

The shared `ResponseParts` object separates thinking and the delivered answer or
action. Trusted factory/registry declarations identify answer, process, format or
composite objectives. Answer objectives use `response`; explicit process/format
objectives may select `thinking` or `full` and separately permit a missing
delivered response. ToolRL declares full context for boundary format and the
presented response for action correctness; token-length graders retain full
trajectory tokens. See the [configuration, interface and
per-path behavior register](grading.md) before changing a run's grading policy.

Verifiers can optionally implement
`async_call_final_answer(tokens, prediction, label, **kwargs)`. In final-answer
mode with a response view, the adapter uses this method when present; otherwise
it calls `async_call` with the selected string. Migrated consumers receive
`selected_input=True`, bypass their thinking-boundary logic and retain their
task-specific answer parsing. All calls run after the component's eligibility
and truncation gates. Response/thinking views receive empty token lists; full
views retain actual trajectory tokens. A structured consumer can use
`async_call_parts`; composites additionally receive separately selected component
strings. Ordinary flag-off calls retain the historical `async_call` path,
including any native thinking removal. Named judges use their sample-aware path.

The built-in `R1FormatVerifier` implements this interface: after the outer gate
selects the final section, it checks for optional leading whitespace followed by
`<answer>...</answer>`, without requiring another closing-think marker. Its
raw-response check is unchanged. This remains a format reward, not an answer
correctness or content check; it retains the existing allowance for empty answer
tags and trailing text. In final-only mode, only the last closing marker's suffix
is checked, so answer tags in an earlier section cannot earn format reward.

For native IF grading, use the built-in adapter. A complete reward registry is:

```json
{
  "ifeval": {
    "factory": "open_instruct.miles.rewards.task_verifiers.ManifestIFVerifier"
  }
}
```

The native adapter uses `IFEvalVerifier`, including its existing task
normalization in both modes. Selected-input mode bypasses its thinking split
while retaining removal of answer tags, `<|assistant|>` and outer whitespace
before grading IF constraints. R1 format
reward checks the tags themselves; IF reward grades the normalized answer.

This option defaults to false and assumes the Olmo thinking format. It permits
multiple closing markers, so it is not an exactly-one-tag format validator.

## Runtime images

Build an application image from the source revision you intend to run, using the
[locked runtime](architecture.md#runtime-sources-and-images). Reusing an image
runs its original code; local Python edits are not submitted with the TOML.
Use the [validation procedure](architecture.md#local-development) for source
changes, then try the selected model and topology on a small run.

## Publication modes

| Mode | Behavior and use |
|---|---|
| `core.publication_mode="refresh"` | Node and three-node templates. Preserve generated tokens and their original behavior log probabilities across publication; rebuild serving state under the new weights and continue decoding. A response can span multiple policy versions. |
| `barrier` | Low-level default, used by dev mechanics example. Does not continue mixed-policy responses across publication. |
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
