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

This works with every verifier, managed judges included; only
`reward_final_answer_only` is restricted to text verifiers. It cannot be combined
with a `custom_reward_post_process_path` other than the default ungraded-response
exclusion described below.

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
`</think>`. A missing closing marker or whitespace-only final answer receives zero
without calling a verifier. This applies to text/code verifiers and named judges;
judge rubrics retain their normal answer-tag extraction within that final region.
The full response and its token IDs remain available for training and recording.

Verifiers can optionally implement
`async_call_final_answer(tokens, prediction, label, **kwargs)`. In final-answer
mode, the adapter uses this method when present; otherwise it calls `async_call`
with the extracted suffix, preserving existing math and code verifier behavior.
Both paths run after the missing/empty-final and truncation gates; `prediction`
is the extracted suffix and `tokens` is empty. Verifiers that validate the raw
thinking boundary need the optional method to grade an already extracted final
answer. Raw-response mode always uses `async_call`. Named judges use their
separate final-region dispatch.

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

The native adapter uses `IFEvalVerifier`, including its existing
`remove_thinking_section` normalization in both modes. That removes answer tags,
`<|assistant|>` and outer whitespace before grading IF constraints. R1 format
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

## Score-candidate collection

MILES already provides top-k score centering (SCT), nested candidate collection,
and the `score_centering` loss. These extensions add array-preserving transport
and SCMC (independent samples with replacement) through native `/generate`.
These are extraction controls; `rollout_top_k` still controls response sampling.
Use matching MILES and SGLang revisions from the runtime lock. The benchmark
image `01M737J78XJT3P2STBR9DT897B` contains those revisions and passed the paired
training checks below; changing TOML alone does not install the extensions in an
older image.

For SCT32 with the existing score-centering loss:

```toml
[miles]
loss_type = "score_centering"
rollout_top_logprobs_num = 32
rollout_score_mode = "topk"
rollout_score_transport = "array"
use_miles_router = true
rollout_top_k = -1
rollout_top_p = 1.0
score_centering_is = "mis"
score_centering_mis_low = 0.5
score_centering_mis_high = 5.0
```

Use 64 or 128 for SCT64/SCT128. `score_centering_is` also supports `none` and
`tis`; `score_centering_tis_clip` controls TIS. Async score-centering runs default
to its own TIS correction, with ordinary `use_tis=false`. Explicitly selecting
both correction paths is rejected. Set `SGLANG_RETURN_ORIGINAL_LOGPROB=0` on the
rollout servers, as required by the native score-centering contract.

For **SCMC training**, select the Monte Carlo candidate mode and native TIS correction:

```toml
[miles]
loss_type = "score_centering"
score_centering_is = "tis"
rollout_top_logprobs_num = 32
rollout_score_mode = "mc"
rollout_score_transport = "array"
use_miles_router = true
rollout_top_k = -1
rollout_top_p = 1.0
```

SCMC supports 32, 64 and 128 draws. Its loss uses detached importance weights
in the existing MILES REINFORCE score-centering surrogate, replacing the
truncated-head correction with an IID Monte Carlo estimate. It does not replace
the default PPO `policy_loss`; comparisons against that baseline measure the
complete loss-path change as well as probe handling.
The SCT estimator continues to reject MC arrays; the dispatcher selects the
separate MC estimator using the explicit mode. Caller sampling seeds are preserved; missing seeds are
assigned independently per request because the probe kernel requires a seed.

Packed `flat` and `array` transport require `use_miles_router=true`. The pinned
SGLang router (0.3.2) strips the extended sampling and packing fields when it
re-serializes requests; MILES forwards the raw request and response bytes.

Transport alternatives remain selectable:

| `rollout_score_transport` | SCT | SCMC | Server/client representation |
|---|---|---|---|
| `nested` (default) | Yes | No | Ordinary JSON lists, converted to NumPy by MILES |
| `flat` | Yes | No | Server lists flattened into little-endian base64 buffers |
| `array` | Yes | Yes | NumPy arrays retained through server IPC, then base64 buffers |

Packed modes collect **generated tokens only**, without streaming. They currently
require unfiltered sampling, non-speculative Python SGLang and the native
`/generate` path. OpenAI/session endpoints are rejected for packed modes; their
existing nested SCT path remains available. `array` requires the local SGLang
extension; `flat` uses its flat/base64 response API.

The decoder checks response shape `[generated_tokens, width]`, zero null prefix,
float32 log probabilities, int32 token IDs, exact byte lengths and score mode.
It preserves normalized behavior-policy log probabilities and repeated MC draws,
without rescoring, deduplicating or renormalizing. The selected-token rollout
log probabilities remain a separate field. Candidate arrays and their mode
survive multi-turn merging, observation padding, truncation, safetensors transport,
DP partitioning and Core sequence packing.

The historical field names `Sample.rollout_topk_token_ids` and
`Sample.rollout_topk_log_probs` carry both modes; `Sample.rollout_score_mode`
distinguishes them and travels with training batches. A missing mode on a legacy
sample means `topk`. SCMC payloads must explicitly identify themselves.

The overhead report keeps nested, flat and array SCT transport alongside the
SCMC count sweep. Run `python -m scripts.miles.score_overhead_report --help` for
the renderer. It compares each completed run against its own paired baseline,
reports HTTP packing/unpacking fractions, and separates model-forward spans
from extraction spans. Those server benchmarks do not measure the new MILES
parser or a training update.

### Trainer benchmark

The H066 baseline and SCMC32-array runs each completed 20 optimizer updates on
Jupiter, using four trainer GPUs (EP4) and one inference GPU. Each update used
eight generated responses of 256 tokens with balanced synthetic rewards. After
discarding four warmup updates, mean trainer-update time was 0.993 seconds for
native PPO/GRPO and 1.028 seconds for native SCMC32 with TIS: a 3.53% increase
(35 milliseconds). Median times were 0.906 and 0.930 seconds. Peak allocated
memory was effectively unchanged at approximately 61 GiB per trainer GPU.

Runs:

1. [Native PPO/GRPO baseline](https://beaker.org/ex/01M737K99YD7WV52KRGGGEP4VY).
2. [SCMC32 with array transport](https://beaker.org/ex/01M737KBZ3DSCQJW4J68BDTNK0).

These are native host-wall trainer timings, excluding generation, rewards,
weight publication and startup, with no added CUDA synchronization. They compare
different loss paths in separate runs with independently evolving policies;
they do not isolate the MC correction or establish convergence or large-batch
performance. Use `scripts/miles/trainer_overhead_report.py` on the job logs to
reproduce the table and pass its JSON to `score_overhead_report.py` to include
trainer results alongside the inference/transport plots. NVFP4 is not qualified
by these BF16 runs.
