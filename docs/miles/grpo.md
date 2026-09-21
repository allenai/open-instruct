# MILES GRPO

Use MILES GRPO for reinforcement learning through Open Instruct, with OLMo-core
training and SGLang inference. This guide covers setup, the verified runtime image
and a first run. [Support and boundaries](feature-parity.md) remain specific to the
model and topology; retain each run's image, configuration and checkpoint identity.

## Run MILES GRPO

1. Obtain the matching source bundle or supplied checkout, then follow the
   [laptop/session setup](launching.md). Sibling development worktrees are unnecessary.
2. Copy [small.toml](../../configs/miles/examples/small.toml) to ignored
   `runs/my-grpo.toml`. Set model and output paths for a two-GPU disaggregated
   mechanics check. Use [medium.toml](../../configs/miles/examples/medium.toml)
   for mixed-workload training, after preparing its policy, data and judge assets.
3. Run `plan` and `validate`, then set `MILES_EXISTING_IMAGE` to the immutable
   image ID and invoke `python -m open_instruct.miles run /path/to/run.toml`.
4. Retain the launch receipt, submitted TOML and [completion artifacts](operations.md).
   Report a symptom with the experiment ID, image, model and configuration.

For an internal source handoff, the standalone `open-instruct-miles-sharing.bundle`
can be cloned without publishing the integration history to the public GitHub
repository:

```bash
git clone -b robertb/miles-olmo-core /path/to/open-instruct-miles-sharing.bundle open-instruct
cd open-instruct
```

Then follow laptop/session setup above. The image contains the runtime dependencies;
the submitting laptop does not need Core, MILES or SGLang sibling checkouts.
The runtime image contains the merged training implementation. The submitted TOML
is carried into the job; local Python edits are not. Use the image and source
identity below when reporting or reproducing a run.

The [support matrix](feature-parity.md) and [measurements](measurements/index.md)
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
[small filtering qualification](measurements/online-filtering-20260919.md) passed
with barrier and refresh publication. Local source changes are not overlaid onto
an existing application image.

## Current runtime and qualification

For online filtering, use **`01M2XGZM2N1V4DQVMYHM52KBHZ`**, application source
`a8f8aca84`. It passed four-update small checks with barrier and refresh publication,
including all-zero/all-one rejection, full batch replenishment and checkpoint
cursor audits. See the [qualification record](measurements/online-filtering-20260919.md)
for the synthetic reward fixture and limits; full-policy qualification remains separate.

The September 18 router-controls runtime was
**`01M2V3EGGYA2YFMYCAS1X6JSD7`**
(`robertb/open-instruct-router-controls-dcc77a875`), application source `dcc77a875`
and Core `ab64c30699d5c3de327830be6f4b2e2277a0edd3`. Dev and small have
completed bounded lifecycle checks; full-policy and optional-objective
qualification is in progress. Read the [current qualification record](measurements/router-controls-20260918.md)
for scope, results and limitations. Do not treat this image as a completed
qualification of every template.

The preceding runtime was **`01M2F1RKZFZVJYAS0XQGEC3SEJ`**
(`robertb/open-instruct-miles-fast-2c477efd5`), built from application source
`2c477efd5`. It contains the merged mixed-policy refresh, queue instrumentation,
packing and scoring optimizations; no source overlay is needed. Its Docker ID is
`sha256:60aa54bfdeba71c49a863e393d7715c567d70cf293512f76f3a6939a6502759f`.

That earlier runtime pinned Core adapter `3d35ab326`, MILES backend `b18b71b18` and serving
adapter `02ccb5d`. The reusable runtime/application image layers retain the
qualified binary foundation. See [architecture](architecture.md) for builds.

| Evidence | What it establishes |
|---|---|
| Packaged CPU suite: 113 passed on build `917fb0a44`; final `2c477efd5` differs only in documentation | Integration imports and tested contracts in the rebuilt runtime |
| EP2 full-SFT fast configuration: 24 updates passed using the matching committed runtime overlay | Live mixed-policy training with packing, no recomputation, guarded scoring skip and two serving engines |
| EP8 GSM8K throughput screen: intended 16 updates passed | Distributed refresh/queue operation on that recorded configuration; not the full mixed-task recipe |
| EP8 mixed-task baseline: stopped after one update with a producer HTTP transport error | First-step scoring, gradients and publication passed; sustained mixed-task operation is not established by that attempt |

[Full-SFT qualification and integration provenance](measurements/full-sft-basket-20260914.md)
and [throughput records](measurements/throughput-20260913.md) retain exact source,
image/overlay identities and limits. Earlier dense/restart and 16K/32K checks
remain historical evidence for their recorded images; they were not rerun merely
by promoting this image. The maintained templates document their model and capacity assumptions; historical
qualification does not establish a new workload or topology in advance.

## Publication modes

| Mode | Behavior and use |
|---|---|
| `core.publication_mode="refresh"` | Medium and large templates. Preserve generated tokens and their original behavior log probabilities across publication; rebuild serving state under the new weights and continue decoding. A response can span multiple policy versions. |
| `barrier` | Low-level default, used by dev and small mechanics examples. Does not continue mixed-policy responses across publication. |
| `engine_drain` | Independent engine drain: finish requests on their admitted version before swapping weights. This mode is distinct from mixed-policy refresh; see [engine drain](engine-drain.md). |

The `small` starter uses one trainer and one TP1 engine with barrier publication,
four prompts × two responses and eight samples per update. The `medium` refresh
template uses **EP8 + seven TP1 engines + one judge**, 64 prompts × four responses,
256 samples per update, FIFO groups, lag at most two optimizer updates, and TIS.
In refresh mode, lag is measured from the oldest token's policy version. Original sampled-token
log probabilities remain the behavior denominator; trainer scoring supplies the
PPO anchor. Replay routes describe the final forward that rebuilt the route table,
not the historical expert choice for every previously sampled token.

Refresh requires resident disaggregated TP1 engines, one optimizer update per
collection, the managed single-turn producer, MILES router metadata and TIS.
Keep `use_rollout_logprobs=false`; it is independent of retaining original behavior
probabilities for TIS. Full decode CUDA graphs or disabled graphs are accepted;
prefill graphs, speculative decoding, serving prefill/decode disaggregation and
automatic engine fault tolerance are rejected. Sampling qualification requires
temperature/top-p 1 and top-k -1. See [configuration](configuration.md),
[async queues](async-pipeline.md) and [throughput settings](throughput-profiles.md).
A configured refresh mode does not prove a particular short run actually crossed
a policy boundary; inspect retained token spans and `refresh_scores` observations.

## Documentation and agent entry point

Agents discover this workflow from the repository README and AGENTS.md; no
special prompt is needed. Follow the [documentation index](index.md) for model
support, configuration, data, parallelism and operations. The dated
[consolidation record](measurements/sharing-20260913/README.md#consolidation-decisions-september-13)
tracks which branches supplied the current implementation.
