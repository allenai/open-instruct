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

## Current runtime and qualification

Use **`01M2F1RKZFZVJYAS0XQGEC3SEJ`**
(`robertb/open-instruct-miles-fast-2c477efd5`), built from application source
`2c477efd5`. It contains the merged mixed-policy refresh, queue instrumentation,
packing and scoring optimizations; no source overlay is needed. Its Docker ID is
`sha256:60aa54bfdeba71c49a863e393d7715c567d70cf293512f76f3a6939a6502759f`.

The lock pins Core adapter `3d35ab326`, MILES backend `b18b71b18` and serving
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

The first-run starter uses **EP2 + two TP1 engines**, 32 prompts × 4 responses,
128 samples per update, FIFO groups, lag at most two optimizer updates, and TIS.
Lag is measured from the oldest token's policy version. Original sampled-token
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
