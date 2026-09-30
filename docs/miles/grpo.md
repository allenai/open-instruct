# MILES GRPO

Use MILES GRPO for reinforcement learning through Open Instruct, with OLMo-core
training and SGLang inference. This guide covers setup, the verified runtime image
and a first run. [Support and boundaries](feature-parity.md) remain specific to the
model and topology; retain each run's image, configuration and checkpoint identity.

## Run MILES GRPO

1. Obtain the matching source bundle or supplied checkout, then follow the
   [laptop/session setup](launching.md). Sibling development worktrees are unnecessary.
2. Copy [grpo-sharing.toml](../../configs/miles/examples/grpo-sharing.toml) to
   Git-ignored `runs/my-grpo.toml`. Set your name and fresh output root; keep the
   supplied checkpoint for the first four-B300-GPU check. W&B is offline, with
   no extra HF/W&B credentials required for these inputs.
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
by promoting this image. The first-run template shortens the measured small recipe
to two updates and adds evaluation; it does not establish a performance or
learning-quality result in advance.

## Publication modes

| Mode | Behavior and use |
|---|---|
| `core.publication_mode="refresh"` | Current full-model starter and small/large profiles. Preserve generated tokens and their original behavior log probabilities across publication; rebuild serving state under the new weights and continue decoding. A response can span multiple policy versions. |
| `barrier` | Low-level config default, used by the older `grpo-basic`, `grpo-disaggregated`, `grpo-async-disaggregated` and `grpo-multitask` examples and tiny dev profiles. Does not continue mixed-policy responses across publication. |
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

## Experimental Megatron verifier-GRPO qualification

The structured module also accepts `training.algorithm = "grpo"` with
`trainer.backend = "megatron"` for a **synchronous, teacher-free Qwen3-1.7B
mechanics qualification**. This adapter is experimental; CPU validation does
not establish GPU support, objective parity, throughput or learning quality.
Use an immutable local original HF checkpoint, pre-rendered `data.prompt_data`,
`data.eval_prompt_data` and a trusted `data.reward_config` registry. Do not add
a teacher or distillation section.

The initial contract owns one optimizer update and weight publication per
rollout, within-prompt centered verifier rewards, explicit PPO clipping and
Adam settings, and a response or token loss denominator. It scores the
pre-update policy in Megatron; no rollout-logprob substitution or reference KL
is enabled. The workflow retains verifier evidence, checks mixed-reward groups
and trainer advantages, saves native state, completes the HF export and loads
it in a fresh SGLang process. A mechanics run must pass those checks before
advancing to performance comparisons. Graphs and radix caching default off.
Async, colocation and offloading remain gated until separately qualified.

Keep output and asset directories separate and fresh, retain checkpoints
(`training.keep_checkpoints = 0`), and run `plan` and `validate` before the
committed-image `run` workflow. Use a native Megatron/SGLang image with the
pushed source overlay for this adapter, rather than the Core-only image.
This path does not establish Qwen3.5 support in OLMo-core.


A previously completed, independently load-audited **initial TP2/PP1 native
release** can be supplied as an inline `[model.native_checkpoint]` provenance
table. The table must retain the original HF source/revision/config hash,
architecture and argument fingerprint, TP/PP, native converter hash, release metadata hash, exact five-file
size/mtime inventory, save/load Beaker identities and UTC load audit. Its
`checkpoint_path` must remain separate from both fresh output directories.
For this initial qualification, the original HF source directory must retain the
first 12 characters of its pinned revision as a suffix.
The runtime verifies the bindings before native preflight, uses the old asset
read-only instead of conversion, and checks the file inventory again after the
training/export/reload workflow. Mismatches fail without automatic reconversion.
Retained `checkpoint-reuse.json` and `conversion-control.json` distinguish reuse
from conversion. This bounded provenance check uses metadata/config hashes and
shard size/mtime, not full shard checksums or independent Hub revision resolution.
It is neither training resume nor optimizer/cursor restoration; the same complete
verifier-update/checkpoint/export/reload qualification remains required.
