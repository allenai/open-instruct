# MILES researcher workflow

A run file describes the checkpoint, data, trainer, serving engines, objective,
logging and allocation together. The section names follow `olmo-miles` examples;
the resolved training backend here is OLMo-core. Existing `[core]` / `[miles]`
TOMLs and the `RunConfig` Python interface remain available for low-level work.

Choose a maintained [starting point](../../configs/miles/examples/README.md),
then copy it into ignored `runs/` before customizing it. The
[generated recipe tables](configuration.md#example-recipes) describe the current
allocations and batch geometry. Historical measurements are linked from the
feature guides and do not override the maintained TOMLs.

Replace `YOUR_USERNAME` and the checkpoint path before launching. A model source
must include the compatible architecture, tokenizer and chat template. Keep the
source read-only and give each run a new output directory. The colocated example
requires a tiny model that fits alongside its optimizer and engine; it does not
establish full-SFT colocation support.

HF inputs must contain `config.json`, safetensors weights and a usable tokenizer
with a chat template. Native Core conversion exports checkpoint metadata and
weights using the Core converter with forward validation disabled. It is not
a new numerical or routing-parity qualification; architecture-specific
conversion and serving checks remain separate. HF staging references source
weights, so retain the original checkpoint for the life of the run.

## One configuration, inspection through launch

```bash
python -m open_instruct.miles plan configs/miles/examples/medium.toml
python -m open_instruct.miles validate configs/miles/examples/medium.toml
```

`plan` resolves the structured sections into the existing Core configuration and
native MILES arguments without allocating GPUs. Inspect model paths, data
sources, placement, batch geometry and objective before submission. `validate`
checks the structured configuration on a CPU host; native MILES validation also
runs before training in the pinned runtime. Neither proves memory fit or learning
quality.

Use `train` inside an allocation. `run` launches the config-driven Beaker
workflow and `status` inspects the submitted run:

```bash
python -m open_instruct.miles run runs/my-run.toml
python -m open_instruct.miles status runs/my-run.toml
```

See the complete [laptop and Beaker-session launch guide](launching.md).

`run` delegates image building and submission to the repository's committed
`scripts/train/build_image_and_launch.sh --miles` workflow. First commit the
working tree and set `MILES_BASE_IMAGE` to the locally loaded base image recorded
in `runtime/miles/runtime.lock.json`. Alternatively, set `MILES_EXISTING_IMAGE`
to a compatible immutable Beaker image ID; aliases are rejected. The submitting
host needs the Beaker CLI and credentials, plus Docker when building.

The config launcher supports single-node runs and replicated disaggregated
allocations with independent trainer, rollout and named-judge GPU counts.
See [multi-node placement and managed judges](managed-judges.md) for the
ownership rules, limits and tiny qualification configuration. `plan` reports
per-node assignments and unused GPUs; multi-node runs may keep `launch.auto_resume=true`: a restart into the same output root resumes from the newest native checkpoint and continues the rollout cursor without repeating or skipping an update, qualified in [multi-node resume](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/multinode-resume-20260922.md). Forced multi-node preemption, and restarts of runs carrying a managed judge, remain unqualified.

`status` reports the latest Beaker attempt, all experiment details and whether
the current config matches the submitted config. Receipts live under
`~/.cache/open-instruct/miles/launches` by default; set `MILES_LAUNCH_RECEIPTS`
to use another directory. It does not read live training metrics from WEKA.

Preparation and resolved settings are retained under the run root. `workflow.json` records `preparing`, `training`,
`complete` or `failed`; logs and small JSON reports are also collected into
Beaker results. Checkpoints and rollout tensors remain on WEKA. Reusing an
output root requires an unchanged run specification; completed runs cannot be overwritten. With `auto_resume=true`,
a retry adopts verified preparation and loads the latest completed Core
checkpoint when one exists. Without a completed checkpoint, training begins
from the initial model again. A killed process may leave the last recorded
workflow state; consult the Beaker attempt status when checking liveness.

Review the copied run's scheduling settings and rendered Beaker spec before
launching. See the [scheduling contract](launching.md#distributed-scheduling-contract)
for GPU placement, protected runtime and the separate CPU-job rules.

All commands accept repeatable TOML overrides, for example:

```bash
python -m open_instruct.miles plan configs/miles/examples/small.toml \
  --set training.num_rollouts=20 \
  --set 'tracking.wandb_group="gsm8k-parity-check"'
```

Strings require TOML quotes inside shell quotes. There is no implicit environment
variable substitution. If structured names and native escape-hatch names
address the same resolved option, different values are rejected rather than silently choosing a winner.

## Input errors and debugging

`plan` checks field names, scalar types, common numeric ranges and batch/GPU
geometry on CPU. The same checks run before `run` builds an image. Unknown
fields suggest a nearby spelling where possible; structured aliases retain
context such as `optimizer.learning_rate` when reporting a native-option error.
Booleans must be unquoted `true`/`false`, counts must be integers, and numeric
controls must be finite. Zero-temperature evaluation and zero auxiliary-loss
coefficients remain valid.

Expected input failures print an actionable error and exit with status 2.
Malformed JSONL reports the source file and line; invalid prepared rows report
the partition and row. These file checks happen during preparation, where the
mounted data and tokenizer are available. Paths are not required to exist on
the submitting host.

Use `--debug` on any command to include the traceback for an input error:

```bash
python -m open_instruct.miles validate run.toml --debug
```

Unexpected runtime failures still retain their tracebacks by default. Python
callers can catch `open_instruct.miles.errors.InputError`, a `ValueError`
subclass. These checks do not replace the pinned runtime's full validation or
prove that a model fits the selected GPUs.

## Configuration sections

Use [which section to edit](configuration.md#which-section-to-edit) for the
section map and [workflow fields](configuration.md#workflow-fields) for accepted
values. `plan` shows how structured aliases resolve into Core and native MILES options.

Data preparation uses open-instruct's task and reward machinery. A familiar
section name does not imply that every olmo-miles catalog entry or external
service is available. Named recipe selection is rejected; express the supported
task mix with `[[data.tasks]]` or adopt an immutable manifest. Reusing an immutable prepared manifest is the most direct
way to keep prompts, splits, rendering and reward configuration fixed between
runs. Compare the recorded question IDs when checking parity; matching counts
alone does not establish identical datasets.

## Collection and policy settings

A collection contains `rollout_batch_size × n_samples_per_prompt` responses;
`global_batch_size` determines responses per optimizer update. Their ratio is
the number of updates per collection. See the
[generated recipe tables](configuration.md#example-recipes) for each starter and
[policy lag and TIS](async-pipeline.md#policy-lag-and-tis) when changing that ratio.
Router replay, auxiliary losses and reference KL are independent settings.

## Trainer mappings that need care

Most optimizer, sampling and SGLang names carry through directly. These trainer
settings have a narrower meaning here:

| olmo-miles concept | Core meaning or limitation |
| --- | --- |
| `trainer.gpus`, `trainer_num_nodes`, `expert_parallel_size` | `trainer.gpus` is GPUs per trainer node; total trainer GPUs are `gpus × trainer_num_nodes`. `expert_parallel_size` sets the expert process group. Consult the topology guide for supported placement and qualification limits. |
| `activation_recompute` | Core activation checkpointing; it does not promise the same recomputation granularity as Megatron. |
| `micro_batch_size` | Core microbatch size is one. Optional document-isolated packing combines samples within an optimizer partition; see [packing](sequence-packing.md). |
| `trainer_flash_attention_version=4` | Core `flash_4`; the exercised full-SFT runtime is B300. A matching number does not establish H100 qualification. |
| `trainer_backend="optimized"` | No equivalent preset: use explicit Core attention, row-specialization and other supported controls. |
| Trainer offload during colocation | Core stays resident. Full-model olmo-miles colocation memory fractions cannot be copied safely. |
| `async_save` | Unsupported; native Core saves are synchronous. Final HF export is a separate post-training operation. |
| Megatron conversion/layout settings | Do not apply to Core. The HF descriptor and native Core checkpoint have distinct roles. |

Size serving concurrency, graph capture and memory pools together using the
[admission guidance](throughput-profiles.md#size-engine-admission-from-memory).
The [archived workflow exercise](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/researcher-workflow-20260911.md)
retains its original recipe, image and sample audit.
