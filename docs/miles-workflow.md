# MILES researcher workflow

A run file describes the checkpoint, data, trainer, serving engines, objective,
logging and allocation together. The section names follow `olmo-miles` examples;
the resolved training backend here is OLMo-core. Existing `[core]` / `[miles]`
TOMLs and the `RunConfig` Python interface remain available for low-level work.

Start from one of the [examples](../configs/miles/examples):

| File | Purpose |
| --- | --- |
| `grpo-basic.toml` | Two-update tiny-model colocated dev/test run using compact generated multiplication prompts; one shared GPU, Core resident throughout. |
| `grpo-disaggregated.toml` | Full-SFT B300 starter: two Core trainer GPUs and one dedicated TP1 SGLang engine, 100 updates. |
| `grpo-async-disaggregated.toml` | The same model and objective with bounded async, one-update staleness, buffer factor two and TIS. |
| `grpo-multitask.toml` | Four-update GSM8K + math example with heldout entries for each task. |

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
python -m open_instruct.miles plan configs/miles/examples/grpo-async-disaggregated.toml
python -m open_instruct.miles validate configs/miles/examples/grpo-async-disaggregated.toml
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
python -m open_instruct.miles run configs/miles/examples/grpo-async-disaggregated.toml
python -m open_instruct.miles status configs/miles/examples/grpo-async-disaggregated.toml
```

`run` delegates image building and submission to the repository's committed
`scripts/train/build_image_and_launch.sh --miles` workflow. First commit the
working tree and set `MILES_BASE_IMAGE` to the locally loaded base image recorded
in `runtime/miles/runtime.lock.json`. Alternatively, set `MILES_EXISTING_IMAGE`
to a compatible immutable Beaker image ID; aliases are rejected. The submitting
host needs the Beaker CLI and credentials, plus Docker when building.

The config launcher currently allocates one Beaker node: one shared GPU for
tiny development or two trainer GPUs plus one serving GPU for the full-SFT
examples. A topology needing multiple nodes is rejected before image building;
use the existing qualified campaign launcher for that layout.

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

The launch defaults in the examples are urgent priority,
`ai2/open-instruct-dev`, `ai2/holmes`, and a one-hour minimum runtime. The
100-update examples allow eight hours before timeout, including preparation,
evaluation and synchronous saves. CPU-only
jobs that need WEKA belong on `ai2/saturn`.

All commands accept repeatable TOML overrides, for example:

```bash
python -m open_instruct.miles plan configs/miles/examples/grpo-disaggregated.toml \
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

| Section | What belongs here |
| --- | --- |
| `[model]` | Input checkpoint and format. HF input loads directly into Core; native Core input needs its compatible HF template for export. No Megatron conversion is involved. |
| `[data]` and `[[data.tasks]]` | Named tasks with train/eval counts and prompt wrappers, prepared JSONL, or a supported `rl_manifest`. Seed and shuffle are recorded with preparation; the baseline `data.recipe` catalog is not ported. |
| `[output]` | Run root and optional final HF export. |
| `[launch]` | Workspace, cluster, priority, minimum runtime, GPU allocation, WEKA mounts, environment, secret names and timeout. Mount every WEKA volume containing model/template/data/cache/output paths. |
| `[training]` | Rollout count, initial/periodic evaluation and checkpoint cadence. |
| `[trainer]` | Trainer GPUs/nodes, expert parallelism, microbatch size, recomputation and attention backend. |
| `[inference]` | Colocation, engine topology, batch geometry, lengths and SGLang admission/cache/graph controls. |
| `[optimizer]` | LR, schedule, Adam, clipping, KL and entropy coefficients. |
| `[async]` | Scheduling, allowed weight age, buffer capacity and off-policy correction. |
| `[tracking]` | W&B mode/project/group/team. Examples retain offline files without requiring credentials. |
| `[runtime]`, `[compiler_cache]` | Core runtime selection and compiler-cache policy. |
| `[core]`, `[miles]` | Explicit adapter/native options without a structured alias. Unknown or unsupported controls fail validation. |

Data preparation uses open-instruct's task and reward machinery. A familiar
section name does not imply that every olmo-miles catalog entry or external
service is available. Named recipe selection is rejected; express the supported
task mix with `[[data.tasks]]` or adopt an immutable manifest. Reusing an immutable prepared manifest is the most direct
way to keep prompts, splits, rendering and reward configuration fixed between
runs. Compare the recorded question IDs when checking parity; matching counts
alone does not establish identical datasets.

## The restored baseline recipe

The training starters collect **8 prompts × 8 responses = 64 samples** and set
`global_batch_size=64`, giving one optimizer update per collection. The older
maintained profiles used 16 × 4. Both have 64 responses, but the number of
independent prompts and the reward-group distribution differ. Historical run
configs and measurement reports remain frozen.

The async example adds:

```toml
[async]
fully_async = true
max_weight_staleness = 1
async_data_buffer_capacity_factor = 2.0
async_unused_samples_handler = "retry"
rollout_submission_granularity = "group"
off_policy_correction = "tis"
```

This resolves to `fully_async=true`, `use_tis=true`,
`use_rollout_logprobs=false` and Core's `max_policy_lag=1`. Core scores the old
policy; TIS uses the serving-versus-scoring log-probability ratio to correct the
behavior-policy mismatch. This differs from the earlier async measurements,
which directly anchored on rollout log probabilities without TIS. It is a
recipe change, not a retrospective correction of those results.

The LR stays `1e-6`, GRPO standard-deviation normalization stays disabled, and
PPO lower/upper clipping stays `0.2` / `0.28`. Router auxiliary and z-loss
coefficients remain `0.01` and `1e-5`. Router replay remains an independent
choice; enabling async or TIS does not enable replay.

## Trainer mappings that need care

Most optimizer, sampling and SGLang names carry through directly. These trainer
settings have a narrower meaning here:

| olmo-miles concept | Core meaning or limitation |
| --- | --- |
| `trainer.gpus`, `trainer_num_nodes`, `expert_parallel_size` | `trainer.gpus` is GPUs per trainer node; total trainer GPUs are `gpus × trainer_num_nodes`. `expert_parallel_size` sets the expert process group. The qualified full-SFT layout is one node, EP2. |
| `activation_recompute` | Core activation checkpointing; it does not promise the same recomputation granularity as Megatron. |
| `micro_batch_size` | Currently one unpadded sample per trainer microbatch. Larger Megatron microbatches or packing are not direct substitutions. |
| `trainer_flash_attention_version=4` | Core `flash_4`; the exercised full-SFT runtime is B300. A matching number does not establish H100 qualification. |
| `trainer_backend="optimized"` | No equivalent preset: use explicit Core attention, row-specialization and other supported controls. |
| Trainer offload during colocation | Core stays resident. Full-model olmo-miles colocation memory fractions cannot be copied safely. |
| `async_save` | Unsupported; native Core saves are synchronous. Final HF export is a separate post-training operation. |
| Megatron conversion/layout settings | Do not apply to Core. The HF descriptor and native Core checkpoint have distinct roles. |

The examples retain the measured dedicated-engine limits together: client and
engine concurrency 64, decode graph size 64, recurrent cache 128, token pool
524288 and static memory fraction 0.6. Evaluation shares those limits. The
runtime's actual memory allocation and long-tail response lengths determine
throughput; these settings are not a guarantee of a particular evaluation time.

For the exact bounded workflow exercise, see
[`workflow-async-gsm8k.toml`](../configs/miles/qualification/workflow-async-gsm8k.toml):
four updates, the previously prepared full-SFT checkpoint, fresh named-task
GSM8K preparation, heldout evaluation, async TIS and 8 × 8 sampling. It disables
checkpoint saving and final export to isolate the configuration-to-training
path. This is a workflow check, not a new 100-update learning comparison.

The [completed exercise and independent sample audit](measurements/miles-researcher-workflow-20260911.md) passed: four updates, 256 training responses, and 12/16 held-out answers correct both initially and finally. The report records the exact scope and observed TIS clipping.
