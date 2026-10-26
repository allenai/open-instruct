# Three MILES starting points

| Config | Purpose |
|---|---|
| [dev.toml](dev.toml) | One GPU, colocated trainer/inference; short GSM8K mechanics check |
| [node.toml](node.toml) | One eight-GPU node: 2 trainer GPUs (EP2) and 6 inference engines; no judge |
| [three-node.toml](three-node.toml) | Three-node learning: 4 trainer GPUs (EP4), 19 inference engines and 1 judge; 24 GPUs allocated, plus background evaluation |

The [generated recipe tables](../../../docs/miles/configuration.md#example-recipes)
show current GPU allocations, batch sizes and publication settings directly from
these TOMLs. Run `plan` on your copy to inspect the resolved settings and unused GPUs.

The examples keep inputs, topology, resource limits and recipe-specific overrides.
Launch settings, mounts and input/output paths remain explicit. The examples
rely on the structured loader's defaults for optimizer settings, seeds,
microbatching, recomputation and dynamic rows. Single-node trainer
counts, TP1 inference, one-GPU judges and rubric defaults (2,048 output tokens,
temperature 1.0) are omitted where they match the run. Non-default topology and
qualification settings remain explicit. See the
[complete configuration reference](https://allenai.github.io/open-instruct/miles/configuration/)
for options, defaults and interactions, including the linked native-option tables.
Use `uv run mkdocs serve` to read the reference for your checkout.

## Prepare and launch

Copy a starter before editing it:

```bash
mkdir -p runs
cp configs/miles/examples/dev.toml runs/my-run.toml
# Replace model/output paths and YOUR_USERNAME; select an appropriate tiny model.
python -m open_instruct.miles plan runs/my-run.toml
python -m open_instruct.miles validate runs/my-run.toml
# Load the pinned binary base into Docker as described in the launch guide.
# Build an application image from this clean, committed checkout.
unset MILES_EXISTING_IMAGE
export MILES_BASE_IMAGE=LOCAL_LOADED_BASE_IMAGE
python -m open_instruct.miles run runs/my-run.toml
```

Do not launch a template unchanged: model, data, output and judge paths are
placeholders. Give each run a fresh output directory and keep personal variants
in ignored `runs/`.

The examples select high priority and Holmes for their B300 profiles. Review
cluster access, mounts, GPU allocation, minimum runtime and timeout before
submission. W&B defaults to offline; set online mode and an API-key secret when
needed. See the [launch guide](../../../docs/miles/launching.md).

Trainer images are built separately from the TOMLs. Follow the
[image build instructions](../../../docs/miles/launching.md#laptop-choose-or-build-an-image)
to build the intended application revision and runtime lock. `MILES_EXISTING_IMAGE`
can reuse a compatible immutable image; local changes are not overlaid onto it.
`three-node` separately pins its evaluator image.

## Inputs and expected outputs

`dev` retains its compatible tiny-checkpoint placeholder and short GSM8K workload.
It exercises preparation,
generation, training, shared-engine evaluation, native checkpoint saving and final
HF export. It is a mechanics check, not an accuracy baseline. A completed run does
not test interrupted-run recovery. Filtering is disabled in this example
because a tiny model may produce only constant rewards; see
[online filtering](../../../docs/miles/data-and-evaluation.md#online-group-filtering)
before changing this for a learning run.

`node` adapts the previous two-node `medium` learning recipe to one B300 node,
using two EP2 trainer ranks and six TP1 inference engines, with no judge.
Its prepared workload contains math, instruction-following and code tasks only;
exclude general-quality and other judge-dependent verifiers from both training
and held-out files. The paths use a separate `math-ifeval-code` preparation.
Its training batch, context and shared-engine evaluation cadence are unchanged.
This topology passes CPU planning checks; sustained throughput and memory fit
still need GPU validation. Supply an HF policy, immutable prepared training and
held-out JSONL files, and a verifier registry with a code-execution endpoint;
the template does not provision a code service.
See [datasets and verifiers](../../../docs/miles/data-and-evaluation.md) and
[managed judges](../../../docs/miles/managed-judges.md).

The former two-GPU `small` example is retired; use `dev` for the mechanics check
and `node` for disaggregated learning. A single-GPU run with the full policy needs
its combined trainer/inference memory footprint checked before replacing the
`dev` checkpoint; single-GPU inference alone does not establish that fit.

`three-node` matches H1's three-node, 24-GPU footprint, replacing one of H1's 20 inference
GPUs with a judge. The nodes contain 4 training + 4 inference GPUs, 8 inference
GPUs, and 7 inference + 1 judge GPU, respectively; no GPUs are unused.
Its training input is the full published `allenai/Dolci-Think-RL-7B` mixture,
including general-quality tasks, prepared according to the
[source and preparation contract](../../../docs/miles/data-and-evaluation.md#large-training-mixture). It requires
a prepared Qwen3-32B judge with named verifier bindings. It also requires an
accessible `launch.secrets.BEAKER_TOKEN` for background evaluation.
Evaluator jobs allocate GPUs beyond the training plan;
overlapping jobs can increase that total. Its 25-entry post-training development
and text-safety panel runs on the initial/final checkpoints only. With `interval = 0`,
periodic evaluation remains disabled even when the run is extended. It preserves pinned task sampling
and full evaluation sizes. olmo-eval launches WildGuard/StrongREJECT judges on an extra GPU
per safety group. Configure the SciCode reference-data path. Code tasks need the sandbox-enabled
evaluator image. The full panel still requires GPU/sandbox qualification and
measured resource bounds; the starter three-hour evaluator timeout is not a
measured full-panel budget.
With offline W&B, sync training and manually publish the retained evaluation results.
Snapshot cleanup is manual after all referencing jobs stop. See
[background evaluation](../../../docs/miles/background-evaluation.md) for the
protocol, outputs and limits.

For either learning template, check model memory and service capacity before
scaling. The [throughput guide](../../../docs/miles/throughput-profiles.md) explains
how to size memory budgets and engine admission for a model. Follow the
[completion checks](../../../docs/miles/operations.md) to verify training,
checkpoint and export outputs.

## Capacity and qualification settings

The learning examples pin `inference.profile = "olmo35-b300-v1"` on Holmes.
The preset sets **512 running requests, 512 HTTP permits and decode graphs through
batch 512 per engine**, 17,825,792 KV tokens, 3,072 recurrent slots, memory fraction
0.7 and prefill chunks of 8,192. It enables radix caching with `extra_buffer`.
Cached hybrid serving requires FP32 tracked-state snapshots in the pinned
[runtime](../../../runtime/miles/runtime.lock.json); rebuild the image when its
pins change. See [model limits](../../../docs/miles/moe-model-support.md#current-limits).

To move an example to Jupiter, change both the cluster and profile to
`olmo35-h100-v1`. This preset also uses admission/HTTP/graph batch 512 and prefill
chunks of 8,192, but disables caching and uses 6,684,672 KV tokens, 512 recurrent
slots and memory fraction 0.9. Both presets require the studied BF16 Olmo 3.5
geometry, dedicated TP1 engines and 34,816 context / 32,768 response caps.
These settings are starting points, not universal memory-fit or throughput guarantees.

Named presets apply whole. Any explicit preset-owned key conflicts, even an equal
value; errors show the offending keys, an editable TOML block, and two exits:
remove the keys to keep the preset, or select `generic` and manage the settings.
Known hardware/topology mismatches fail during planning. Model geometry is checked
when actual checkpoint metadata is readable and must pass after preparation before
training. Preset settings remain identical before and after preparation.

Omitted or `generic` profiles preserve generic defaults and explicit settings.
The default recurrent pool is 128 slots. A separately measured **cache-off B300
alternative uses 1,024 recurrent slots**; configure that alternative explicitly
in generic mode. A cached-320 H100 alternative uses 4,456,448 KV tokens and 1,920
recurrent slots and likewise requires explicit generic settings. Neither alternative
is selected automatically. `dev` retains its explicit small colocated configuration.

The planner offers read-only tips when a preset matches or explicit pools resemble
a preset for different hardware. Tips appear in a stderr box and
`plan.runtime.inference_profile.tips`; routine warnings remain plain lines.
There is no `auto` selector. Topology, metrics, producer budget and trainer controls
remain separate. To experiment, follow
[departing from a preset](../../../docs/miles/throughput-profiles.md#deliberately-departing-from-a-preset).
Generic mode does not resize pools, estimate memory fit or retain a hardware pin.

The unfinished-sample budget, when omitted, is
`max(collection samples, 2 × engines × min(HTTP permits, running admission))`,
rounded to complete prompt groups. This gives **6,144 for node and 19,456 for
three-node** with the named presets; generic 64-admission settings give 768 and
2,432. Explicit `async.async_max_concurrent_samples` wins. A smaller budget may
underfeed the engines, while a larger one may produce more stale work than the
trainer can use. Completed siblings can remain outside the unfinished budget.

Batch 256 and lag 6 are unchanged, so these budgets intentionally trigger policy-age
advisories. Batch-equivalents do not predict actual policy age or drain duration.
Measure trainer wait, useful throughput, oldest-token age and stale-work discards;
reduce the producer budget or fleet when production outruns useful consumption.
Per-engine settings do not qualify a whole fleet's routing, refresh latency or
learning behavior. See [qualification limits](../../../docs/miles/throughput-profiles.md#qualification-limits).

These templates inherit the disabled optional startup weight snapshot/reset audit
(`check_weight_update_equal = false`) and explicitly disable periodic score-reuse
audits (`scoring_check_interval = 0`). The first score check at startup/resume, required
scoring passes, TIS, and publication acknowledgments remain. `dev`
retains weight qualification. Re-enable audits when changing the publication or
scoring implementation. Learning templates save every 25 updates; choose the
cadence from measured save cost and acceptable recovery loss.

All three examples explicitly retain two native checkpoints and disable router
auxiliary losses; these differ from the defaults. The learning recipes also keep
their model-specific context, packing, cache and memory settings. Review these
choices when changing models. `three-node` keeps its background evaluator configuration
explicit because evaluation changes independently of training topology.

`plan` reports copied configurations with no request headroom, forced rescoring,
weight qualification, periodic scoring audits, or other diagnostic overhead.
Review those warnings instead of inheriting settings from an old experiment.
