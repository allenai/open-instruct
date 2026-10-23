# Three MILES starting points

| Config | Purpose |
|---|---|
| [dev.toml](dev.toml) | One GPU, colocated trainer/inference; short GSM8K mechanics check |
| [node.toml](node.toml) | One eight-GPU node: 2 trainer GPUs (EP2) and 6 inference engines; no judge |
| [large.toml](large.toml) | Three-node learning: 4 trainer GPUs (EP4), 19 inference engines and 1 judge; 24 GPUs allocated, plus background evaluation |

The [generated recipe tables](../../../docs/miles/configuration.md#example-recipes)
show current GPU allocations, batch sizes and publication settings directly from
these TOMLs. Run `plan` on your copy to inspect the resolved settings and unused GPUs.

The examples keep inputs, topology, resource limits and recipe-specific overrides.
Launch settings, mounts and input/output paths remain explicit. The examples
rely on the structured loader's defaults for optimizer settings, seeds,
microbatching, recomputation, dynamic rows and async capacity. Single-node trainer
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
`large` separately pins its evaluator image.

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

`large` matches H1's three-node, 24-GPU footprint, replacing one of H1's 20 inference
GPUs with a judge. The nodes contain 4 training + 4 inference GPUs, 8 inference
GPUs, and 7 inference + 1 judge GPU, respectively; no GPUs are unused.
Its training input is the full published `allenai/Dolci-Think-RL-7B` mixture,
including general-quality tasks, prepared according to the
[source and preparation contract](../../../docs/miles/data-and-evaluation.md#large-training-mixture). It requires
a prepared Qwen3-32B judge with named verifier bindings. It also requires an
accessible `launch.secrets.BEAKER_TOKEN` for background evaluation.
Evaluator jobs allocate GPUs beyond the training plan;
overlapping jobs can increase that total. Its 25-entry post-training development
and text-safety panel runs on the initial/final checkpoints for the 200-update
template, and every 1,000 updates if extended. It preserves pinned task sampling
and full evaluation sizes. Configure the separate WildGuard/StrongREJECT judge
endpoints and SciCode reference-data path. Code tasks need the sandbox-enabled
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

The examples inherit disabled prefix caching (`inference.radix_cache = false`).
The pinned Olmo hybrid KDA kernel cannot produce the FP32 tracked-state snapshots
required by SGLang's `extra_buffer` prefix-cache path. Enabling that path fails
during rollout; selecting `linear_attn_prefill_backend = "triton"` does not change
the Olmo adapter's kernel override. Keep prefix caching disabled for this runtime;
the recurrent-state pool is still required for active requests. See
[hybrid model limits](../../../docs/miles/moe-model-support.md#current-limits).

The learning templates (`node`, `large`) separate 64 GPU-running requests per
engine from 128 end-to-end HTTP permits. The automatically sized producer supplies
two waves; the completed buffer holds one collection. HTTP permits include
response delivery/processing, so setting both limits to 64 can leave GPU slots
empty. Increasing only the producer cannot bypass the HTTP limit. Confirm warm
occupancy, latency and memory on your workload before increasing either further.

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
choices when changing models. `large` keeps its background evaluator configuration
explicit because evaluation changes independently of training topology.

`plan` reports copied configurations with no request headroom, forced rescoring,
weight qualification, periodic scoring audits, or other diagnostic overhead.
Review those warnings instead of inheriting settings from an old experiment.
