# Four MILES starting points

| Config | Purpose |
|---|---|
| [dev.toml](dev.toml) | Tiny-model colocation mechanics |
| [small.toml](small.toml) | Disaggregated GSM8K mechanics; the recommended first run |
| [medium.toml](medium.toml) | Mixed math, instruction-following, code and general-task MoE training |
| [large.toml](large.toml) | Provisional multi-node trainer layout with background evaluation; not qualified |

The [generated recipe tables](../../../docs/miles/configuration.md#example-recipes)
show current GPU allocations, batch sizes and publication settings directly from
these TOMLs. Run `plan` on your copy to inspect the resolved settings and unused GPUs.

## Prepare and launch

Copy a starter before editing it:

```bash
mkdir -p runs
cp configs/miles/examples/small.toml runs/my-run.toml
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

`dev` and `small` require a compatible tiny checkpoint. They exercise preparation,
generation, training, shared-engine evaluation, native checkpoint saving and final
HF export. They are mechanics checks, not accuracy baselines. A completed run does
not test interrupted-run recovery. Filtering is disabled in these two examples
because a tiny model may produce only constant rewards; see
[online filtering](../../../docs/miles/data-and-evaluation.md#online-group-filtering)
before changing this for a learning run.

`medium` targets the latent KDA MoE on B300 hardware. Supply an HF policy,
immutable prepared training and held-out JSONL files, a verifier registry and a
prepared Qwen3-32B judge. The registry needs a code-execution endpoint and the named
general-quality verifiers; the template does not provision a code service.
See [datasets and verifiers](../../../docs/miles/data-and-evaluation.md) and
[managed judges](../../../docs/miles/managed-judges.md).

`large` additionally requires an accessible `launch.secrets.BEAKER_TOKEN` for
background evaluation. Evaluator jobs allocate GPUs beyond the training plan;
overlapping jobs can increase that total. Configure the intended evaluation tasks
and credentials: the starter benchmarks do not cover code or general-task quality.
With offline W&B, sync training and manually publish the retained evaluation results.
Snapshot cleanup is manual after all referencing jobs stop. See
[background evaluation](../../../docs/miles/background-evaluation.md) for the
protocol, outputs and limits.

For either learning template, check model memory and service capacity before
scaling. The [throughput guide](../../../docs/miles/throughput-profiles.md) explains
the model-specific memory budgets and measured limits; its historical results do
not qualify a new model, image or topology. Follow the
[completion checks](../../../docs/miles/operations.md) to verify training,
checkpoint and export outputs.
