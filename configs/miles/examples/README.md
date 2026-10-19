# MILES starting points

| Config | Purpose |
|---|---|
| [dev.toml](dev.toml) | Tiny-model colocation mechanics |
| [small.toml](small.toml) | Disaggregated GSM8K mechanics; the recommended first run |
| [medium.toml](medium.toml) | Mixed math, instruction-following, code and general-task MoE training |
| [qwen3_4b_deepscaler.toml](qwen3_4b_deepscaler.toml) | Qwen3-4B-Base Deepscaler benchmark, 16 prompts x 32 completions |
| [large.toml](large.toml) | Provisional multi-node trainer layout with background evaluation; adapt before use |

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
how to size memory budgets and engine admission for a model. Follow the
[completion checks](../../../docs/miles/operations.md) to verify training,
checkpoint and export outputs.

## Qwen3 Deepscaler

`qwen3_4b_deepscaler.toml` is the standard Deepscaler recipe: 16 prompts x 32
completions, a 512-sample global batch, four trainer GPUs and four inference
GPUs. It uses learning rate 1e-6, an 8,192-token response limit, a 10,240-token
context, and barrier publication. Math rewards use the MILES-native 1/0 scale,
and gradient clipping defaults to 1.0. These differ from the original Open
Instruct experiment's 10/0 rewards and gradient clipping at 5; the recipe does
not claim identical learning curves.

Prepare its training and AIME 2025 evaluation inputs:

```bash
uv run python scripts/data/miles/prepare_deepscaler.py /weka/oe-adapt-default/YOUR_USERNAME/open-instruct/data/deepscaler-10k-base-01 --tokenizer /weka/oe-adapt-default/YOUR_USERNAME/checkpoints/Qwen3-4B-Base
```

The default training dataset is
`mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles`. The helper renders
`qwen_instruct_user_boxed_math` prompts, excludes reference assistant answers,
and drops prompts over 2,048 tokens. It writes `train.jsonl`,
`eval-aime_2025.jsonl`, and `verifiers.json`. `--train-dataset` accepts another
dataset with the same `messages` and `ground_truth` schema.

Replace `YOUR_USERNAME`, review the workspace, budget, mounts and secrets,
and choose a fresh output root. This recipe enables online W&B, so configure
its API-key secret or switch `tracking.wandb_mode` to `"offline"`. Copy the
recipe to `runs/`, then follow the plan/validate/launch workflow above.
