# Four MILES starting points

| Config | Purpose | GPU allocation |
|---|---|---|
| [dev.toml](dev.toml) | Exercise colocation and basic plumbing with a tiny model | 1 shared GPU |
| [small.toml](small.toml) | Exercise disaggregated GSM8K at small scale | 1 trainer + 1 inference |
| [medium.toml](medium.toml) | Modest mixed math/IF/code/general training | 8 trainers + 7 inference + 1 judge; 16 B300 GPUs |
| [large.toml](large.toml) | Provisional production layout; not qualified | 16 trainers + 32 inference + 1 judge; currently reserves 56 GPUs |

Copy a starter before editing it:

```bash
mkdir -p runs
cp configs/miles/examples/small.toml runs/my-run.toml
# Replace model/output paths and YOUR_USERNAME; select an appropriate tiny model.
python -m open_instruct.miles plan runs/my-run.toml
python -m open_instruct.miles validate runs/my-run.toml
# Set MILES_EXISTING_IMAGE to the compatible immutable runtime image.
python -m open_instruct.miles run runs/my-run.toml
```

`dev` and `small` use short GSM8K responses with a tiny checkpoint to test
mechanics. They are not accuracy baselines. They exercise evaluation, saving and HF export as well as generation and
training. Resume needs a separate interrupted-run check; a completed run alone
does not test recovery. Both explicitly disable `training.filter_zero_std_groups`
because a tiny model may produce only constant rewards. Learning runs default to
online filtering; `medium` and `large` explicitly enable it. The filter removes
constant-reward groups and replenishes a full accepted training batch. Offline
correctness preprocessing remains compatible; see
[filtering semantics](../../../docs/miles/data-and-evaluation.md#online-group-filtering).

`medium` targets our latent KDA MoE on B300 hardware. Supply an HF policy,
prepared immutable mixed-task training/held-out JSONL files and verifier registry,
and a prepared Qwen3-32B judge. The registry must include the code execution
endpoint and the named general-quality verifiers. Those assets are prerequisites;
this template does not provision a code service. See [managed judges](../../../docs/miles/managed-judges.md)
and [datasets and verifiers](../../../docs/miles/data-and-evaluation.md).

The medium trainer uses sequence packing, dynamic-row SwiGLU, periodic checks of
scoring-pass skipping, router replay and flattened weight publication. At the
32K response budget, activation recomputation remains enabled: the measured
no-recompute win at 4K is not a memory qualification for 32K packs. SGLang uses
radix caching and decode graphs, with client admission and active-request limits
both set to 16 per engine. That concurrency is a starting estimate for long
responses. Watch KV occupancy, retractions, tokens/s/GPU, trainer wait and stale
sample drops before increasing it. These MoE cache budgets must not be copied
unchanged to dense Olmo 3 or a different architecture.

The completed queue holds one 256-response collection; the producer permits
1,024 outstanding responses, including those waiting for inference admission.
Mixed-policy refresh and TIS retain a lag limit of two updates. Evaluation runs
before training and every 50 updates; native checkpoints every five updates
protect progress against preemption. All four templates explicitly keep only
the newest committed checkpoint (`core.checkpoint_keep_last=1`); retention runs
after a successful commit. Compiler-cache restore is enabled, but
publication before preemption remains follow-up work.

`large` keeps the same workload and numerical settings as a planning baseline.
Trainer parallelism is two eight-GPU nodes, EP8 with DP2. Its topology,
throughput, recovery and memory use require qualification before production.
Thirty-two engines plus a one-GPU judge currently need a fifth serving/service
node, leaving seven allocated GPUs unused. `plan` reports this explicitly.

Router grouping, auxiliary averaging and balancing-count source are explicit
in all four examples and preserve the existing pack/token/token/dispatch defaults.
See [router objectives](../../../docs/miles/core.md#router-auxiliary-objectives)
for alternatives and their runtime requirements. A qualification of the defaults
does not by itself exercise the optional objectives.

W&B defaults to offline; configure an API-key secret and online mode when needed.
Do not run any template unchanged: model, data, output and judge paths are
placeholders. Put all experiment-specific variants in ignored `runs/`.
