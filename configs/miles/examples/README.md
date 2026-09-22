# Four MILES starting points

| Config | Purpose | GPU allocation |
|---|---|---|
| [dev.toml](dev.toml) | Exercise colocation and basic plumbing with a tiny model | 1 shared GPU |
| [small.toml](small.toml) | Exercise disaggregated GSM8K at small scale | 1 trainer + 1 inference |
| [medium.toml](medium.toml) | Modest mixed math/IF/code/general training | 8 trainers + 7 inference + 1 judge; 16 B300 GPUs |
| [large.toml](large.toml) | Provisional production layout with background evaluation; not qualified | 56 training/service GPUs, plus 1 independent evaluator GPU per task group |

Copy a starter before editing it:

```bash
mkdir -p runs
cp configs/miles/examples/small.toml runs/my-run.toml
# Replace model/output paths and YOUR_USERNAME; select an appropriate tiny model.
python -m open_instruct.miles plan runs/my-run.toml
python -m open_instruct.miles validate runs/my-run.toml
# Tested small barrier/refresh image; see the qualification scope below.
export MILES_EXISTING_IMAGE=01M2XGZM2N1V4DQVMYHM52KBHZ
python -m open_instruct.miles run runs/my-run.toml
```

The example TOMLs do not pin a trainer image; `MILES_EXISTING_IMAGE` selects it at launch.
`large` separately pins its evaluator image. Use a trainer image containing
background-evaluation support when running that example; the older small image
above is not its qualification.
The image above includes online filtering and passed the
[small barrier/refresh qualification](../../../docs/miles/measurements/online-filtering-20260919.md).
That check does not qualify every model or topology in these templates.

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
radix caching and decode graphs, with client admission, active-request limits
and decode graph capture all set to 64 per engine. The KV pool holds 64 full-context
requests, and about 165 GiB of each B300 stays free. See
[admission sizing](../../docs/miles/throughput-profiles.md#size-engine-admission-from-memory)
for the arithmetic and the metrics that confirm it. The live throughput gain over
the previous 16 has not yet been measured at 32K. These MoE cache budgets must not
be copied unchanged to dense Olmo 3 or a different architecture.

The completed queue holds one 256-response collection. The producer budget is
derived from the fleet: two waves of the 448 serving slots, or 896 outstanding
responses, including those waiting for inference admission.
Mixed-policy refresh and TIS retain a lag limit of two updates. Evaluation runs
before training and every 50 updates; native checkpoints every five updates
protect progress against preemption. All four templates explicitly keep only
the two newest committed checkpoints (`core.checkpoint_keep_last=2`); retention runs
after a successful commit. Compiler-cache restore is enabled, but
publication before preemption remains follow-up work.

`large` keeps the same training workload and numerical settings as a planning baseline.
Trainer parallelism is two eight-GPU nodes, EP8 with DP2. Its topology,
throughput, recovery and memory use require qualification before production.
Thirty-two engines plus a one-GPU judge currently need a fifth serving/service
node, leaving seven allocated GPUs unused. `plan` reports this explicitly.

Unlike medium, large uses **background olmo-eval**, with separate Beaker jobs at
updates 0/50/100/150/200. It does not borrow or drain rollout engines for evaluation;
frozen snapshot export is still synchronous trainer work. The starter tasks are
128 GSM8K and 128 IFBench (`ifeval_ood`) examples, greedy with an independent 8K
output cap. Matching generation settings share one one-GPU job per milestone.
These benchmarks do not preserve the old mixed-workload holdout protocol or cover
its code/general categories; configure the intended suite and required services
before production. The published evaluator image was exercised on tiny MoE;
full-model memory, tasks and serving settings still require qualification.

Set `launch.secrets.BEAKER_TOKEN` to an accessible Beaker secret. For automatic
scores on the training dashboard, set online W&B and its API-key secret; with the
offline default, sync training and manually publish retained evaluation results.
Snapshots require manual cleanup after all referencing jobs stop. Submission is
best-effort: missing evaluations do not stop training. See
[background evaluation](../../../docs/miles/background-evaluation.md) for image
pins, credentials, publication and cleanup. The evaluator GPU is additional to
the 56-GPU training allocation; overlapping jobs can use more than one extra GPU.

All four RL examples disable router balancing and z-loss (`router_aux_loss_weight=0.0`
and `router_z_loss_weight=0.0`). This is the example recipe; low-level CoreConfig
defaults remain unchanged. The controlled task sweep motivated aux-off, while
the mixed-workload comparison is still in progress; this is not a claim that
aux-off is best for every model or workload.

Router grouping, auxiliary averaging and balancing-count source are explicit
in all four examples and preserve the existing pack/token/token/dispatch defaults.
See [router objectives](../../../docs/miles/core.md#router-auxiliary-objectives)
for alternatives and their runtime requirements. A qualification of the defaults
does not by itself exercise the optional objectives.

W&B defaults to offline; configure an API-key secret and online mode when needed.
Do not run any template unchanged: model, data, output and judge paths are
placeholders. Put all experiment-specific variants in ignored `runs/`.
