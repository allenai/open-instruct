# MILES GRPO documentation

Use **`python -m open_instruct.miles`** for new GRPO work in this project.
Open Instruct owns the researcher configuration and data/reward integration; MILES
owns rollout orchestration and shared RL machinery; the adapter trains through
OLMo-core and serves through SGLang. Check model and workload support below.

The older `grpo.py` Core/vLLM and `grpo_fast.py` DeepSpeed/vLLM entry points are
**deprecated** and retained for existing runs and historical reproduction only.
Their [legacy reference](../algorithms/grpo.md) does not define MILES behavior.
If a required capability is missing here, identify the gap rather than silently
switching to a deprecated backend.

## Start here

1. Start with the [MILES GRPO guide](grpo.md) for setup, the runtime image and a first run.
2. Check [support and limits](feature-parity.md), [model support](models-and-checkpoints.md)
   and [topology](topology.md), and read the [development defaults](development-defaults.md).
3. Copy a [structured example](../../configs/miles/examples/README.md).
4. Read the [workflow](workflow.md), then follow the [launch guide](launching.md)
   for your host. Planning does not require GPUs or mounted checkpoints.
5. Use [operations](operations.md) to check completion and inspect run artifacts.

## Documentation map

| Document | Use it when |
|---|---|
| [MILES GRPO](grpo.md) | Setting up MILES GRPO, selecting its runtime image and launching a first run |
| [Support and limits](feature-parity.md) | Checking what the integration supports and its current limits |
| [Development defaults](development-defaults.md) | Choosing starting settings and what to weigh when changing them |
| [Workflow](workflow.md) | Editing one TOML from preparation through training and export |
| [Launching](launching.md) | Submitting from a laptop or Beaker session, or executing in an allocation |
| [Configuration reference](configuration.md) | Looking up fields, defaults, aliases, restrictions and overrides |
| [Native option appendix](native-options.md) | Looking up an advanced MILES/SGLang flag or its original help |
| [Models and checkpoints](models-and-checkpoints.md) | Selecting a supported checkpoint, converting, saving, resuming or exporting |
| [Core-compatible serving](core-compatible-serving.md) | Choosing the scoped fused-rounding default, opt-out and numerical reference modes |
| [Throughput](throughput-profiles.md) | Sizing trainer/inference allocation and engine admission, and reading queue, waste and warmup measurements |
| [Async queues and discard metrics](async-pipeline.md) | Sizing producer and completed buffers, tracing waits, and measuring discarded work |
| [Topology and capacity](topology.md) | Choosing GPU placement, async scheduling, batch geometry and serving capacity |
| [Data and evaluation](data-and-evaluation.md) | Selecting tasks, importing mixtures and retaining held-out generations |
| [Background evaluation](background-evaluation.md) | Configuring best-effort external olmo-eval jobs, receipts and W&B publishing |
| [Managed judges](managed-judges.md) | Binding named rubrics and placing fixed-weight judge services |
| [Operations](operations.md) | Reading metrics, diagnosing failures and establishing run completion |
| [Architecture and development](architecture.md) | Understanding adapter ownership, runtime images and local checks |
| [Long sequences](long-sequences.md) | Choosing prompt/response budgets, admission and memory controls |
| [Packing](sequence-packing.md) | Understanding document isolation and loss semantics |
| [Compiler caches](compiler-cache.md) | Understanding cache restore/publication and bounded shutdown |
| [Inference records](inference-records.md) | Recording every scored group's outcome by prompt and checkpoint for later selection and analysis |
| [Run-control semantics](run-controls.md) | Comparing Core controls with the Megatron implementation's terminology |
| [Router auxiliary objectives](core.md#router-auxiliary-objectives) | Selecting grouping, averaging, count source and coefficients |
| [Implementation contracts](core.md) | Reviewing detailed trainer, replay and publication checks |

## Authority and maintenance

Current instructions live in this directory. Example TOMLs express starting
recipes. Configuration code, the pinned native parser snapshot, and
`runtime/miles/runtime.lock.json` define accepted settings and runtime provenance.

Maintain the generated reference with `python -m scripts.miles.generate_docs`
and check it with `python -m scripts.miles.generate_docs --check`. See
[development](architecture.md#documentation-checks) for validation.

See [publication modes](grpo.md#publication-modes) for current mixed-policy
refresh, barrier controls and independent [engine drain](engine-drain.md).
