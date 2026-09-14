# MILES GRPO documentation

Use **`python -m open_instruct.miles`** for new GRPO work in this project.
Open Instruct owns the researcher configuration and data/reward integration; MILES
owns rollout orchestration and shared RL machinery; the adapter trains through
OLMo-core and serves through SGLang. Check model and workload support below.

The older `grpo.py` Core/vLLM and `grpo_fast.py` DeepSpeed/vLLM entry points are
**deprecated** and retained for existing runs and historical reproduction only.
Their [legacy reference](../algorithms/legacy_grpo.md) does not define MILES behavior.
If a required capability is missing here, identify the gap rather than silently
switching to a deprecated backend.

## Start here

1. Start with the [MILES GRPO guide](grpo.md) for setup, the runtime image and a first run.
2. Check [model support](models-and-checkpoints.md) and [topology limits](topology.md).
3. Copy a [structured example](../../configs/miles/examples/README.md).
4. Read the [workflow](workflow.md), then follow the [launch guide](launching.md)
   for your host. Planning does not require GPUs or mounted checkpoints.
5. Use [operations](operations.md) to check completion and inspect retained evidence.

## Documentation map

| Document | Use it when |
|---|---|
| [MILES GRPO](grpo.md) | Setting up MILES GRPO, selecting its runtime image and launching a first run |
| [Support matrix](feature-parity.md) | Distinguishing exercised paths, experiments and remaining gaps |
| [Workflow](workflow.md) | Editing one TOML from preparation through training and export |
| [Launching](launching.md) | Submitting from a laptop or Beaker session, or executing in an allocation |
| [Configuration reference](configuration.md) | Looking up fields, defaults, aliases, restrictions and overrides |
| [Native option appendix](native-options.md) | Looking up an advanced MILES/SGLang flag or its original help |
| [Models and checkpoints](models-and-checkpoints.md) | Selecting a supported checkpoint, converting, saving, resuming or exporting |
| [Topology and capacity](topology.md) | Choosing GPU placement, async scheduling, batch geometry and serving capacity |
| [Data and evaluation](data-and-evaluation.md) | Selecting tasks, importing mixtures and retaining held-out generations |
| [Managed judges](managed-judges.md) | Binding named rubrics and placing fixed-weight judge services |
| [Operations](operations.md) | Reading metrics, diagnosing failures and establishing run completion |
| [Architecture and development](architecture.md) | Understanding adapter ownership, runtime images and local checks |
| [Long sequences](long-sequences.md) | Choosing prompt/response budgets, admission, memory controls and interpreting length qualification |
| [Packing](sequence-packing.md) | Understanding document isolation, loss semantics and qualification |
| [Compiler caches](compiler-cache.md) | Understanding cache restore/publication and bounded shutdown |
| [Run-control semantics](run-controls.md) | Comparing Core controls with olmo-miles/Megatron terminology |
| [Implementation contracts](core.md) | Reviewing detailed trainer, replay and publication checks |
| [Measurements](measurements/index.md) | Finding point-in-time evidence, configurations and limitations |
| [Historical plans](plans/index.md) | Understanding previous proposals, not selecting operating defaults |

## Authority and maintenance

Current instructions live in this directory. Example TOMLs express starting
recipes; they are not frozen evidence. Configuration code, the pinned native
parser snapshot, and `runtime/miles/runtime.lock.json` define accepted settings
and runtime provenance. A branch name or a dated study is not a runtime pin.

Measurements establish only their recorded model, hardware, topology and feature
combination. A small execution check is not a learning comparison or a production
capacity qualification. Keep historical runs unchanged when updating starters.

Maintain the generated reference with `python -m scripts.miles.generate_docs`
and check it with `python -m scripts.miles.generate_docs --check`. See
[development](architecture.md#documentation-checks) for validation.

Experimental: [independent engine drain and rolling publication](engine-drain.md)
(isolated qualification; barrier publication remains the default).
