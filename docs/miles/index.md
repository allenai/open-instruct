# MILES + OLMo-core GRPO

MILES is the preferred GRPO path in this project for **supported models and workloads**.
Open Instruct owns the researcher configuration and data/reward integration; MILES
owns rollout orchestration and shared RL machinery; the adapter trains through
OLMo-core and serves through SGLang. The existing `grpo.py` Core/vLLM and
`grpo_fast.py` DeepSpeed/vLLM paths remain available. See the
[GRPO implementation chooser](../algorithms/grpo.md#implemented-variants).

## Start here

1. Check [model support](models-and-checkpoints.md) and [topology limits](topology.md).
2. Copy a [structured example](../../configs/miles/examples/README.md).
3. Read the [workflow](workflow.md), then follow the [launch guide](launching.md)
   for your host. Planning does not require GPUs or mounted checkpoints.
4. Use [operations](operations.md) to check completion and inspect retained evidence.

## Documentation map

| Document | Use it when |
|---|---|
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
