# Architecture and development

The public entrypoint is `python -m open_instruct.miles`. Its structured RunSpec
compiles into a RunConfig containing CoreConfig and native MILES options.
The Python API is available for integrations; use the structured workflow when
preparation, placement and submission should come from one file.

| Layer | Owns |
|---|---|
| Open Instruct wrapper | Config validation, preparation, verifiers, launch/receipts, retained evidence |
| MILES runtime | Rollout actors, SGLang engines, sample/advantage/loss machinery, shared transport helpers |
| Core adapter (in MILES) | Model construction, document/replay alignment, scoring/training forward, gradient synchronization, optimizer/checkpoints and tensor export |
| OLMo-core | Model and distributed-training primitives, attention/expert execution |
| olmo-sglang | Serving architecture registration and model-specific execution |

The Core actor at `miles.backends.core_utils.actor.OLMoCoreTrainRayActor` implements
the MILES trainer boundary. Its model/checkpoint code and publication machinery
live in the pinned MILES fork. The driver coordinates collections, optimization and versioned weight
publication. Dense standard-model support is isolated from the specialized MoE
path. [Implementation contracts](core.md) describe the detailed lifecycle and
[packing](sequence-packing.md) describes document/routing alignment.

## Runtime sources and images

[`runtime/miles/runtime.lock.json`](../../runtime/miles/runtime.lock.json) is the
source of truth for exact dependency commits and the immutable binary base.
Builds fetch those commits without patches or moving branch tips. MILES uses the
organized adapter hooks; OLMo-core provides the model, objective and checkpoint
APIs required by this adapter; olmo-sglang supplies compatible serving models.
The ordinary Open Instruct environment keeps its own dependency versions.

Working branches help development, but the lock/image determines a
run. Source changes require a new application image; dependency/kernel changes may
require a qualified new binary base. The Dockerfile separates a `runtime-base`
stage (locked dependency sources and verifier packages) from `application`
(Open Instruct, tests, scripts, configs and MILES docs). Build the former with
`build_image.py --target runtime-base` to reuse the prepared layer. Ordinary builds
select `application` and reuse that layer through Docker caching. The immutable
binary base in the lock is unchanged; the prepared layer is not interchangeable
with that pin. Image metadata records the application Git revision. Reusing an image does not apply local source edits.

```bash
python scripts/miles/build_image.py --base-image LOCAL_LOADED_BASE_IMAGE --tag open-instruct:miles-core
```

The build needs read access to the private `allenai/miles` and `allenai/olmo-sglang` repositories.
`build_image.py` uses `GH_TOKEN`, `GITHUB_TOKEN`, or the active `gh auth login`
credential, passed through a temporary BuildKit secret. It is not stored in image
layers or Git URLs. For standalone source preparation, pass
`--github-token-file PATH` or use `--cache olmo-sglang=/path/to/clone`.
Existing published images retain their original sources; rebuild before using
this pin.

prepare_runtime refuses to replace existing source directories. Its --cache
arguments can use local clones as fetch sources; they do not select uncommitted
sibling code. build_image checks the immutable base Docker ID. For actual
submission use the [committed-image launcher](launching.md), not a one-off Beaker
command with different mounts or source provenance.

## Local development

The ordinary CPU suite covers configuration, planning, launch specifications,
rewards, data preparation and workflow contracts without installing MILES or
SGLang. The `MILES contracts` workflow also checks the maintained scripts and
checks application types without requiring the private runtime.
The moved adapter is checked against that Core API in the dedicated runtime
environment; Open Instruct's ordinary type check covers the application integration.

```bash
uv run pytest open_instruct/test_miles*.py
```

The dedicated runtime suite uses the separate application image. From its
`/opt/core-rl` working directory, run:

```bash
bash scripts/miles/test_runtime.sh -q
```

This command fails if runtime dependencies are missing. Without the explicit
runtime flag, ordinary CPU collection omits `tests/miles/` when MILES or SGLang
is absent. CUDA numerical cases skip when no GPU is visible. A passing CPU run
therefore does not qualify GPU kernels, distributed gradients or publication.
The ordinary Open Instruct GPU CI image does not include the private MILES
runtime; dedicated runtime checks are a separate maintainer validation step.

For a training lifecycle check, copy [small.toml](../../configs/miles/examples/small.toml)
to `runs/`, select a compatible tiny checkpoint, and follow the
[committed-image launcher](launching.md). Verify the rendered replica group and
retain image, config and completion artifacts. Exercise refresh separately when
changing publication behavior; a tiny mechanics check does not establish learning
quality. Archived research harnesses and their tests live at the
[pre-cleanup revision](https://github.com/allenai/open-instruct/tree/813bd5988beb16be5b4d879ee3e2c49d8d859ee5).

## Documentation checks

```bash
python -m scripts.miles.generate_docs --check
uv run pytest open_instruct/test_miles_documentation.py
uv run mkdocs build
make style-check
make quality-check
```

The generator derives inventories from configuration sources and pinned parser
metadata, and joins reviewed descriptions from docs/miles/reference-help.json.
Native help is captured separately from the pinned runtime, with its image and
parser provenance; parser defaults are not model-dependent resolved defaults.
Regenerate that snapshot with the documented capture command in the native appendix
when parser flags change. Update descriptions and rerun generation, reviewing the
diff. Do not edit generated tables manually or copy a measurement's settings into
current defaults without a deliberate recipe change.

### Type checking the moved adapter

In a development environment with the pinned MILES and Core sources, run:

```bash
ty check /path/to/miles/miles/backends/core_utils --extra-search-path /path/to/OLMo-core/src
```

Open Instruct's CPU checks do not fetch the private MILES repository. The committed
application image and runtime tests exercise both repositories together.

## Adapter package layout

`open_instruct/miles/` groups code by responsibility. The public CLI remains
`python -m open_instruct.miles`; package initializers do not load the GPU runtime.

| Package | Responsibility |
|---|---|
| `configuration` | RunSpec/CoreConfig, native options, validation, topology and capacity planning |
| `execution` | Preparation, Beaker submission, cluster bootstrap and training coordination |
| MILES `backends/core_utils` | Core actor, model backends, packing, optimizer scheduling, checkpoints and trainer diagnostics |
| `rollout` | Application data-source selection and rollout metrics; managed generation, admission and completed queues live in MILES `backends/core_utils/rollout` |
| MILES `backends/core_utils/publication` | Weight delivery, engine drain, policy versions and durable policy state |
| `rewards` | Verifiers, reward routing and managed judges |
| `datasets` | Dataset preparation, mixtures, inference records and prompt selection |
| `evaluation` | Background evaluation submission, workers and result publication |
| `infrastructure` | Application artifact publication and bounded infrastructure waits; compiler/HF cache lifecycle lives in MILES |

Keep diagnostics next to the component they observe. Keep configuration imports
CPU-only, model backends loaded on demand, and package initializers minimal.
MILES `backends/core_utils/data.py` adapts samples to the trainer; `rollout/data_source.py` owns
runtime data-source behavior; `datasets/run_data.py` prepares researcher inputs.

Python integration imports now use these package paths, for example
`from open_instruct.miles.configuration.run_spec import RunSpec`. The pinned
AllenAI MILES fork must use the matching trainer and rollout hook paths. Rebuild
the application image when updating this layout and its runtime pin together;
existing images retain their original code. Custom verifier factories and saved
native options containing old Python paths need the corresponding package prefix
before use with a new image. Historical run artifacts retain their original paths.

The CPU CLI checks in `open_instruct/test_miles_package_layout.py` run without
site-packages. `tests/miles/test_package_layout.py` resolves hooks supplied by
both repositories inside the pinned runtime.

### Application callbacks and CPU tools

The backend trainer and managed rollout modules do not import Open Instruct.
Open Instruct supplies `core_records_factory` for per-question recording and
`core_evaluation_snapshot` for requested post-update HF exports. Those callbacks
preserve the existing recording and evaluation schedules. Dataset preparation,
reward callbacks, data-source selection, Beaker launch and evaluation submission
remain application-owned. The native `olmo_core` argument loader still resolves
Open Instruct's run configuration; that is an explicit configuration integration.

Small CPU planning and artifact-reader helpers remain local so `plan`, `validate`
and offline analysis work without the private runtime. Runtime integration tests
compare their algorithms with the MILES counterparts: scoring policy, capacity
reporting, wait diagnostics, wire-version parsing, atomic JSON publication,
startup timing, model input checks and throughput arithmetic.
