# Architecture and development

The public entrypoint is `python -m open_instruct.miles`. Its structured RunSpec
compiles into a RunConfig containing CoreConfig and native MILES options.
The Python API is available for integrations; use the structured workflow when
preparation, placement and submission should come from one file.

| Layer | Owns |
|---|---|
| Open Instruct wrapper | Config validation, preparation, verifiers, launch/receipts, run artifacts |
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

The MILES pin includes the v0.1.2 async-consumer compatibility fix from
[miles-olmo-core PR #10](https://github.com/allenai/miles-olmo-core/pull/10).
Its refresh and engine-drain dequeues accept the native consumer's optional trainer
identity and preserve Core's explicit rejection of multi-policy training.

The binary base contains the compiled dependencies, including FA2, FA3 and FA4.
It excludes the MILES, Core, SGLang and Open Instruct source overlays. Application
builds add locked source repositories as `runtime-base`, then the committed Open
Instruct checkout as `application`. They do not compile FlashAttention, even
without a local Docker cache. Source-only lock changes rebuild source layers;
binary dependency changes require a new base. Editable installs still fetch
build-system dependencies such as setuptools and pybind11, and Megatron builds
a small C++ dataset helper. Those build-tool requirements are not fully pinned;
the source and base pins do not make application builds bit-for-bit reproducible.

The selected base is Beaker `01M5ZYQZVF7DDJ1VR619KRSS42`
(`robertb/miles-binary-base-20261027`), Docker
`sha256:947e615f93cb88f8739ba270f30adc7f012af8bc0abf43e96f11cd5d3b2d6354`.
Its package versions, CPU imports and source isolation were checked, and a pull
by immutable ID reproduced the same Docker ID. The
[single-GPU Core RL smoke](https://beaker.org/ex/01M61SA3GN3AKJA5KDD90J3JJA)
passed on application image `01M61RPR1SVAACNX7AAVH2ENYJ`: four optimizer updates,
shared-engine evaluations, checkpoints and HF export on Holmes. This tiny-model
check uses Torch attention; it does not qualify every FlashAttention backend or
a Qwen OPD GPU training run with the public Megatron sources.

A subsequent CPU-only `runtime-base` rebuild from commit
`3563eb52ac4b02d731786e8f1a4e39a52951f2a2` is published as Beaker image
`01M61YFEWE1BATA8KQ4VAV2DY8` (`robertb/miles-runtime-base-3563eb52a`), Docker
`sha256:f97469549508d2af9896161889cc45060c52ece5743ecfcbe937039d3043b784`.
Its packaged lock matches the current lock, all six source checkouts match their
pinned revisions without patches, and serving/attention CPU imports passed.
This records CPU build verification of the runtime sources separately from the
application-image GPU smoke above.

The source layer includes Megatron-LM from the Radix fork, pinned to
`235952df607b3820716e5e67728a5ab470ca33ae` (the previous runtime's revision,
including its Qwen3.5 FP32 optimizer fix), rather than the current `miles-main`
tip used by the pinned MILES Dockerfile. Megatron Bridge uses the public Radix
revision selected by that Dockerfile. These dependencies support Qwen OPD.
The image does not install `olmo-megatron`, the private Olmo Bridge fork, or
Olmo-specific Megatron patches; Olmo training uses the Core backend. Package
metadata and entrypoints are installed with `--no-deps` to preserve the binary
stack. `/src/Megatron-LM` remains a compatibility symlink for the OPD launcher.

EOPD metadata transport is implemented in the pinned MILES source, without a
local runtime patch. Open Instruct's `opd_hooks` stores teacher top-k values in
`Sample.train_metadata`;
[rollout conversion](https://github.com/allenai/miles-olmo-core/blob/296a1e80c9c726bc41a764ce7cfcab2c886da64a/miles/ray/rollout/train_data_conversion.py#L173-L174)
copies those dictionaries into `train_data["metadata"]`, and the
[Megatron training step](https://github.com/allenai/miles-olmo-core/blob/296a1e80c9c726bc41a764ce7cfcab2c886da64a/miles/backends/megatron_utils/model.py#L465-L496)
requests `metadata` from `get_batch` for the custom loss. This training field is
separate from the input dataset metadata selected by `--metadata-key`.

For ordinary application builds:

```bash
python scripts/miles/build_image.py --base-image LOCAL_LOADED_BASE_IMAGE --tag open-instruct:miles-core
```

Use `--target runtime-base` to build only the locked dependency sources, without
adding the Open Instruct application checkout. The default target is `application`.
The image arguments are local Docker tags loaded on the build host; Beaker names
and immutable IDs identify the images to pull before invoking the helper.

All repositories in the runtime lock are public; no GitHub credential is needed
for these source fetches. Standalone source preparation can use
`--cache olmo-sglang=/path/to/clone`.
Existing published images retain their original sources; rebuild after changing
the lock.

prepare_runtime refuses to replace existing source directories. Its --cache
arguments can use local clones as fetch sources; they do not select uncommitted
sibling code. build_image checks the immutable base Docker ID. For actual
submission use the [committed-image launcher](launching.md), not a one-off Beaker
command with different mounts or source provenance.

## Rebuilding the binary base

The reproducible maintainer recipe is `runtime/miles/Dockerfile.base`; its two
immutable inputs are recorded in `runtime/miles/base-build.lock.json`. Pull both
Beaker images into Docker, then run:

```bash
python scripts/miles/build_base_image.py \
  --compiled-parent LOCAL_COMPILED_PARENT \
  --flash-attention-builder LOCAL_FA3_BUILDER \
  --tag open-instruct:miles-binary-base
```

Both Docker IDs are verified, and the parent-image label comes from the same
lock. The helper passes the compiler image as `FLASH_ATTENTION_BUILDER` and the
runtime parent as `COMPILED_PARENT`. Only FA wheels are transferred from the compiler
image to the clean compiled parent. FA2/FA4 wheel downloads are checksum-verified;
the final stage resets inherited application metadata and entrypoints. This
preserves the qualified CUDA/PyTorch stack rather than rebuilding all dependencies
from an upstream CUDA distribution. Build on an amd64 Linux host: FA3 compilation
defaults to two jobs, with `--max-jobs` available for hosts with sufficient RAM.
The published base's FA3 build took about 31 minutes with eight jobs.

The final environment list was compared with the locked compiled parent. It
preserves its runtime settings while dropping legacy `OLMO_MILES_*` cache and
verifier paths. Recheck this list against the parent image whenever changing the
parent pin; `FROM scratch` does not inherit environment variables automatically.

Before changing the pin again, verify CPU imports, an application build and GPU
forward/backward and training on the intended hardware. CPU imports alone do not
establish GPU correctness.

## Local development

The ordinary CPU suite covers configuration, planning, launch specifications,
rewards, data preparation, rendezvous, heartbeat, recovery and topology contracts
without installing MILES or
SGLang. The `MILES contracts` workflow also checks the maintained scripts and
checks application types without requiring the private runtime.
Recording and summary implementation tests live with their MILES implementation;
Open Instruct keeps record configuration tests and runtime selection/buffer integration tests.
The moved adapter is checked against that Core API in the dedicated runtime
environment; Open Instruct's ordinary type check covers the application integration.
The MoE numerical tests deliberately require the Core revision in the runtime
lock. The ordinary `uv.lock` Core version is for existing SFT/DPO paths and is
not upgraded to make those runtime-only tests importable.

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
therefore does not check GPU kernels, distributed gradients or publication.
The ordinary Open Instruct GPU CI image does not include the private MILES
runtime; dedicated runtime checks are a separate maintainer validation step.

For a training lifecycle check, copy [dev.toml](../../configs/miles/examples/dev.toml)
to `runs/`, select a compatible tiny checkpoint, and follow the
[committed-image launcher](launching.md). Verify the rendered replica group and
retain image, config and completion artifacts. Test refresh separately when
changing publication behavior; a tiny mechanics check does not establish learning
quality.

## Documentation checks

```bash
python -m scripts.miles.generate_docs --check
uv run pytest open_instruct/test_miles_documentation.py
uv run mkdocs build
make style-check
make quality-check
```

MkDocs generates the four reference pages in memory from configuration sources,
the compact pinned parser schema and reviewed `reference-help.json` descriptions.
Neither the pages nor native help are tracked. Ordinary documentation builds and
local `plan`/`validate` commands need no MILES installation. Without a native help
capture, option tables show names, types, choices, restrictions and reviewed
supplements; other help and value defaults are explicitly marked unavailable.

When runtime pins change, regenerate the compact schema inside the matching
runtime and review its diff:

```bash
python -m scripts.miles.snapshot_options open_instruct/miles/configuration/options.json
```

For full upstream help and parser defaults, capture an artifact in the pinned
runtime using a writable checkout, then supply it to the documentation build:

```bash
python -m scripts.miles.capture_option_help --image IMMUTABLE_IMAGE_ID --output runs/native-help.json
MILES_NATIVE_HELP=runs/native-help.json uv run mkdocs build
```

The capture checks the installed parser against the expanded compact schema;
the docs build rejects stale hashes or missing actions. Parser defaults are not
model-dependent resolved defaults. To inspect Markdown separately, run
`python -m scripts.miles.generate_docs --output-dir runs/miles-docs` (optionally
with `--native-help runs/native-help.json`). `--check` validates the reference
inputs without writing files. Edit source definitions and reviewed descriptions,
not generated output. Change maintained examples only as a deliberate recipe
change; see [development defaults](development-defaults.md).

### Type checking the moved adapter

In a development environment with the pinned MILES and Core sources, run:

```bash
ty check /path/to/miles/miles/backends/core_utils --extra-search-path /path/to/OLMo-core/src
```

Open Instruct's CPU checks do not fetch the private MILES repository. The committed
application image and runtime tests check both repositories together.

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

Python integration imports use these package paths, for example
`from open_instruct.miles.configuration.run_spec import RunSpec`. The pinned
AllenAI MILES fork must use the matching trainer and rollout hook paths. Rebuild
the application image when updating this layout and its runtime pin together;
existing images retain their original code. Custom verifier factories and saved
native options must reference these package paths.

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
reporting, timeout diagnostics, atomic JSON publication, input validation and
throughput arithmetic.

## Core runtime compatibility

The `olmo-core` revision in
[`runtime.lock.json`](../../runtime/miles/runtime.lock.json) is the Core source used
by the runtime. The adapter interfaces it relies on were merged in
[OLMo-core PR #888](https://github.com/allenai/OLMo-core/pull/888). The runtime pins
merge commit `6ceda6530625caa5d23e3be8a455c600f62a7338`, including the reviewed
RoPE import fixes, alongside the MILES v0.1.2 integration.

Core's routed-expert `match_eager_rounding` defaults to false. The pinned MILES
adapter explicitly enables it for policy scoring and accepts legacy MILES
checkpoint manifests with equivalent scoring semantics. Other Core users keep
the default behavior.

Core's PR checks and application validation cover different boundaries.
Rebuild runtime/application images from the lock and validate the intended
training topology; an existing image retains its original source revisions.
Record application validation with the exact image and source pins. The compact
options snapshot must be regenerated whenever runtime source pins change.
