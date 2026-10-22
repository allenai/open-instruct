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
Builds fetch those commits, including SGLang itself, without patches or moving branch tips. MILES uses the
organized adapter hooks; OLMo-core provides the model, objective and checkpoint
APIs required by this adapter; olmo-sglang supplies compatible serving models.
The ordinary Open Instruct environment keeps its own dependency versions.

Working branches help development, but the lock/image determines a
run. Source changes require a new application image; dependency/kernel changes may
require a new, validated binary base. The Dockerfile separates a `runtime-base`
stage (locked dependency sources and verifier packages) from `application`
(Open Instruct, tests, scripts, configs and MILES docs). Build the former with
`build_image.py --target runtime-base` to reuse the prepared layer. Ordinary builds
select `application` and reuse that layer through Docker caching. The binary base is pinned separately from the prepared layer. The Miles v0.1.2
candidate reuses the binary stack from the local 4090 qualification image, including
`sglang-kernel==0.4.7`; its inherited application overlay is removed before the
locked sources are fetched. The prepared layer is not interchangeable with that pin. Image metadata records the application Git revision. Reusing an image does not apply local source edits.

The main `runtime/miles/Dockerfile` adds FlashAttention 2 and 3 to the original
binary base. Its separate `flash-attention-build` stage compiles FA3 from an
immutable source commit against that base's Torch/CUDA libraries, and downloads
checksum-verified FA2 and FA4 wheels. The runtime stage installs FA3 and FA2,
then restores FA4's overlapping files. Source trees, compiler outputs and wheels
stay in the build stage. A cold build needs network access and CUDA compilation
time, but no GPU; Docker caches this work independently of application changes.
FA3 compilation defaults to two jobs (`FLASH_ATTENTION_MAX_JOBS` build argument).
Build-time imports check that the extensions load; submitted jobs run the GPU
forward/backward preflight for the selected backend.

```bash
python scripts/miles/build_image.py --base-image LOCAL_LOADED_BASE_IMAGE --tag open-instruct:miles-core
```

The build needs read access to the private `allenai/miles-olmo-core` and `allenai/olmo-sglang` repositories.
`build_image.py` uses `GH_TOKEN`, `GITHUB_TOKEN`, or the active `gh auth login`
credential, passed through a temporary BuildKit secret. It is not stored in image
layers or Git URLs. For standalone source preparation, pass
`--github-token-file PATH` or use `--cache olmo-sglang=/path/to/clone`.
Existing published images retain their original sources; rebuild after changing
the lock.

prepare_runtime refuses to replace existing source directories. Its --cache
arguments can use local clones as fetch sources; they do not select uncommitted
sibling code. build_image checks the immutable base Docker ID. For actual
submission use the [committed-image launcher](launching.md), not a one-off Beaker
command with different mounts or source provenance.

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

For a training lifecycle check, copy [small.toml](../../configs/miles/examples/small.toml)
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
by the runtime. The adapter interfaces it relies on are proposed upstream in
[OLMo-core PR #888](https://github.com/allenai/OLMo-core/pull/888).

Core's routed-expert `match_eager_rounding` defaults to false. The pinned MILES
adapter explicitly enables it for policy scoring and accepts legacy MILES
checkpoint manifests with equivalent scoring semantics. Other Core users keep
the default behavior.

Core's PR checks and application validation cover different boundaries.
Rebuild runtime/application images from the lock and validate the intended
training topology; an existing image retains its original source revisions.
Record application validation with the exact image and source pins. The compact
options snapshot must be regenerated whenever runtime source pins change.

### Optional local integration group

Run the isolated group explicitly inside the candidate application runtime:

```bash
python tests/miles/qualification/runner.py --case all \
  --output /results/tiny-hybrid-v1 \
  --miles-source /opt/core-rl/sources/miles \
  --image-id sha256:IMAGE_ID
```

Use a new output directory each time. `--case lifecycle` runs three uninterrupted
updates; `--case resume` or `all` also runs two updates, restarts the process for
update three, and compares the result with the control. Each process has a
600-second timeout. The group requires exactly one visible GPU and the complete
runtime; missing prerequisites fail. It is a standalone command, outside ordinary
pytest discovery and the default CPU/GPU CI selections, and submits no cluster jobs.

`tests/miles/qualification/fixture.json` fixes the tiny target-family hybrid-MoE
geometry, seed, three arithmetic prompts and comparison tolerances. Preparation
creates a local tokenizer and random BF16 weights without network downloads.
Each update processes one prompt with four greedy responses. Router auxiliary
losses provide nonzero gradients despite zero task advantages. This checks
integration and resume consistency, not task learning or sampled-generation RNG
continuity. The existing sampled GSM8K smoke is a separate check.

The audit requires correct policy versions and prompt cursor, valid routing
replay, finite nonzero gradients and updates, complete native checkpoints,
serving/training logprob error at most 0.05, and exact HF export round-trip.
The resume comparison requires identical generated tokens and routes, scheduler
and data state, and compares all native model/Adam state with `rtol=1e-5` and
`atol=1e-6`. These are strict acceptance thresholds, not a promise of bitwise
identity across hardware. A failure must be investigated before changing them.

Retain `result.json`, provenance, fixture hashes, process logs, audit JSON and
comparison metrics. Provenance records imported source paths and Python source
hashes as well as available Git revisions; the container launcher supplies the
immutable image ID. A checked-out revision alone does not identify local edits.
Keep the trained H045 two-GPU WEKA recipe as a separate qualification gate;
this local fixture does not replace it or qualify distributed/publication modes.

Adam first/second moments use the stricter absolute tolerance `1e-12` (relative
tolerance `1e-5`), since small moments must not compare equal to a reset optimizer.

The first validated RTX 4090 run completed this group in about seven minutes:
three control updates versus two plus a fresh-process resumed update. Generated
tokens/routes and all 560 compared floating checkpoint tensors matched exactly;
both exports reloaded and ran. Independent fixture preparations produced identical
file hashes. A separate fresh-container control differed after its first update,
however. The cause of that numerical variability is not established. Treat the
paired resume result as scoped evidence, and keep cross-container cold-start
repeatability separate from this gate; no stable golden numerical baseline or
required GPU CI check is claimed. Do not widen tolerances to hide that finding.

## Miles v0.1.2 candidate

The candidate pins the merged fork tag `v0.1.2` (`9ea6759`), SGLang
`b84bec5`, and the olmo-sglang rank-logging fix `109f881` from
[PR #8](https://github.com/allenai/olmo-sglang/pull/8). OLMo-core remains at
`7e97bc4`. The branch contains the consumer lifecycle updates and optional
qualification fixtures. The parser snapshot was regenerated against these sources.

The runtime build installs matching FlashInfer Python and cubin packages from
`runtime/miles/serving-requirements.txt` (0.6.18 for this SGLang revision). Its build
check compares the installed version with SGLang's source requirement, validates
FlashInfer's direct dependencies, and imports the Blackwell BF16 kernel modules.
The inherited Torch/CUDA stack remains pinned. This targeted check does not
qualify every optional SGLang backend. `inference.sglang_bf16_gemm_backend = "torch"`
remains available for the existing Torch path; automatic backend selection is
exercised separately on Blackwell.

Tracking initializes before launching the Miles worker manager so the serialized
worker arguments include the primary W&B run ID, including in offline mode.

The earlier source-mounted 4090 qualification passed; a newly built application
image still needs its own exercise. Candidate image identities and exact build
revision are recorded in `runtime/miles/candidate-image.json` after publication.
Use its `application.beaker` ID with `MILES_EXISTING_IMAGE` to select that exact
image. This candidate is not a claim that the trained H045 or distributed gates
have passed.
