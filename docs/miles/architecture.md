# Architecture and development

The public entrypoint is `python -m open_instruct.miles`. Its structured RunSpec
compiles into a RunConfig containing CoreConfig and native MILES options.
The Python API is available for integrations; use the structured workflow when
preparation, placement and submission should come from one file.

| Layer | Owns |
|---|---|
| Open Instruct wrapper | Config validation, preparation, verifiers, launch/receipts, retained evidence |
| MILES runtime | Rollout actors, SGLang engines, sample/advantage/loss machinery, shared transport helpers |
| Core adapter | Model construction, document/replay alignment, scoring/training forward, gradient synchronization, optimizer/checkpoints and tensor export |
| OLMo-core | Model and distributed-training primitives, attention/expert execution |
| olmo-sglang | Serving architecture registration and model-specific execution |

The Core actor implements the MILES trainer boundary; no Megatron trainer is
selected. The driver coordinates collections, optimization and versioned weight
publication. Dense standard-model support is isolated from the specialized MoE
path. [Implementation contracts](core.md) describe the detailed lifecycle and
[packing](sequence-packing.md) describes document/routing alignment.

## Runtime sources and images

`runtime/miles/runtime.lock.json` pins every runtime source to an exact Git commit;
image builds fetch those commits without applying patches. `olmo-sglang` uses its
merged main commit `72f194a35045f02cc7d87980819bd0e4652cc931`. OLMo-core is pinned
to `e505356353aa7ce1f6ff83e24d6eb945f463714e` on `robertb/miles-rl-main`.
That branch starts from Jacob's [production MoE PR #872](https://github.com/allenai/OLMo-core/pull/872)
and carries the MILES adapter plus inherited HF interchange support. MILES uses
`571560bb4d22259cd1373dc6a2bfa1fb7551d651` on
[`allenai/miles:main`](https://github.com/allenai/miles/tree/main).
This integrates the Open Instruct runtime changes on upstream
`e89b45f7f85a0e76fba6a99474b1dd9b67a3c20e`. Worker launch, inference lifecycle,
and rollout execution follow the upstream ownership boundaries. Builds fetch exact
commits, not moving branch tips. See the [migration checks](measurements/miles-upstream-20260922.md)
for validation and dependency limits.
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

CPU-only plan/structured validate exercise the public configuration contract.
For a GPU lifecycle check use the [tiny example](https://github.com/allenai/open-instruct/blob/fe4d9f2bdc994adb35f839718d86e420d8481e12/configs/miles/examples/dev.toml)
and a compatible tiny HF fixture in the pinned runtime. The existing
[historical local MoE and restart procedure](measurements/implementation-history/core-before-sharing-20260913.md#local-moe-task-and-restart-check) covers
model preparation and execution. A host-only parser test cannot qualify routing,
attention, distributed gradients or publication.

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

### Type checking with an external Core checkout

The repository's type-check command does not require a hidden runtime source
directory. The MILES adapter still needs the pinned Core APIs: an older installed
Core can report missing members. To resolve its types against the exact Core
branch during MILES development, pass that source path explicitly:

```bash
uv run ty check --extra-search-path /path/to/OLMo-core/src
```

Do not add an unconditionally required, ignored runtime directory to the global
`tool.ty.environment.extra-paths`: a fresh clone has no such directory and the
checker exits before examining any files. The runtime image and dependency lock
continue to define the actual training implementation.
