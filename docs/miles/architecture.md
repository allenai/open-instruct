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

`runtime/miles/runtime.lock.json` and its checksum-verified patches reconstruct
the runtime. Working branches help development, but the lock/image determines a
run. Source changes require a new overlay image; dependency/kernel changes may
require a qualified new base. Reusing an image does not apply local source edits.

```bash
python scripts/miles/prepare_runtime.py runtime/miles/sources
python scripts/miles/build_image.py --base-image LOCAL_LOADED_BASE_IMAGE --tag open-instruct:miles-core
```

prepare_runtime refuses to replace existing source directories. Its --cache
arguments can use local clones as fetch sources; they do not select uncommitted
sibling code. build_image checks the immutable base Docker ID. For actual
submission use the [committed-image launcher](launching.md), not a one-off Beaker
command with different mounts or source provenance.

## Local development

CPU-only plan/structured validate exercise the public configuration contract.
For a GPU lifecycle check use the [tiny example](../../configs/miles/examples/grpo-basic.toml)
and a compatible tiny HF fixture in the pinned runtime. The existing
[local MoE and restart procedure](core.md#local-moe-task-and-restart-check) covers
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
