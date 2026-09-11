# Opt-in Core compiler-cache lifecycle

`python -m scripts.miles.compiler_cache_run` runs one bounded, single-node command with private node-local compiler caches. It can restore a verified immutable generation before any child imports and publish a new generation after a successful command exits. This is an opt-in qualification tool; existing training launchers and running experiments are unchanged.

The design follows `olmo-miles/src/olmo_miles/runtime/{compiler_cache,cache_lifecycle}.py` and `olmo-miles/docs/runtime-images.md`, with a Core-specific fingerprint and explicit Triton path relocation. It does not reuse Megatron's model/layout fingerprint.

## Identity and lifecycle

The key includes the caller's immutable Docker SHA256 or Beaker image ID, the entire runtime lock, hashes of actual Core/MILES/open-instruct/olmo-sglang code files, HF architecture config, and Core/MILES run settings. It also includes Python/package versions, actual visible GPU properties, NVIDIA driver, CUDA/C/C++ compiler versions, and hashes of compile-affecting environment variables. The environment families include `OLMO_`, `SGLANG_`, `NVTE_`, `FLA_`, `TILELANG_`, `TRITON_`, `TORCHINDUCTOR_`, `PYTORCH_`, `FLASH_ATTENTION_`, `CUTE_`, and `CUDNN_`; selected compiler/precision variables are included explicitly. Values are hashed so broad environment capture does not expose credentials.

Checkpoint weight paths, output paths, reward config and W&B labels are excluded. Model architecture, EP/DP dimensions, context budgets, activation checkpointing and compile flags remain. An external `core.model_config` is rejected until its source dependency graph is qualified. Source/config identity is checked again before publication. The supplied image ID is provenance supplied by the caller, not an independent image attestation.

Every invocation creates new private directories for Triton, TileLang, TorchInductor, FA4, DeepEP, and DeepGEMM. Shared/network filesystems are rejected for mutable directories; published WEKA roots require a TTL component such as `tmp-30d`. Families with no generated files stay empty. Custom Triton cache managers, kernel override directories and explicitly enabled Inductor remote caches are rejected.

Published layout:

```text
SHARED/core-v1/FINGERPRINT/FAMILY/
  CURRENT
  .publish.lock
  generations/INVENTORY_SHA256/
    manifest.json
    cache.tar.gz
```

Generations have per-file names, sizes, executable bits and SHA256 checksums plus an archive checksum. Restore verifies the manifest identity, archive and every extracted file in staging, rejects links/traversal/duplicates, and then atomically installs the complete private copy. Incompatible or corrupt restore is a recorded cold miss (`status: rejected`), without partial installation. Corrupt published generations are never repaired implicitly.

Publication locks one fingerprint/family, verifies the previous generation, merges disjoint entries, and rejects conflicting bytes under the same cache key. The new generation is completed before atomically replacing `CURRENT`; previous generations are never modified. Existing generation directories are verified before reuse. Archives use gzip for a stdlib-only implementation; archive cost must be measured on real training caches.

Triton's `__grp__*.json` stores absolute artifact paths. Publication canonicalizes these known fields to relative paths and restore points them at the new private directory. Unknown files embedding the current private root reject publication. Other families' relocation formats have **not** been qualified; that scan is not a guarantee against every possible foreign absolute path or dynamically generated module reference.

The child must finish its whole process group. Leftover workers prevent publication; the wrapper kills that owned process group and retains its private cache for inspection. Interruptions also prevent publication. Detached/daemonized workers, an existing external Ray cluster, remote Ray actors and multi-node propagation are outside this wrapper's qualified scope. Do not wrap such a launcher until explicit per-node environment/lifecycle wiring is added.

## Cold/restored screen

Run both commands in the same allocated GPU job, with identical image, source trees, configuration and child workload. Use the actual deployed source locations; add `--source NAME=PATH` for custom code dependencies. The wrapper must be the outer command before training/serving imports. The fingerprinted TOML must describe the actual child configuration; arbitrary CLI overrides cannot be inferred automatically.

For a self-contained portability screen, the included probe executes an exact FP32 vector add and counts real Triton `FileCacheManager.put` calls. Replace the paths below with those present in the image:

```bash
cache_args=(
  --shared-root /weka/oe-training-default/robertb/tmp-30d/core-compiler-cache
  --local-parent /tmp
  --image "$IMMUTABLE_IMAGE_ID"
  --runtime-lock runtime/miles/runtime.lock.json
  --hf-config "$HF_PATH/config.json"
  --run-config "$RUN_TOML"
  --source olmo-core=/opt/core-rl/sources/olmo-core/src
  --source miles=/opt/core-rl/sources/miles/miles
  --source open-instruct=/opt/core-rl/open-instruct/open_instruct
  --source olmo-sglang=/opt/core-rl/sources/olmo-sglang/src
  --source qualification=/opt/core-rl/open-instruct/scripts/miles
)
python -m scripts.miles.compiler_cache_run "${cache_args[@]}" \
  --mode cold --publish --report /output/cache-cold.json -- \
  python -m scripts.miles.compiler_cache_probe \
    --output /output/kernel-cold.json --expect-compiler-writes present
python -m scripts.miles.compiler_cache_run "${cache_args[@]}" \
  --mode restore --publish --report /output/cache-restored.json -- \
  python -m scripts.miles.compiler_cache_probe \
    --output /output/kernel-restored.json --expect-compiler-writes absent
```

For the later training screen, substitute the same tiny bounded Core training command in both invocations, using separate output directories and no overlapping writers. Require unchanged correctness checks and compare first optimizer-step latency, whole-command latency, restore/publish costs, and observed compiler activity. Keep the original cold run: a cache hit alone does not establish useful end-to-end speedup. Decode graph capture is process-local and is not persisted by these compiler archives.

Reports record the full key inputs, family hits/misses/rejections, generation IDs, relocated group counts, bytes, probe/restore/command/recheck/publication timings, child return code and lifecycle outcome. A child failure is propagated. Optional cache publication rejection is recorded per family without turning a successful workload into a training failure; inspect those records before claiming a warm run.

## Validation and remaining gates

CPU tests cover immutable generations, changed compile inputs, changed source during execution, concurrent merge/conflict handling, corrupt archives/manifests, unsafe archive members, real Triton group restoration after deleting the old private root, and real child success/failure/leftover-worker lifecycle.

A local RTX 4090 screen with Torch 2.13/CUDA 13 and the pinned image's Triton executed the actual kernel in separate cold/restored containers: exact output both times, nine compiler writes cold and zero restored, unchanged generation after republishing, and one relocated group. The first kernel launch measured 0.521s cold versus 0.237s restored; restoring the 83.6KB inventory took 2.6ms. This single small measurement establishes Triton artifact portability only, not training speedup. Raw reports are under `/tmp/miles-validation/compiler-cache-probe/`; [the retained measurement](measurements/core-compiler-cache-20260910.json) records their hashes and source/runtime provenance. The GPU screen preceded the final conservative environment-prefix expansion and remote-cache rejection; the final 33-test CPU suite covers those refinements.

Remaining gates: same-job Core training cold/restored screen; actual TileLang/Inductor/FA4/DeepEP/DeepGEMM artifact relocation and reuse; launch-time Ray environment propagation on every node; realistic WEKA archive timing; and compatibility across replacement nodes of the same hardware class. Image, driver, architecture or compile-setting changes intentionally miss. FlashInfer and CUDA graph recordings are not covered.

## Ray startup integration (qualification in progress)

The MILES/Core driver now has a worker-level Triton lifecycle controlled by
`core.compiler_cache`. It is temporarily off by default while the full-model
cold/restored qualification runs. Explicitly enabled runs use
`core.compiler_cache_root`, or the mounted default
`/weka/oe-training-default/olmo-miles/compiler-cache/tmp-30d/core-rl`.
With no default WEKA mount, automatic persistence is skipped and logged.
`core.compiler_cache_restore=false` supplies the cold control while still
publishing after success. `core.compiler_cache_diagnostics=true` records actual
Triton group reads and writes; leave this instrumentation off for ordinary runs.

Small MILES hooks supply a Ray setup hook and explicit per-worker environment.
Each trainer rank and TP1 serving engine restores into a new `/tmp/core-triton-*`
directory before actor imports. Keys include the logical worker slot and actual
node hardware/toolchain, so trainer/serving and distinct trainer autotuning
histories do not overwrite one another. SGLang spawned children inherit their
engine's cache. Diagnostics additionally install a private Python startup hook
for those children; it does not alter kernel selection or ship in cache artifacts.

Only Triton artifacts are restored/published in this integration. TP>1 serving
is excluded pending per-scheduler-rank wiring. Driver warmups/preflights and CUDA
graph capture are not covered. The separate standalone cache wrapper still
supports its broader experimental families. Core's no-gradient dynamic SwiGLU
row setting remains independent of persistence.

After successful driver cleanup (including shutdown of serving children), small
Ray tasks pinned to the original nodes publish immutable generations and clean
private copies. Failed training does not publish. Cache miss/rejection compiles
locally; publication failures are recorded, not treated as RL failures.
`compiler-cache.json` in the run's metric directory records each worker's cache
outcome and diagnostic counters. Raw worker identity/restore records remain in
the shared cache's `runs/` directory.

Startup timings are now retained separately: `driver_timing.jsonl` includes cache
preparation, placement, serving startup and trainer startup; `startup_rankN.jsonl`
includes distributed initialization, HF reads, native parameter initialization,
HF conversion, overall native model/optimizer construction and native restore.
Nested intervals overlap and must not be summed. Existing evaluation, publication,
scoring and training timings complete the timeline.

The paired full-SFT trial is launched through
`scripts/train/debug/miles_startup.sh`. It uses one three-B300 allocation for two
fresh Ray lifetimes, EP2 training and TP1 serving, with two short greedy rollouts
per arm. Other compiler families remain isolated. Both arms must complete, all
three workers must consume restored Triton groups, and compiler writes must fall.
HF read time is reported separately because filesystem page-cache warming is a
confound in a same-allocation comparison.
