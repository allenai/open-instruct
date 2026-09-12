# Core compiler-cache lifecycle

`python -m scripts.miles.compiler_cache_run` runs one bounded, single-node command with private node-local compiler caches. It can restore a verified immutable generation before any child imports and publish a new generation after a successful command exits. This standalone wrapper is an opt-in qualification tool. The production Ray lifecycle described below automatically persists Triton caches for Core RL.

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
  --shared-root /weka/oe-training-default/open-instruct-compiler-cache/tmp-7d/standalone
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

A local RTX 4090 screen with Torch 2.13/CUDA 13 and the pinned image's Triton executed the actual kernel in separate cold/restored containers: exact output both times, nine compiler writes cold and zero restored, unchanged generation after republishing, and one relocated group. The first kernel launch measured 0.521s cold versus 0.237s restored; restoring the 83.6KB inventory took 2.6ms. This single small measurement establishes Triton artifact portability only, not training speedup. Raw reports are under `/tmp/miles-validation/compiler-cache-probe/`; [the retained measurement](../measurements/core-compiler-cache-20260910.json) records their hashes and source/runtime provenance. The GPU screen preceded the final conservative environment-prefix expansion and remote-cache rejection; the final 33-test CPU suite covers those refinements.

Remaining gates beyond the qualified Triton trials below: actual TileLang/Inductor/FA4/DeepEP/DeepGEMM artifact relocation and reuse; multi-node Ray operation; and compatibility across replacement nodes of the same hardware class. Image, driver, architecture or compile-setting changes intentionally miss. FlashInfer and CUDA graph recordings are not covered.

## Ray startup integration

The MILES/Core driver now has a worker-level Triton lifecycle controlled by
`core.compiler_cache`, enabled by default. Set it to `false` to opt out.
Enabled runs use
`core.compiler_cache_root`, or the mounted default
`/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d`.
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
private copies. Publication runs at most two tasks per node, with one shared
240-second wait budget across all workers (including queued tasks), instead of
120 seconds sequentially for every rank. Expired tasks are explicitly cancelled
in Ray with retries disabled. This budget covers the Ray publication wait;
reading/writing the small completion report is separate.

Snapshotting, extracting old generations, merging, hashing and gzip compression
happen on local disk. Only the finished archive and manifest are uploaded to
shared storage before atomically advancing `CURRENT`. The gzip archive format
and integrity/relocation checks remain compatible with existing generations.
Logs identify task submission/start, lock wait, merge, compression, upload and
pointer phases; completed reports include file counts, bytes and phase timings.
[Local proxy and timeout validation](measurements/cache-publication-20260912/README.md)
cover the reliability change; WEKA performance remains to be observed on the next run.

Failed training does not publish. Cache miss/rejection compiles
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

The real local Ray/spawn probe passed after adapting the pinned Ray 2.58
actor-options setup-hook translation: exact vector-add output in both processes,
nine compiler writes cold and zero restored, an actual restored group hit, and
unchanged republished generation. See [the retained probe](measurements/startup-ray-probe-20260911.json).
This tests the actual actor environment and spawned-child observation boundary,
not full-model speed. The first full-model submission `01M2954XAX1ETXNSQN14HT13KX`
was stopped while queued, before allocation, to include this fix.


### Complete local hybrid-MoE screen

A local RTX 4090 run exercised two fresh Ray lifetimes through the public
`open_instruct.miles train` command, with the existing tiny two-block hybrid-MoE
fixture, one resident trainer and one TP1 serving engine sharing the GPU. Each
arm completed two optimizer updates, initial all-weight equality, and repeated
publication/reset/equality checks after each update. The cache configuration and
runtime identity are retained in [the measurement](measurements/startup-tiny-20260911.json).

| Interval | Cold | Restored |
| --- | ---: | ---: |
| Driver entry through first optimizer update | 173.57 s | 67.34 s |
| Serving startup, including health/warmup and initial snapshot/reset | 82.70 s | 42.13 s |
| Trainer startup | 12.31 s | 13.48 s |
| First training call, including scoring | 77.50 s | 11.23 s |
| Second training call | 0.091 s | 0.075 s |

The serving cache contained 46.7 MB and the trainer cache 83.9 MB. Both restored
workers consumed actual Triton groups, wrote **zero** new compiler artifacts,
and republished unchanged generations. Combined restore time was 0.62 s;
combined cold publication time was 1.95 s. HF read/conversion was negligible
for this tiny fixture, so these numbers do not estimate full-SFT startup.
The first-update interval starts inside the driver and excludes earlier Python
imports and `ray.init`; nested intervals must not be added to it.

Generation sequences **and their log-probabilities matched as multisets** in both
updates. Two final tokens exchanged sample IDs between duplicate prompts at
update zero; per-request ordering was not exact. Gradient diagnostics, sampled
update norms, and auxiliary/policy objective diagnostics matched. The random
tiny model earned zero task reward, so the updates exercise auxiliary gradients
and the startup contract, not task learning. TileLang and other compiler families
remained cold/separate; CUDA graphs were disabled in this local profile.

The first local attempt lacked the external-model registration environment
variable and failed before trainer startup; it published no cache. The corrected
run set `SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models`, already supplied by
the Beaker launcher.

The full SFT EP2/TP1 comparison is submitted as
[Beaker 01M295M9QNN2ME018S25WM6P40](https://beaker.org/ex/01M295M9QNN2ME018S25WM6P40)
using image `01M295M38BD4W67Q5P6FV5FXT6` and source `4936986ce`.
At 2026-09-11 21:45 UTC it remained queued on Holmes (urgent, minimum runtime
one hour). That historical queue snapshot is superseded by the completed result below.


### Full-SFT result and promotion

The EP2/TP1 full-model job completed successfully at 2026-09-11 22:55 UTC.
The retained report also passes the newer post-hoc gate requiring two optimizer
steps on each rank, correct sample counts and behavior versions, and the complete
initial/updated publication sequence. [Full measurement](measurements/startup-full-sft-20260911.json).

| Interval | Cold | Restored |
| --- | ---: | ---: |
| Driver entry through first optimizer update | 830.52 s | 452.87 s |
| Serving startup | 404.46 s | 216.85 s |
| Trainer startup | 125.86 s | 127.86 s |
| First training call, including scoring | 279.91 s | 86.18 s |
| Whole command, including cache publication | 1031.79 s | 720.69 s |

All three restored workers consumed Triton groups with zero new compiler writes
and unchanged republished generations. HF reads took approximately 18–25 seconds
per rank and HF-to-native conversion approximately 40–42 seconds in both arms.
The measured improvement therefore was not explained by a faster HF import.
Caches totaled about 286 MB uncompressed. Publication remains expensive: the
current publisher stages/verifies many small files on WEKA, taking 174 seconds
cold and 238 seconds on the unchanged restored arm. This is a follow-up optimization;
whole-command savings above already include this cost.

### Shared path, retention, and archive format

The default is a shared, user-independent root that follows olmo-miles' TTL naming convention:

```text
/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d/
  core-v1/<fingerprint>/triton/
    CURRENT
    .publish.lock
    generations/<inventory-sha256>/
      manifest.json
      cache.tar.gz
  runs/<run-id>/<worker-report>.json
```

The `tmp-7d` component marks artifacts and reports for the existing WEKA TTL
cleanup system. No separate cleanup daemon or retention metadata file is added.
Missing/expired artifacts simply cause a cold miss. Custom WEKA roots must have
an exact `tmp-N[hdwmy]` component, and roots must be absolute. Invalid retention
paths now fail configuration validation before launching workers or creating reports.

Olmo-miles also uses versioned, fingerprinted, checksummed immutable generations,
but its layout starts with `v1/<family>/<fingerprint>` and archives use Zstandard
(`cache.tar.zst`). Core currently uses gzip and its separate `core-v1`
namespace; the archive formats are deliberately not treated as interchangeable.
The TTL naming convention is the same.

The researcher interface exposes the default and opt-out as:

```toml
[compiler_cache]
enabled = true
# shared_root = "/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d"
```

Low-level files use `[core] compiler_cache = true` instead. Set the corresponding
boolean to `false` to disable persistence; compiler diagnostics stay off by default.


### Automatic miss, reuse, and deliberate invalidation

The final local screen ran three fresh public-CLI processes with restoration
**enabled in every arm**, using the default-on Core policy. Each completed two
updates and initial plus post-update weight-equality checks:

| Arm | Trainer and serving restore | Compiler activity | Publication |
| --- | --- | --- | --- |
| Empty shared cache, Core max sequence 512 | miss | new artifacts | published |
| Same settings, different dataset/output file paths | hit | zero new writes | unchanged |
| Only Core max sequence changed to 256 | miss | new artifacts | published |

The audit compares the full recorded fingerprint inputs: the only identity change
in the third arm is `compile_settings.core.max_sequence_length`, for both workers.
Dataset files had identical contents at different paths, demonstrating that paths
and run labels do not bust the cache. Generations and log-probabilities matched
as multisets between the first two arms. [Evidence and reproduction script](measurements/startup-cache-compatibility-20260911.json).

The screen ran with the default-on/early-TTL-validation patch. The final validation
also rejects the bare `/weka` root, covered by the CPU tests. No model arithmetic,
weight conversion, optimizer, or OLMo-core pretraining defaults changed here.
