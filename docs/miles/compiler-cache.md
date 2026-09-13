# Compiler-cache lifecycle

Ordinary MILES/Core runs persist **Triton artifacts only**, through the Ray worker
lifecycle below. Mutable caches remain private on local disk; immutable verified
generations are restored from and published to WEKA. Persistence is an optimization:
a miss compiles locally, and optional publication failure does not fail completed
training. CUDA graphs are captured per process and are not saved this way.

```toml
[compiler_cache]
enabled = true
# shared_root = "/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d"
```

The root must be absolute; custom WEKA paths need an exact `tmp-N[hdwmy]` TTL
component. Expired artifacts are cold misses. No new cleanup daemon is required.

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
cover the reliability change. Compare actual publication reports on the selected image; old shared-filesystem staging timings do not describe the new local-staging implementation.

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


## Identity and archive contract

Keys include runtime/source identity, relevant model architecture and compile
settings, actual device/toolchain and compile-affecting environment hashes.
Output paths, checkpoint weight paths and tracking labels are excluded. Different
source, image, architecture or compile settings can intentionally miss; a new
sharing image must not assume every older cache is reusable.

Generations contain checksummed manifests and `cache.tar.gz`. Restore validates
identity, archive and individual files in staging before atomic installation;
corruption is a recorded rejection/cold miss. Triton group metadata is relocated
to the new private directory. Publication merges compatible entries under a lock
and atomically advances `CURRENT`; conflicting bytes are not silently overwritten.

Olmo-miles uses a separate namespace and Zstandard archives. Those are not
interchangeable with this gzip format. Local staging addressed the shared-file
I/O problem without a codec migration. Inspect measured lock/merge/compression/
upload phases before changing timeouts or attributing delay to gzip.

## What has been measured

The [full-SFT cold/restored screen](measurements/startup-full-sft-20260911.json)
observed restored Triton groups, zero new compiler writes and faster first updates
on all three workers. Its 174/238-second publication timings came from the old
WEKA staging implementation. The [publication fix](measurements/cache-publication-20260912/README.md)
qualifies local staging, shared timeout and best-effort behavior. Current run
reports determine current end-to-end cost; warm caches do not remove HF loading,
optimizer setup or CUDA graph capture.

The separate `python -m scripts.miles.compiler_cache_run` command is an experimental
single-node wrapper for controlled probes. Its broader TileLang/Inductor/FA4/
DeepEP/DeepGEMM families are not qualified Ray persistence. See the
[historical probe procedure](measurements/implementation-history/compiler-cache-before-sharing-20260913.md#coldrestored-screen)
for its explicit source/fingerprint contract. In the image, Open Instruct is at
`/opt/core-rl/open_instruct`, scripts at `/opt/core-rl/scripts/miles`, and the lock
at `/opt/core-rl/runtime/miles/runtime.lock.json`. Do not wrap an existing remote
Ray cluster with that standalone lifecycle.
