# Compiler-cache lifecycle

Ordinary MILES/Core runs persist **Triton artifacts only**. Each worker compiles
into a private local directory; verified, immutable generations of those
directories are restored from and published to shared WEKA storage. Persistence is
an optimization: a miss compiles locally, and a publication failure does not fail
completed training. CUDA graphs are captured per process and are not saved.

The implementation lives in the pinned MILES fork under
`miles.utils.compiler_cache`. Open Instruct supplies the run configuration and
application source root, then calls the MILES lifecycle from its driver.

## Configuration

The cache is enabled by default. Structured run files use the `[compiler_cache]`
section; low-level files use the matching `core.compiler_cache*` fields.

```toml
[compiler_cache]
enabled = true                       # core.compiler_cache; false opts out
# shared_root = "/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d"
restore = true                       # false gives a cold start but still publishes
diagnostics = false                  # opt-in Triton disk-cache counters
max_storage_bytes = 8_589_934_592    # 8 GiB per cache key; 0 disables uploads
publish_interval_seconds = 600       # minimum time between publication checks
```

The default shared root is
`/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d`. A custom root
must be absolute and, on WEKA, contain an exact `tmp-N[hdwmy]` expiry component
(the WEKA TTL naming convention, for example `tmp-7d`); this is validated before
launch. Expired artifacts are simply cold misses, so no cleanup job is needed.
Without the default WEKA mount, persistence is skipped and logged. Limits and
intervals are operational and do not change the cache key.

## What is cached

Each trainer rank and TP1 serving engine restores into a new `/tmp/core-triton-*`
directory before actor imports; SGLang's spawned children inherit their engine's
cache. Serving engines with TP greater than one, driver warmups/preflights and
other compiler families (FA4/CuTe DSL, TileLang, TorchInductor, FlashInfer,
DeepEP/DeepGEMM) are not persisted by the training lifecycle.

Keys include runtime/source identity, relevant model architecture and compile
settings, the logical worker slot, the actual device/toolchain and
compile-affecting environment. Settings that affect only orchestration, I/O or
diagnostics are excluded; unknown settings stay in the key. A different source
tree, image, architecture or compile setting therefore misses, and a new runtime
image should not expect older caches to be reused. Triton still checks each
kernel's own key when loading from the restored directory.

## Publication

After the first completed training collection in each process, the driver
publishes in the background. Later completed collections trigger a check only
after `publish_interval_seconds` has elapsed; this is wall time, not a collection
count, and does not depend on checkpoint saving or HF export. Workers whose files
are unchanged since the last publication skip the upload. After successful
shutdown, a final publication captures remaining artifacts and removes the
private copies. Failed training performs no final publication, but generations
published earlier remain reusable.

Publication is best-effort. Archives are built on local disk and only the finished
archive and manifest are uploaded before atomically advancing `CURRENT`; conflicting
bytes are never silently overwritten. Rounds run at most two tasks per node within
one shared 240-second wait budget, and expired tasks are cancelled. Hashing and
compression still use CPU, local disk and shared-storage bandwidth. Restore
validates identity, archive and file checksums before installing; corruption is a
recorded rejection and cold miss.

### Storage bound

`max_storage_bytes` limits each key directory, `SHARED/core-v1/FINGERPRINT/`,
counting every retained generation, manifest and pointer across runs. It is not an
aggregate limit for an experiment or the shared root, since different keys have
separate limits. When the next generation would exceed the limit, that worker
reports `storage_limit` and stops publishing for the rest of the run; existing
generations are left intact and restores continue. Lowering the limit does not
delete stored data, and the cap does not replace expiry through the root's
`tmp-N` component.

## Reading the results

Startup logs one summary per worker with its fingerprint, restore status and
reason, and restored file count. A miss reports `no_published_generation`; a
corrupt or incompatible archive reports `rejected` with the reason. This is an
archive-level result, not a kernel hit rate, and an archive hit alone does not show
that a restart avoided all compilation.

`compiler-cache.json` in the run's metric directory records each worker's cache
outcome, and `compiler-cache-progress.json` records publication rounds. Per-round
logs report lock wait, merge, compression, upload and pointer phases with file
counts, bytes and timings. Startup timings are in `driver_timing.jsonl` and
`startup_rankN.jsonl`; nested intervals overlap and must not be summed.

With `diagnostics = true`, a hook on Triton's `FileCacheManager.get_group` and
`put` appends events to local `activity-PID.jsonl` files, which publication
aggregates into `group_hit`, `group_miss` and `put` counts. These count disk-cache
operations, not compilations or GPU launches. The hook writes one local file
append per operation, so leave it off for ordinary runs.

The separate `python -m miles.utils.compiler_cache.run` command is an
experimental single-node wrapper for controlled probes of additional compiler
families. Do not wrap an existing remote Ray cluster with it.
