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
diagnostics = false                  # opt-in cache/compilation diagnostics
tilelang = false                     # experimental TileLang persistence
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
other compiler families (FA4/CuTe DSL, TorchInductor, FlashInfer,
DeepEP/DeepGEMM) are not persisted by the training lifecycle.

Triton uses a `core-triton-v3` compatibility envelope. It includes the binary
base, actual GPU/toolchain, installed Triton compiler source/binaries, native
source/extensions in the application and runtime packages, model configuration,
parallel geometry and compile-affecting settings/environment. Trainer ranks
remain separate; equivalent TP1 serving slots share a role key.

Full application/MILES source hashes and the runtime lock remain in worker
receipts as provenance, but a Python-only orchestration edit no longer discards
all kernel artifacts. Triton's own kernel keys distinguish source/dependency,
signature, specialization and compiler-option changes. Known non-kernel settings
including collection batch size, policy lag, admission, prepared-batch overlap,
layer profiling and truncated-response reward gating do not change the envelope;
unknown settings remain conservative. This compatibility policy applies only to
Triton, not to other compiler families.

Legacy v2 archives lack the new compiler/native identity evidence and are not
automatically adopted. The first v3 run can be cold; later compatible Python
revisions reuse its artifacts. Rebuilding an application image alone does not
invalidate the cache: the pinned binary base and compatibility inputs matter.

## Experimental TileLang persistence

`compiler_cache.tilelang = true` additionally persists TileLang in a separate
`core-tilelang-v1` envelope. This is opt-in pending GPU qualification. The pinned
FLA KDA backward path uses TileLang, so a Triton hit alone cannot eliminate that
compilation. The key includes the installed TileLang Python compiler, bundled
TVM and native libraries as well as the device/toolchain and compile settings.
The native-library stamp is forced on for this mode. TileLang's own TIR/target
keys select each kernel; dataset identity is not a compiler input.

Each worker gets a private TileLang directory and separate compiler scratch.
Publication rejects active staging exports and snapshots that change during
copying; it never publishes a partially exported kernel. Restore verifies the
archive before loading executable artifacts. Each TileLang key has its own
storage cap. Receipts include a separate `tilelang` restore/publication result;
`diagnostics = true` also enables TileLang's native JIT phase logs.

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
bytes are never overwritten. In the v3 Triton scope, a collision rejects the
entire incoming kernel group while preserving the published group; unrelated
new groups can still be saved. Receipts list `rejected_incoming_groups`. Rounds run at most two tasks per node within
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
operations, not compilations or GPU launches. A supported Triton compilation
listener also records `compile_hit` / `compile_miss`, kernel name, compiler
microseconds and lowering-stage timings. CUDA graph capture, module loading and
other compiler families are outside those measurements. The hook writes one local file
append per operation, so leave it off for ordinary runs.

The separate `python -m miles.utils.compiler_cache.run` command is an
experimental single-node wrapper for controlled probes of additional compiler
families. Do not wrap an existing remote Ray cluster with it.

With diagnostics enabled, worker receipts also contain
`compatibility_explanation`: changed identity field names, the previous observed
fingerprint, and whether source provenance changed. It compares one bounded
previous-startup record per model/worker; it does not select a cache or imply that
the previous run published successfully. A same-key miss can still mean no
published generation, expiry or rejected publication; the restore/publication
receipts distinguish these cases. Diagnostics failures do not block restoration.

The runtime's `tools/probe_triton_cache.py` runs a bounded single-GPU cold/warm
probe in fresh processes, including numerical checks, changed kernel/dependency
and shape cases, KDA training kernels, SGLang expert inference and experimental
FA4 persistence. It does not download a model or qualify distributed training.
