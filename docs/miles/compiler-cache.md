# Compiler-cache lifecycle

Ordinary MILES/Core runs persist **Triton artifacts only**, through the Ray worker
lifecycle below. Mutable caches remain private on local disk; immutable verified
generations are restored from and published to WEKA. Persistence is an optimization:
a miss compiles locally, and optional publication failure does not fail completed
training. CUDA graphs are captured per process and are not saved this way.

```toml
[compiler_cache]
enabled = true
# Defaults; limits are operational and do not change the compiler-cache key.
max_storage_bytes = 8_589_934_592  # 8 GiB per key, across runs and families
publish_interval_seconds = 600  # minimum time between progress checks
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
Triton disk-cache group reads and artifact writes; leave this instrumentation off
for ordinary runs. See [diagnostic cost](#diagnostic-cost-and-meaning) below.

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

After the first completed training collection in each process, the driver
schedules background publication. Later completed collections can trigger a check
only after `publish_interval_seconds` has elapsed (default 600 seconds). This is
elapsed time, **not** a request count or a ten-collection cadence. Checks compare
local file paths, sizes and modification/change times with the last successful
publication. Unchanged workers skip shared-cache extraction, hashing, compression
and upload. Changed workers still pass the full content/integrity checks.

This does **not** depend on model checkpoint saving, the save interval, or HF
export. The clock starts fresh on resume; a collection can contain multiple
optimizer updates. At most one publication round runs at a time. No timer wakes
an otherwise idle driver; checks happen at completed-training boundaries. A final
successful cleanup attempts a flush regardless of the interval, except for workers
already stopped by the storage cap.

These small Ray tasks run on the workers' original nodes and keep each live
worker's mutable cache in place. If compilation changes a snapshot while it is
being copied, validation rejects that snapshot; later rounds can retry. No
training or inference barrier is added. Hashing/compression still consumes CPU,
local disk and shared-storage bandwidth, so background does not mean free.
`compiler-cache-progress.json` records rounds with `collections_seen` and
`rollout_id` (older reports used `checkpoints_seen`).

After successful driver cleanup (including shutdown of serving children), a
final publication also captures remaining artifacts and cleans private copies. Publication runs at most two tasks per node, with one shared
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

Failed training does not perform a final publication; generations published
after earlier completed collections remain reusable. Cache miss/rejection
compiles locally; publication failures are recorded, not treated as RL failures.
`compiler-cache.json` in the run's metric directory records each worker's cache
outcome and diagnostic counters. Raw worker identity/restore records remain in
the shared cache's `runs/` directory.

### Storage bound and recency

`compiler_cache.max_storage_bytes` defaults to **8 GiB per fingerprint directory**:
`SHARED/core-v1/FINGERPRINT/`. It counts logical file bytes already stored across
all families, including **every retained generation**, manifests, pointers and
interrupted-upload files. It is not reset for a new run. Concurrent publishers
using this implementation take the same per-key lock before counting and writing,
so different runs/families cannot each spend a fresh allowance. Different keys
have separate limits; worker slots are currently part of the key, so this is **not
an 8 GiB aggregate limit for the experiment or shared root**.

The publisher builds the candidate archive on local disk, then checks whether the
archive, manifest and temporary pointer would fit before uploading. An already
full key or an oversized next generation returns `storage_limit` with current,
required and allowed bytes. That worker stops publishing for the rest of the run,
including final flush. Later runs check the existing directory again against the
same cap; changing the limit does not change the fingerprint. Existing generations
and `CURRENT` are left intact. Setting the limit to zero disables uploads while
retaining restores. A lower limit does not delete already-stored data.

This bounds persisted file contents, not filesystem block/metadata overhead,
node-local compilation directories, or the small per-run reports outside the key
namespace. Seven-day retention remains the default through `tmp-7d`; a cap does
not replace expiry. Older publisher binaries do not enforce this new limit.

Successful restores and publications touch one zero-byte `.last_used` marker in
the selected generation directory. This records recency once per use of the
archive, with no per-kernel activity tracking or mutation of the checksummed
archive/manifest. Failure to record recency does not prevent a hit. The marker
supports future coarse LRU cleanup; **no eviction is enabled here**. Within a key,
new generations contain earlier kernels, so pruning superseded snapshots is
simpler than kernel-level LRU. LRU between different keys would need a separate
shared-root budget. The requested behavior at the current cap is to stop writing.

Startup timings are now retained separately: `driver_timing.jsonl` includes cache
preparation, placement, serving startup and trainer startup; `startup_rankN.jsonl`
includes distributed initialization, HF reads, native parameter initialization,
HF conversion, overall native model/optimizer construction and native restore.
Nested intervals overlap and must not be summed. Existing evaluation, publication,
scoring and training timings complete the timeline.


## Identity and archive contract

Keys include runtime/source identity, relevant model architecture and compile
settings, actual device/toolchain and compile-affecting environment hashes.
Operational exclusions are catalogued below; unknown settings remain in the key. Different
source, image, architecture or compile settings can intentionally miss; a new
runtime image must not assume every older cache is reusable.


### Operational inputs excluded from the key

These settings affect orchestration, I/O or diagnostics, not compiled kernel
compatibility. The explicit exclusions live in `compiler_cache.py`; this is a
conservative denylist, so adding a new option cannot silently bypass invalidation.
Input batches may still introduce new specializations; Triton checks each kernel's
own key when loading it from the restored directory.

| Group | Excluded inputs |
|---|---|
| Cache lifecycle | `compiler_cache`, `compiler_cache_root`, `compiler_cache_restore`, `compiler_cache_diagnostics`, `compiler_cache_max_storage_bytes`, `compiler_cache_publish_interval_seconds` |
| Checkpoint storage | `checkpoint_profile`, `checkpoint_thread_count`, `checkpoint_process_count`, `checkpoint_compact_storage`, `checkpoint_dedup_save_to_lowest_rank`, `checkpoint_constant_memory_planning`, `checkpoint_keep_last`, `checkpoint_keep_every` |
| Observation and checks | `diagnostic_interval`, `pipeline_observation_interval`, `replay_diagnostics`, `max_train_rollout_logprob_abs_diff`, `scoring_check_interval`, `scoring_check_tolerance` |
| Deadlines | `engine_drain_timeout`, `engine_update_timeout`, `refresh_request_timeout`; MILES `sglang_watchdog_timeout`, `sglang_dist_timeout` |
| Records and prompt selection | `records_root`, `records_responses`, `records_response_sample_rate`, `selection_table`, `selection_sha256`, `reward_config` |
| Serving endpoints and storage | `sglang_host`, `sglang_port`, `sglang_download_dir`, `sglang_file_storage_path` |
| Serving observability | `sglang_log_level`, `sglang_log_level_http`, `sglang_log_requests`, `sglang_log_requests_level`, `sglang_log_requests_format`, `sglang_show_time_cost`, `sglang_enable_metrics`, `sglang_enable_metrics_for_all_schedulers`, `sglang_collect_traces`, `sglang_otlp_traces_endpoint` |
| Existing MILES exclusions | `hf_checkpoint`, `save`, `load`, `prompt_data`, `eval_prompt_data`, `save_debug_rollout_data`, `rollout_sample_rate`, `save_debug_train_data`, `wandb_project`, `wandb_group`, `wandb_entity`, `wandb_run_name` |
| Environment paths | Family cache-directory variables, `CUDA_VISIBLE_DEVICES` (actual hardware is probed), `SGLANG_DG_CACHE_DIR`, `TILELANG_TMP_DIR` |

The dynamic `SGLANG_DG_CACHE_DIR` previously made an otherwise identical serving
restart eligible for a false miss. Its per-process **behavior flag** remains in
the key; only the directory location is excluded.

EP degree, attention backend, precision, compilation flags, packing/sequence
geometry, row specialization, scoring-path override, router-loss settings and
unknown Core/MILES options still invalidate. Archive verification, per-worker
isolation and private mutable restores are unchanged. These changes do not
search for or adopt artifacts stored under an older, different key.

### Remaining overly broad inputs

The following are catalogued opportunities, **not removed in this change**:

- Whole source-tree hashes include unrelated Python and native files in Open
  Instruct, MILES, Core and olmo-sglang. A reporting-only Python edit can still
  invalidate every worker. Narrowing this needs an audited kernel/import
  dependency boundary; keeping dirty kernel changes detectable is essential.
- The complete runtime lock includes descriptive status, repository/private
  metadata and old patch provenance as well as actual revisions. A normalized
  compatibility identity could separate those from source provenance.
- Trainer keys include serving concurrency, engine counts, graph sizes and KV
  capacity; serving keys also include trainer configuration. Split identities
  by role before claiming independent trainer/serving reuse.
- Logical worker names are part of the key. The migration renamed
  `train-actor-cell0-rank0` to `trainer-actor-0-0`; the label change alone misses.
  Stable role/cell/rank identities could preserve rank isolation across renames.
- Hardware identity includes the list of visible devices and repeated driver
  output. A serving wrapper seeing a different number of otherwise identical
  GPUs can miss, even when its scheduler uses one GPU. Probe the actual scheduler
  rank/device before normalizing this.
- Model config is hashed in full, including any provenance-only fields. Separate
  architecture/precision inputs from metadata only after auditing each supported
  model's configuration consumers.

Generations contain checksummed manifests and `cache.tar.gz`. Restore validates
identity, archive and individual files in staging before atomic installation;
corruption is a recorded rejection/cold miss. Triton group metadata is relocated
to the new private directory. Publication merges compatible entries under a lock
and atomically advances `CURRENT`; conflicting bytes are not silently overwritten.

Olmo-miles uses a separate namespace and Zstandard archives. Those are not
interchangeable with this gzip format. Local staging addressed the shared-file
I/O problem without a codec migration. Inspect measured lock/merge/compression/
upload phases before changing timeouts or attributing delay to gzip.

## Diagnostic cost and meaning

Default startup logging emits one summary per worker containing fingerprint,
restore status/reason, restored file count and whether detailed diagnostics are
enabled. A miss now says `no_published_generation`; corrupt/incompatible archives
report `rejected` with the reason. This is an archive-level result, not a measured
kernel hit rate. Publication has additional phase and completion logs each round.

Detailed diagnostics are **not** a fixed 20–100 startup lines. The optional hook
wraps Triton's `FileCacheManager.get_group` and `put`. Each call opens a local
`activity-PID.jsonl`, appends one event, and closes it. SGLang scheduler children
install the same hook. It runs throughout the process whenever these disk-cache
operations occur, including later shapes/autotuning, but does not observe every
GPU launch or count in-memory JIT hits. Publication aggregates those files into
`group_hit`, `group_miss` and `put` counts; the individual events are not printed
to application stdout.

The retained September 11 cold trial recorded 138 group misses and 1,112 writes
on one worker: 1,250 local file appends, not 138 compilations or 1,250 log messages.
A compile can write several artifacts, and group lookup counts need not equal
kernel-compilation counts. No controlled overhead measurement is available for
the current full-model workload. Keep it opt-in for a cold/restored qualification.
A future low-cost default should aggregate counters in memory and publish bounded
summaries, measure its overhead, and preserve counts from spawned schedulers.

## Coverage audit

The following controls were checked against application image
`open-instruct-miles-core-2aec2d8b0616` on September 23. Coverage refers to the
ordinary Ray startup lifecycle; the standalone wrapper's larger `FAMILIES` table
does not enable those families in training.

| Family | Current behavior | Missing qualification/work |
|---|---|---|
| Triton, including kernels using Triton's standard cache manager | Private directory restored/published to WEKA; optional disk-group/write counters | Real GPU restart after these lifecycle/key changes; no per-compile timing yet |
| FA4 / CuTe DSL | Pinned FA4 defaults to in-memory caching. `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` enables disk persistence and `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` sets its directory; ordinary startup sets neither | Highest-priority trainer candidate: set environment before imports, qualify fresh-process export/load and relocatability, then include verified artifacts in Ray restore/publication |
| TileLang | Its own disk cache, default `~/.tilelang/cache`, controlled by `TILELANG_CACHE_DIR`; ordinary lifecycle does not manage it | Identify used kernels, qualify local-root relocation, concurrent snapshotting and restart reuse |
| TorchInductor | Separate cache controlled by `TORCHINDUCTOR_CACHE_DIR`; ordinary lifecycle does not manage it | Model/optimizer compilation are off in the current recipe; qualify only for a path that uses it |
| FlashInfer | Separate version/architecture cache under `FLASHINFER_WORKSPACE_BASE/.cache/flashinfer`, plus optional packaged/downloaded cubins | Distinguish prebuilt artifacts from actual JIT misses; not even listed among standalone managed families |
| DeepEP / DeepGEMM | Standalone wrapper lists `EP_JIT_CACHE_DIR` / `DG_JIT_CACHE_DIR`; SGLang also controls DeepGEMM's per-process directory | Establish which backend executes and its actual cache controls before persisting; presence in a table does not establish use |
| CUDA graphs | Captured for each running process | Not a portable compiler-cache artifact |

For each added family, the acceptance check is a cold run followed by the same
image/model/config in fresh processes, with correct outputs, demonstrated reuse,
compile/load timing, and rejection of incompatible or corrupt artifacts. Start
with FA4, then TileLang where observed in the selected trainer. A Triton archive
hit alone cannot establish that a restart avoided all compilation. No broader
family is enabled by this change, and no GPU speedup is claimed from CPU tests.

## What has been measured

The [full-SFT cold/restored screen](measurements/startup-full-sft-20260911.json)
observed restored Triton groups, zero new compiler writes and faster first updates
on all three workers. Its 174/238-second publication timings came from the old
WEKA staging implementation. The [publication fix](measurements/cache-publication-20260912/README.md)
qualifies local staging, shared timeout and best-effort behavior. Current run
reports determine current end-to-end cost. The [full-SFT runtime check](measurements/sharing-20260913/README.md)
published all three workers in 11.2 seconds total, with no timeout. Warm caches do not remove HF loading,
optimizer setup or CUDA graph capture.

The separate `python -m scripts.miles.compiler_cache_run` command is an experimental
single-node wrapper for controlled probes. Its broader TileLang/Inductor/FA4/
DeepEP/DeepGEMM families are not qualified Ray persistence. See the
[historical probe procedure](measurements/implementation-history/compiler-cache-before-sharing-20260913.md#coldrestored-screen)
for its explicit source/fingerprint contract. In the image, Open Instruct is at
`/opt/core-rl/open_instruct`, scripts at `/opt/core-rl/scripts/miles`, and the lock
at `/opt/core-rl/runtime/miles/runtime.lock.json`. Do not wrap an existing remote
Ray cluster with that standalone lifecycle.
