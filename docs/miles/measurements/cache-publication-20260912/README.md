# Compiler-cache publication: local proxy, 2026-09-12

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Packing was merged first into Open Instruct `robertb/miles-olmo-core` at
`ebebad29e` and Core `robertb/miles-rl-adapter` at `3d35ab326`. This cache-only
follow-up starts from that Open Instruct merge, on `robertb/miles-cache-publication`.

## What the timeout established

The packing exercise had three sequential 120-second publication timeouts, adding
six minutes to shutdown. The old report had empty timeout reasons and no phase
information: it does **not** establish whether workers were waiting to start,
waiting on locks, or doing filesystem work. A local Ray proxy successfully ran
that original path, so there is no universal dispatch failure.

The olmo-miles implementation publishes complete nonempty cache families, not a
size-ranked subset. It uses Zstd level 3 and node-local shared caches, with
node-level publication rather than a new Ray task per logical worker. Its
`docs/measurements/backend-parity-post-training-study.md` records 190.97 seconds
publishing Triton (41,442,908 compressed bytes), 206.94 seconds across families.
Therefore a 120-second per-task deadline was not supported by that history.
Both implementations previously staged extracted cache trees on shared storage.
Our production worker hook currently publishes only Triton.

## Local disk proxy

Read-only source: `/home/robert/.triton/cache` (about 1.8 GiB). Copied consecutive
whole kernel directories, stopping at 256 MiB, into a private `/tmp` fixture.
Rewrote only Triton's documented group paths to the copied fixture. The sample
contained **5,386 files, 269,405,679 bytes, 662 groups**. Canonical publication
inventory has 269,234,221 bytes because absolute group paths become relative.
No source cache was modified. These are SSD measurements, not WEKA estimates;
OS caches were not flushed and this is a single bounded proxy exercise.

| Operation | Before | After |
|---|---:|---:|
| First publication | 3.56 s | 3.66 s |
| Unchanged-generation publication | 4.70 s | 4.41 s |
| Verified restore | 1.12 s | 1.22 s |

Both implementations produced the identical generation digest and restored all
662 groups. The change relocates work off the shared filesystem; it does not
claim to accelerate a local SSD. After the change, local merge/hash took 2.32 s,
archive preparation 1.32 s, and copying the two bulk files 0.025 s.

Independent archive comparison on the same sample: gzip level 1 took **1.29 s /
52.27 MB**; Zstd level 3 with four threads took **0.33 s / 36.09 MB**. Zstd is
better, but saves about one second on this fixture. We retain the existing gzip
format for this bounded reliability fix rather than introduce a format/dependency
migration. Compression alone does not explain the historical multi-minute cost.
Raw results: `local-before.json`, `local-after.json`.

## Reliability change and validation

- Extract, snapshot, merge, relocate, hash and compress on node-local disk.
  Upload only the finished archive and manifest; retain the immutable generation
  and atomic `CURRENT` protocol. No arbitrary artifact pruning.
- At most two publishers per node. One shared **240-second publication wait**
  across all nodes/ranks, including queueing. Explicit force cancellation in Ray
  on deadline, with automatic retries disabled. Report metadata I/O is separate
  from this wait budget.
- Log submission/start and publication phases; record file/byte counts and phase
  timings. Timeout errors now explain the deadline. Cache/report errors remain
  optional and cannot convert completed training into a failure.

**52 targeted tests passed** in the pinned runtime, including concurrent merge,
conflicting keys, corrupt/unsafe archives, group relocation, interrupted upload,
old-generation preservation, per-node limits, mixed outcomes and deadline
cancellation. `make style`, `make quality`, and targeted test-file Ruff passed.

Real Ray local-container exercises used `open-instruct-miles-core-a318a75a886c`
(the packing runtime), source mounted read-only, two CPUs and no GPU:

- Three successful publications: 0.211 s before, 0.192 s after. Tiny fixtures;
  this validates dispatch and collection, not real-cache throughput.
- Held publication locks while three workers attempted to publish new artifacts.
  With the common budget reduced to two seconds, completion returned in **2.005 s**.
  Two workers started and blocked on the locks, the third stayed queued. All
  three reported timeout; successful training remained successful. Releasing
  the locks and waiting another two seconds left all previous `CURRENT`
  generations unchanged: the cancelled workers did not resume publishing.

Raw Ray completion records are `ray-before.json`, `ray-after.json`, and
`ray-timeout.json`. No new Beaker allocation was needed. The next ordinary run
must establish the actual WEKA publication time and cache-hit outcome. We have
not attributed the original six-minute timeout to a specific phase.
