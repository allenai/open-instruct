# Native checkpoint performance qualification

## Latest result: full-model candidate passed

At 20:03 UTC, [the fast save/read qualification](https://beaker.org/ex/01M28YRXD7NGGTHWBZ2RY3R1XX)
finished successfully with **zero correctness failures** and both acceptance gates passed.
[Retained final audit](miles-checkpoint-fast-audit-20260911.json).

- Full-model save: **118.466 seconds**, versus baseline **437.128 seconds** (3.69x faster).
- Fresh-process load: **144.307 seconds**.
- Live state before/after save and restored boundary state: exact on both ranks.
- Scoring log-probabilities for both subsequent updates: exact on both ranks.
- Final model, optimizer, history, scheduler, RNG, clock, and data cursor: exact.
- HF exports: exact across resume.

This qualifies the arithmetic save/read candidate with compact storage and balanced
replicated ownership on the tested two-B300 EP2 full-model configuration. It used fixed
synthetic responses and a persistent Triton cache per global rank. It does not qualify
SGLang transport/decode, every pretraining topology, or the process writer. The original
baseline with shared rank caches had exact restored state but divergent continuation;
the successful controlled-cache run supports execution reproducibility as the distinction,
without isolating every possible cause.

The validated arithmetic planner, compact storage, and balanced ownership are now the
defaults in the MILES/Core MoE RL wrapper, with independent opt-outs documented in
[run controls](../miles-run-controls.md). Profiling and process workers remain opt-in.
Shared Core/pretraining defaults and production Ray cache handling remain unchanged.
The additional same-node metadata and process-writer arms are still running.
Earlier pending statements below describe intermediate measurements, superseded by this result.

Isolated branches: `robertb/miles-checkpoint-perf` in open-instruct and OLMo-core.
Bases: open-instruct `12235d54d`, Core `307d20590`.
The implementation is integrated into `robertb/miles-olmo-core` (open-instruct)
and `robertb/miles-rl-adapter` (Core), with all performance options disabled by default.

The initial trial preserves production defaults. Candidate save options are explicit harness
arguments, not new run-profile defaults:

| Mode | CPU storage | Replicated owner | Writer |
|---|---|---|---|
| baseline | unconditional clone | lowest rank | existing thread default |
| compact | clone only noncompact backing storage | lowest rank | existing thread default |
| balanced | compact storage | PyTorch planner balances | existing thread default |
| processes | compact storage | PyTorch planner balances | 2 spawned workers, 2 buckets |

Contiguity is insufficient to skip cloning: storage offset must be zero and backing storage
bytes must equal logical tensor bytes. A contiguous prefix otherwise serializes unrelated
storage. The shared writer defaults remain unchanged.

Profile fields distinguish state-dict construction, persistent buffers, local/global planning,
DCP total, writer wall time, coordinator metadata, optimizer reload and cache clearing. Per-item
resolve, blocking CPU copy, clone, serialization, flush and rename/upload durations are **summed
worker times**, which can exceed writer wall time. They do not by themselves establish GIL
contention. GPU boundaries synchronize only when profiling is requested.

The direct save now restores optimizer storage in a `finally` block if buffer preparation or
DCP writing fails. Construction of the optimizer state dict itself remains the optimizer's
responsibility. Asynchronous saves are deferred: the current state-dict method releases some
live storage and returns views, which cannot safely be mutated by subsequent training.

## Checks completed locally

- CPU writer/reshard and compact-storage tests: 11 passed, 4 GPU skips.
- Actual toy MoE on RTX 4090: both balanced/compact threads and spawned processes passed native
  save, live-state retention, HF-file content verification, fresh-process restore, and exact
  next-two-update model/optimizer state and log-probabilities.
- Toy writer saves: approximately 0.07 seconds with threads and 9.31 seconds with processes;
  process startup dominates a 9 MB checkpoint. These are correctness checks, not evidence about
  full-model performance.
- Open-instruct `make style quality` passed using the existing environment and Core source path.
- Changed Core files pass isort, Black and Ruff. Repository-wide `make checks` stops on existing
  import-order failures in unrelated hero example/attention/objective files.

## Full-model gate

`build_image_and_launch.sh --miles scripts/train/debug/miles_checkpoint_benchmark.sh`
launches two B300 GPUs on Holmes, urgent, open-instruct-dev, minimum runtime 60 minutes, maximum
3 hours. Native checkpoints and HF exports remain under a fresh WEKA experiment directory;
small reports are copied to Beaker results. Existing topology tests run first, including
EP1-to-EP2 save/load and both writer policies (EP4/8 require larger allocations).

Each arm trains two fixed batches, saves and exports, then continues for two more batches in
the **same process**. A fresh process loads that exact checkpoint, exports, and consumes the
same next batches. Compare all local master/moment/step bytes, model bytes, scheduler, clock,
data cursor, RNG, and scoring log-probabilities. The two exported HF artifacts must also match.
This avoids comparing two independently initialized/autotuned training trajectories.

Acceptance requires exact correctness plus max rank save time <= 340 seconds on the full SFT
model. Synthetic responses isolate the checkpoint contract; serving and GSM8K learning are
outside this experiment. Enabling a candidate by default requires the full-model gate. The implementation can be
merged while opt-in so other work can exercise it without changing existing run profiles.


## Follow-up qualification

The first Beaker attempt, `01M28SJFTT0H8K9WX3N270DE83` (image
`01M28SJ1XWPY6HZJSZEA2WBBQ1`), passed GPU attention preflight and seven existing Core tests,
but its two EP1-to-EP2 fixtures selected `rowwise_nvshmem`, whose optional extension is absent
from this MILES image. Those cases failed before the EP2 restore, so no full-model saves ran.
The checkpoint fixture now explicitly selects `sync_1d`, the MILES adapter's EP path; the
rounded-gradient fixture's default remains unchanged.

A separate opt-in `constant_memory_planning` candidate computes contiguous chunk metadata
arithmetically. It falls back to PyTorch for strided, partial, symbolic or custom local tensors.
Metadata matches PyTorch over more than 1,000 uneven/empty/repeated-shard layouts. The expanded
CPU save/reshard suite passed 18 tests (six multi-GPU skips). A real toy CUDA MoE run also passed
exact save/export/resume, including optimizer rolling histories, using this planner.

Follow-up modes are baseline, balanced/compact, constant-memory metadata, and that same metadata
mode with two writer processes. Each retains an independent checkpoint and two HF exports;
full-model storage is approximately 300 GB per mode. None changes the production defaults.


## Run configuration interface

The normal `[core]` TOML interface now forwards these options to the MoE native save:

```toml
[core]
checkpoint_profile = true
checkpoint_compact_storage = true
checkpoint_dedup_save_to_lowest_rank = false
checkpoint_constant_memory_planning = true
# Optional independent worker controls:
# checkpoint_thread_count = 2
# checkpoint_process_count = 2
```

Defaults preserve the legacy policy. Profiling also writes `save_metrics_rank_N.json` next to
checkpoint manifests. These options do not change model geometry or resume compatibility.
The standard dense trainer explicitly rejects overrides until its separate path is qualified.
The real toy training/save/export/resume harness passed through this configuration interface;
68 option, topology, model-dispatch and audit tests also passed.

Corrected full-model experiment: `01M28TFNGP78F24E6YNTTY4XMJ`, image
`01M28TF7SDRPAFKQAH3EQ0E2Z9`, source `71d5cd206`, Core `f628ec416`. It started at 18:27 UTC after waiting for allocation slots. The corrected GPU topology gate
passed 10 tests, with six larger-topology skips and three deselections. Its image predates
the optional CLI forwarding, but exercises the same native save options directly.
The first two full-model optimizer updates completed at 18:38 UTC. Live stack samples then
observed PyTorch shard-offset planning followed by writer threads; phase timings are pending.
No completed full-model performance or resume claim yet.


## Existing checkpoint evidence (Saturn CPU probe)

`01M28V8PAWD0BBX90136JKZGSZ` successfully read the completed controls checkpoint metadata.
Raw structured evidence: [metadata profile](miles-checkpoint-metadata-20260911.json).

- Rank 0: 111,089,192,982 bytes, 2,014 entries, 16 files.
- Rank 1: 111,086,027,103 bytes, 1,095 entries, 16 files.
- Total: 222,175,220,085 bytes. Rank byte ratio is 1.0000285; byte imbalance is not a
  plausible large bottleneck for this checkpoint. Rank 0 owns more small entries.
- The largest shape (623,902,720 elements) occurs 57 times. Ordinary two-way metadata
  planning took 1.168 s, and a repeated-shard `(2,1)` mesh took 3.923 s, versus 4–5 us
  for the arithmetic candidate. Returned sizes and offsets matched exactly.
- The next large shape (311,951,360 elements), also repeated 57 times, took 0.550 s
  or 2.177 s, versus 3–5 us.

These are CPU probes of actual saved shapes, not the full GPU job's phase timings. DCP does
not persist the original device mesh, so both canonical contiguous layouts were probed;
do not add these durations and label the sum an observed production save time.


## First full-model save profile

The baseline save completed at 18:48 UTC.
[Raw per-rank timers](miles-checkpoint-baseline-profile-20260911.json).

| Phase | Rank 0 seconds | Rank 1 seconds |
|---|---:|---:|
| Total direct save | 437.10 | 437.13 |
| Local metadata plan | 307.19 | 320.65 |
| Writer wall | 115.68 | 110.18 |
| Optimizer reload | 0.100 | 0.099 |
| State dict construction | 0.016 | 0.014 |
| Coordinator metadata write | 0.044 | — |

The ranks synchronize between planning and writing, so the slowest planning rank and
slowest writer determine the critical path. They wrote the same 222.18 GB total as
the prior controls checkpoint. Aggregate throughput is about 0.51 GB/s including
planning, or 1.92 GB/s for the writer phase alone. This run does not reproduce the
reported 1,000-second save, but directly identifies metadata planning as its main
checkpoint cost. Export, fingerprinting, and fresh-process initialization are excluded
from this direct-save timer. Exact resume is still pending.


## Read-side follow-up

Baseline resume stacks also observed the expensive PyTorch shard-offset calculation.
The opt-in metadata control now selects an arithmetic read planner as well. It keeps
PyTorch's chunk-overlap/resharding algorithm, strict shape checks, and missing-key/legacy
key migration fallback. Original and candidate readers both consume either save policy
in the CPU compatibility tests (44 passed, six GPU skips). A local CUDA toy run passed
exact fresh-process save/export/resume and next-update log probabilities with this path.
`checkpoint_profile` also logs total native load and individual DCP load-pass durations.
The first full-model experiment uses the original reader throughout; a separate full-model
run is required before promoting the read-side optimization.


## Baseline continuation finding

The first full-model baseline restored every recorded model/optimizer/history/RNG/cursor
byte exactly on both ranks, and both HF exports matched. The strict continuation gate
failed: rank 0's pre-update-3 scoring differed, rank 1's matched, and subsequent gradients
changed both ranks' final states. This is evidence of execution reproducibility drift
with identical restored state, not missing checkpoint tensors.

Both ranks shared a Triton autotuning cache. A rank-specific persistent cache is the next
hypothesis test; it does not relax the exactness gate. The first faster-reader trial
`01M28Y59M19RG252HTEY4BZ16D` was stopped during startup so it can be replaced with this
controlled setup. The original four-arm job continues to measure save phases.


## Compact/balanced writer result

At 19:29 UTC this arm saved in 448.22 seconds (max rank), with local planning
329.97 seconds and writing 117.51 seconds. Worker clone time fell from roughly
42 seconds summed per rank to less than 0.02 seconds. Overall save time did not
improve; eliminating these copies and changing replicated ownership did not
remove the measured dominant cost. [Raw timings](miles-checkpoint-balanced-profile-20260911.json).

[Baseline continuation details](miles-checkpoint-continuation-20260911.json) distinguish
exact restored state/export from divergent post-restart computation.

Replacement full-model read/write qualification: `01M28YRXD7NGGTHWBZ2RY3R1XX`,
image `01M28YRP8WWWJGYTQ1JYRXSTVR`, OI `63c8cae42`, Core `48bb6d7e1`.
It passed the updated GPU topology gate (10 passed, six larger-topology skips,
three deselections) and uses separate persistent caches for each global trainer rank.
The local cold-cache toy qualification also passed with all 18 cache fingerprints
unchanged across restart. Full-model exactness is pending.


## Integration status and arithmetic planner measurement

The implementation is available for opt-in integration; it is **not fully qualified for
default enablement**. Existing production and pretraining defaults remain unchanged.
The normal Core pretraining CheckpointerConfig does not yet expose the planner control.

The full-model arithmetic save in `01M28YRXD7NGGTHWBZ2RY3R1XX` took **118.466 seconds**
(max rank), versus baseline **437.128 seconds**. Local planning fell from **320.655 seconds**
to **0.05654 seconds**; writer wall time remained **116.518 seconds**. The 222.18 GB
checkpoint therefore meets the <=340-second save target.
[Raw candidate save timings](miles-checkpoint-fast-profile-20260911.json).

Its fresh-process native load completed in **144.307 seconds**, with 144.026 seconds
in the DCP load pass. The strict subsequent-update comparison is still running.
The baseline restored all captured state and HF export tensors exactly, but failed
post-restart numerical continuation; that distinction remains an open qualification item.
Per-rank persistent autotuning caches are enabled in this qualification launcher only.
The ordinary production Ray launch path has not gained that cache policy.

Local checks after incorporating the current parent branch: 66 selected integration tests
passed and open-instruct `make style quality` passed. Core metadata, writer compatibility,
failure restoration, and toy exact continuation checks passed as recorded above.
Larger GPU topologies and the full-model process writer are still pending.

To exercise the planner through an open-instruct run config, set
`core.checkpoint_constant_memory_planning = true` and optionally
`core.checkpoint_profile = true`. Writer compaction, ownership, and process settings
remain separate options; they are not required to select arithmetic metadata planning.


Post-integration verification: **97 tests passed**, with 14 deselected (GPU-marked
and remote-storage cases), covering wrapper options, continuation auditing, topology
contracts, metadata parity, local writer/reader compatibility, and save-failure recovery.
Open-instruct formatting, lint, compilation, and type checks passed on the working branch.
The Core runtime commit and overlay checksum match the merged Core working branch.
The candidate full-model live state is byte-exact before and after save on both ranks;
the fresh-process subsequent-update audit remains pending.
