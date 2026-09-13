# Independent engine drain (experimental)

This isolated implementation keeps `core.publication_mode = "barrier"` as the
normal/default path. `"engine_drain"` selects rolling updates for resident,
disaggregated Core MoE with TP1 engines, one optimizer step per collection, and
blocking shared-engine evaluation. It does not implement mixed-policy continuation.
No shared operating example has been switched to it.

## Ownership and execution

The managed async producer owns one event-loop controller. It reserves every
sibling of a prompt group on one engine and version before sending any HTTP
request. This includes requests waiting for the client semaphore or queued at the
server. New groups bypass the load-balancing router and use their reserved engine's
URL. The actual response metadata must acknowledge the reserved version; it is not
read from the rollout manager's changing global clock. Decoding releases the engine
reservation before reward computation. Group identity remains owned through grading
and the existing data-source ledger retains it until consumption or deliberate drop.

After each optimizer step, all trainer ranks export in the existing collective order.
Rank zero takes owned BF16 copies into one GPU bucket, packs the established
flattened byte layout on the GPU, then copies that bucket to host memory and places
it in Ray's immutable object store. Capture temporarily needs the owned bucket
and its packed GPU buffer, rather than another complete GPU model. The exporter
retains no live parameter views. There is one full snapshot per version, shared by
all receivers. `snapshot_capacity` bounds versions retained; the driver waits for
capacity before another capture. Ray's own object-store spill policy still applies.
Provision host/object-store space for capacity times full exported model bytes,
plus capture staging and normal rollout data. The roughly 37 GB model therefore
needs roughly 74 GB of snapshot storage at capacity two.

Each engine has a separate delivery actor process on the source trainer node,
sharing its source GPU for a temporary bucket and a CUDA context. Each delivery
actor owns a two-rank NCCL communicator with exactly one TP1 receiver. Training
communicators and live parameters are never used by background delivery. A pinned
host staging bucket feeds the sender GPU. This preserves bucketed/fused transfer,
but adds host copies and per-engine CUDA contexts compared with the synchronous
GPU export. The first measured CPU-packing capture was too expensive (about 65 s);
GPU bucket packing replaces it, with full-model measurement tracked below. Initial capture is also delivered and
checked against the startup serving-weight audit before training admission opens.

Publication closes admission independently, waits for owned requests, loads the
snapshot, ends the weight-update session, flushes caches, and checks the engine's
version before reopening. Healthy peers can reopen while another is draining.
There is no normal-path pause, retract or abort request. Admission reserves one
optimizer step of lag headroom; completed groups still undergo the existing actual
consumption-time lag checks. Packing, replay, masks, TIS and reward semantics are
unchanged. Completed, graded groups use the existing single FIFO buffer, ordered
by insertion after completion. There is no per-version queue or newest-policy
priority. The consumer rejects expired entries and uses the oldest queued eligible
group; the qualification configs retry expired groups' pristine prompts. Groups
are internally homogeneous, but an optimizer batch may contain several eligible
policy versions. `snapshot_ready_step`, per-engine versions and fleet minimum are
separate from the consuming optimizer step.

## Boundaries and failures

`engine_drain_timeout` bounds drain, request and admission waits;
`engine_update_timeout` bounds update acknowledgement and transport setup.
Snapshot versions are delivered in order per engine. Partial updates and wrong
versions make the engine unavailable and propagate a terminal error to the driver.
This first implementation has **no automatic per-engine retry/recovery**; the run
fails and a fresh process must restore a committed checkpoint. Automatic fault
tolerance, external engines, custom generators/filters, serving TP/DP/EP/PP greater
than one, and snapshot-fleet evaluation are rejected.

Evaluation, checkpoint and final export stop producer submission, drain already-owned
work and join publication. To prevent a full completion queue from blocking that
join, its capacity expands only by the number of already-owned groups; it returns
to its configured capacity afterward. Completed data is retained, and new producer
submission waits until the excess completion queue is consumed. Repeated boundaries
therefore cannot accumulate another generation wave on every save. Checkpoints use
the existing atomic cursor/pristine-pending-prompt ledger and native model commit.
Resume regenerates outstanding prompts at restored weights; it does not restore
partial decodes, KV/KDA state, publisher processes, or in-memory completion queues.
Initial publication precedes all resumed admission. Shutdown joins publishers and
closes their communicators before engine disposal.

## Qualification and use

The separate configs under `configs/miles/qualification/engine-drain/` select six
updates, EP2 trainers, two TP1 engines, packing, replay, TIS and initial/final held-out
GSM8K evaluation. Save boundaries are at three/six updates. The barrier config is
the matched rollback/control. Run via the committed-image wrapper:

```bash
python -m open_instruct.miles plan configs/miles/qualification/engine-drain/ep2-gsm8k.toml
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  python -m open_instruct.miles run configs/miles/qualification/engine-drain/ep2-gsm8k.toml
```

The separate slow fixture sets `OI_MILES_ENGINE_DRAIN_TEST_DELAY_SECONDS=120`.
This qualification-only switch delays exactly one reserved request on engine 1
before its HTTP send, after that engine first receives trained weights. The hold
waits for actual admission closure before starting the 120-second delay, so cold
trainer compilation cannot consume the intended slow-drain interval. It tests
queued/unsent ownership after cold-start training; it does not simulate slow grading
or establish transport bandwidth. Values outside 0–120 are rejected. Default is zero.

`engine_drain.jsonl` records group/request ownership, actual versions, decode tokens,
grade completion, per-engine drain/update/reopen transitions, snapshot retention and
delivery time/bytes/GPU allocation. Existing training-contract and driver timers
provide optimizer/queue/evaluation/checkpoint evidence. Overlapping stages must not
be summed as elapsed wall time. The controller has no source of partial generated
tokens when HTTP fails; do not infer their number from unfinished groups.

Analyze a completed run's retained protocol and driver timelines with:

```bash
python -m scripts.miles.analyze_engine_drain /path/to/run --output /tmp/drain-audit.json
```

The audit checks request-attempt uniqueness and actual response/group versions,
and reports independent engine progress and training intervals fully contained
inside a peer's drain/update interval. It does not certify numerical equality,
consumed lag, learning, or recovery.

A separate two-GPU transport probe uses real Ray object-store snapshots and NCCL,
but a synthetic receiver. Run through the same committed-image wrapper:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles scripts/train/debug/miles_engine_delivery_probe.sh
```

CPU tests have demonstrated independent progress, version checks, admission ordering,
last-reader snapshot retention, deadlines and saturated-buffer lifecycle behavior.
The six-update rolling run passed weight equality, useful learning, checkpoints,
evaluation and shutdown; the slow peer allowed 71 responses from the updated
engine. Fresh-process resume, complete optimizer overlap, and deliberate failure
are separate follow-up gates. See the [qualification measurements](measurements/engine-drain-20260913/README.md)
for exact images, attempts, timings and limitations. Per-update full audits remain
disabled; initial weight equality and fixed-byte snapshot tests are separate
numerical checks.

## Future mixed-policy work

The reusable pieces are immutable snapshot ownership, per-engine communicators,
request identities, version clocks and lifecycle barriers. Continuing a partial
response after a weight change would additionally require explicit KV/KDA-state
semantics, per-token behavior versions/log-probabilities, token-age masks and replay
handling across policy changes. None of that is implemented or claimed here.
