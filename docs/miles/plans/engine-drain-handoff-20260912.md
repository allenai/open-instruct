# Handoff: independent engine drain and rolling weight publication

## Objective and scope

Implement an opt-in disaggregated async publication mode that lets each SGLang
engine finish its existing requests, receive a frozen trainer version, and reopen
admission independently. Other engines keep generating and the trainer can consume
completed batches. Routine publication must not cancel and regenerate unfinished
work or wait for the whole fleet to drain.

This is **not mixed-policy generation**. Every response, and initially every GRPO
prompt group, must use exactly one policy version. Different groups/engines may
use different versions within the configured staleness budget. Do not implement
pause/resume across a weight change, old KV/KDA state reuse, concatenated multi-policy
replay, or token-age masking in this task. Stable snapshots, independent publication,
and request ownership should be reusable by that later work, without speculative
implementation of it now.

Keep the current publication mode as rollback. Do not change existing examples to
the new mode until it passes qualification. First support disaggregated Core MoE,
TP1 serving, packing and router replay; validate or clearly reject unsupported
layouts. An engine means the whole TP replica, not one worker rank. TP>1 needs an
atomic update across all workers before reopening; it is not claimed by TP1 tests.

## Repositories and isolation

Create a new branch/worktree from the current primary branches; inspect their
current heads before editing. These names are authoritative, not old hero aliases:

| Repository | Primary branch | Worktree relative to its repository |
| --- | --- | --- |
| open-instruct | robertb/miles-olmo-core | .worktrees/miles-integration |
| OLMo-core | robertb/miles-rl-adapter | .worktrees/miles-core-adapter |
| olmo-sglang | robertb/miles-serving | .worktrees/miles-serving |
| MILES | robertb/olmo-core-backend | .worktrees/miles-runtime |

Suggested Open Instruct branch: `robertb/miles-engine-drain`. Coordinate any
runtime changes through that repo's own isolated branch and update image pins and
patch overlays using the existing workflow. Do not edit disposable runtime sources
and assume they reach built images. Core model arithmetic should not need changes.

The existing campaign is on `robertb/miles-colleague-exercises`, worktree
`.worktrees/miles-colleague-exercises`; do not edit it, cancel its jobs, or reuse its
output directories. Relevant configuration commits: `48d5f8d21` (startup-only
periodic-audit posture), `04b427592` (explicit producer concurrency and sample
replenishment). Copy only needed configurations, with fresh run/output identities;
do not merge the whole campaign merely to obtain fixtures.

## Current evidence and code to inspect

Read `docs/miles/core.md`, `docs/miles/configuration.md`,
`docs/miles/plans/publication-transport-plan-20260912.md`, and the campaign's
`docs/miles/measurements/colleague-20260912/README.md`.

Current flow and constraints:

- `open_instruct/miles/driver.py`: awaits the whole publication, with global async
  producer pause before and resume afterward.
- `async_rollout.py`: `prepare_publication()` cancels/joins active groups and aborts
  engine requests. `data_source.py:requeue_pending_groups()` restores pristine
  prompts, discarding partial response progress. Completed output buffers survive.
- `actor.py:update_weights()`: pauses all engines, exports weights, updates all,
  flushes caches, sets one manager version, resumes all. Trainer collectives and
  model reads are synchronous with this call.
- `publication.py:FlattenedDistributedUpdater`: a shared NCCL broadcast involving
  the engine fleet. Waiting on selected engines without changing communicator
  membership/transport will hang; independent HTTP calls alone are insufficient.
- `async_buffer.py:HomogeneousPolicyDataBuffer`: rejects groups with multiple
  policy versions. `data.py:policy_versions()` and trainer checks enforce lag.
- `state.py`, `arguments.py`, `config.py`, `run_spec.py`: publication clock,
  adapter registration, config/validation. Find `core_publication_boundary` in
  the runtime patches to trace the manager hook too.
- Pinned MILES: `miles/ray/actor_group.py`, rollout manager/server groups,
  `rollout/fully_async_rollout.py`, `rollout/submission_scheduler.py`,
  `rollout/inference_rollout/inference_rollout_common.py` and generation endpoint
  routing. Existing global engine locks and health-monitor pauses need review.
- Tests: `tests/miles/test_async_publication_boundary.py`, `test_async.py`,
  `test_publication.py`, `test_publication_profile.py`; researcher config tests
  under `open_instruct/test_miles_run_spec.py` and `test_miles.py`.

Measured warm EP8 workloads had 512 responses/update, ~60 s training and ordinary
weight transfer ~1.5 s. Full audits every update inflated publication to 44–51 s;
new throughput fixtures disable periodic audits, retaining startup checks.
Cancellation logs showed 57–64 unfinished groups of eight; this is not an exact
count of active decodes or discarded tokens. Approximate decode duration was
37–47 s for a ~3650-token response, excluding queue/prefill/grading.

The legacy producer defaults to `rollout_batch_size` groups: 64 x 8 = 512 responses,
regardless of engine count. Use explicit `miles.async_max_concurrent_samples` and
sample-level replenishment. Completed-buffer capacity is a separate control.

## Implementation sequence

1. **Define ownership and state transitions.** Track each engine's incarnation,
   serving version, admission state, queued/admitted request IDs, and target
   version. Suggested states: serving -> draining -> updating -> serving; failure
   -> unavailable. Closing admission and assigning a request must be atomic with
   respect to one another. Drain includes requests accepted but queued at SGLang,
   not just currently decoding requests. Do not wait for external grading merely
   to release engine weights once the model has finished its response. Keep
   request/group ownership through grading and completed-buffer insertion.

2. **Separate snapshot creation from delivery.** Collectively capture/export a
   complete version at an optimizer boundary, then let the trainer proceed.
   Publication workers must never read live parameters that an optimizer can
   mutate. Decide GPU vs pinned-host staging after measuring snapshot cost and
   peak memory; retaining ~37 GB per version is material. Reuse one snapshot for
   all engines, not one per engine. Bound retained snapshots and release only
   after all readers finish. Backpressure if no safe slot exists. Start with a
   bounded, explicit version queue; coalescing undelivered versions is optional
   and must preserve lag/admission semantics. Avoid concurrent collectives on
   trainer communicators that interfere with EP/FSDP training.

3. **Provide independent engine delivery.** Use a measured per-engine transfer
   path or dedicated update communicators; preserve fused expert export and
   bucketed transport where possible. Do not sequentially re-export the model
   from live trainer state for every engine. Loading, cache invalidation, and
   version acknowledgement must complete before reopening admission. A partial
   update never becomes a serving version. Keep health checks active for healthy
   peers; an update's local health suppression must not hide fleet failures.

4. **Add version-aware dispatch and rolling drains.** Route each new group to a
   chosen version and reserve/dispatch all eight responses to compatible engines
   before those engines close admission. Do not leave unsent siblings depending
   on an old version whose last engine has drained. Groups may span compatible
   engines; pinning a whole group to one engine is a simpler initial choice but
   measure its load balance. Ensure strict draining prevents new work from keeping
   an old engine alive forever. Different engines reopen as soon as each finishes
   updating; no fleet-wide join on the normal path. Restrict simultaneous drains
   if necessary to retain useful serving capacity, using engine states rather
   than introducing a permanent cohort abstraction.

5. **Keep the trainer consuming safely.** Separate trainer step, snapshot-ready
   version, per-engine serving versions, and any fleet-converged watermark.
   Staleness must be measured against the consuming optimizer step, not the oldest
   engine version. Reserve lag headroom for in-flight requests; do not silently
   increase the lag budget. If no eligible completed batch exists, wait and report
   why. A queue of unfinished requests cannot substitute for completed, graded
   groups. Keep TIS/behavior logprobs, replay data, masks and group normalization
   unchanged. Record engine/version from actual execution, not a mutable global
   manager variable at HTTP response completion.

6. **Bound exceptions and integrate lifecycle.** Define drain and update deadlines,
   retry budgets and terminal handling. On expiry, stop/quarantine that engine
   or fail explicitly; never quietly return to routine cancellation. Any forced
   cancellation must be counted and preserve prompt accounting. Checkpoint/resume,
   evaluation, final export and shutdown must have explicit barriers and join
   publication tasks; checkpoint pending-group bookkeeping atomically with model
   state. Startup/recovery publishes before admission. Pending publication errors
   must reach the driver, not disappear in background futures. Avoid deadlock if
   completed-output backpressure blocks producer progress during a drain/shutdown.

Expose a small, validated configuration surface: a publication mode (existing
behavior vs engine drain), bounded drain/update timeout, and bounded snapshot
capacity if needed. Exact names are the implementer's choice. Reject unsupported
combinations early. Do not overload `pause_generation_mode` to imply support that
our producer/transport does not provide. Avoid re-enabling per-update full audits
for performance measurements; use them in a dedicated numerical check instead.

## Validation and acceptance

CPU state-machine and concurrency tests first, using controlled clocks/events:

- One slow engine does not stop a fast engine draining, updating and reopening.
- No new admission after drain begins; queued requests are accounted for.
- Every response/group retains its actual version; unsent siblings cannot become
  stranded, duplicated, dropped, or silently switch versions.
- A snapshot stays unchanged while simulated training advances; bounded retention
  releases only after the last reader. Errors and timeouts reach the driver.
- Completed-buffer saturation, slow grading, no eligible groups, eval/checkpoint
  boundaries and shutdown cannot deadlock or lose prompt accounting.
- Partial publication failure leaves the engine unavailable; recovery republishes
  a complete version before admission. No routine aborts in the successful path.

Then one small disaggregated GPU exercise: existing supported SFT MoE, two Core
trainer GPUs at EP2, two TP1 engines, 4–6 updates with packing/replay and TIS.
Retain real nonzero advantages, gradients and parameter changes. Introduce a
bounded slow-engine condition to visibly demonstrate independent progress.
Compare against the current mode with the same image/model/prompts/recipe and
periodic audits off. Extend the existing tests rather than suppressing failing
modules; distinguish incompatible baked-image tests from regressions explicitly.

Acceptance must demonstrate both normal training and the feature itself:

- Zero publication-induced request cancellation during normal operation.
- Fast engine generates under v+1 while its peer is still draining v.
- Trainer performs an optimizer step while an earlier snapshot is being delivered
  or a peer drains, when an eligible completed batch is available.
- Single policy per response AND group; bounded consumed-policy lag; replay and
  numerical publication checks pass. Confirm actual useful learning work.
- Fresh-process checkpoint/resume and one bounded engine failure pass before
  promoting the mode to shared examples. A happy-path run alone is not recovery
  qualification. Use separate short exercises if that makes diagnosis clearer.

Do not require bit-identical sampled generations across changed scheduling. Use
fixed-token/weight tests for numerical correctness and retained run evidence for
lifecycle, version accounting, learning-path execution and performance.

## Measurements and experiment hygiene

Log per engine/request/group: assigned and executed version, admission/decode/end
and reward timestamps, generated tokens, drain start/end, update start/end,
reopening, cancellations and reason. Report sampled latency separately from group
completion latency. Count useful trained tokens, completed-but-unused responses,
stale rejections, and discarded generated tokens where observable; do not infer
lost tokens from unfinished-group counts.

Separate trainer time, snapshot capture, delivery/load, drain, data starvation,
eval, and shutdown. Include producer/client/engine occupancy, completed queue depth,
lag distributions, peak host/GPU snapshot memory, bytes transferred, and GPU-time
per useful update. Do not sum overlapping stage durations as wall time.

Launch through the documented committed-image MILES wrapper (`python -m
open_instruct.miles run CONFIG`, which uses `build_image_and_launch.sh`). GPU jobs:
Holmes, urgent, `ai2/open-instruct-dev`, positive minimum runtime (normally 1h).
CPU-only jobs needing WEKA: Saturn. Read Beaker job events for pending replicas;
avoid multiple half-allocated multi-node jobs timing out at rendezvous. Retain exact
source/image/config/launch receipts, generations, event timelines and all attempts.
Do not start another full EP8 pool-sizing campaign without coordinating with the
agent owning the current runs.

## Deliverables

A reviewable branch with implementation, focused tests, exact runtime pins,
opt-in small example, lifecycle/support documentation and measured limitations.
Include a concise comparison to the old mode, plus a separate list of work still
needed for mixed-policy continuation. Keep that future capability unclaimed.

Coordination reference: the separate 8-trainer/16-engine, 1024-outstanding-response
throughput run is https://beaker.org/ex/01M2C867KWDKSN1XRH228A6VGM, launched from
`04b427592` on image `01M2BX9JPT44DGQ9QWY5C0HF52`. It still uses cancellation at
publication and is not evidence for this new mode. Query current status; this
handoff deliberately does not freeze a live-run status claim.
