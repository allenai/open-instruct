# Independent engine drain (experimental)

`core.publication_mode = "engine_drain"` is an experimental alternative to
barrier and refresh publication. It selects rolling updates: each engine stops
admitting new work, finishes the requests it already owns on their admitted
weights, loads the new snapshot and reopens, while its peers keep serving. No
maintained example uses it; the full-model starting profiles use `refresh`, and
small synchronous checks use `barrier`.

## Enabling it

Copy an example to `runs/`, set `core.publication_mode = "engine_drain"`, and
choose a supported MoE model and topology. Check the
[configuration reference](configuration.md), run `plan` and `validate`, and submit
through the [committed-image launcher](launching.md).

The mode requires resident, disaggregated Core MoE with TP1 engines, one
optimizer step per collection and blocking shared-engine evaluation. Automatic
fault tolerance, external engines, custom generators/filters, serving
TP/DP/EP/PP greater than one, and snapshot-fleet evaluation are rejected.

| Field | Meaning |
|---|---|
| `core.engine_drain_timeout` | Bounds drain, request and admission waits. |
| `core.engine_update_timeout` | Bounds update acknowledgement and transport setup. |
| `core.snapshot_capacity` | Number of exported weight versions retained at once (default 2). |

## How it works

The managed async producer reserves every sibling of a prompt group on one engine
and version before sending any HTTP request, including requests still waiting for
the client semaphore. New groups bypass the load-balancing router and use their
reserved engine's URL. Response metadata must acknowledge the reserved version.
Decoding releases the engine reservation before reward computation; the existing
data-source ledger retains the group until consumption or deliberate drop.

After each optimizer step, rank zero packs the exported BF16 weights into a GPU
bucket, copies it through reusable pinned host staging and places one full
snapshot per version in Ray's object store, shared by all receivers. The driver
waits for snapshot capacity before another capture. Provision host/object-store
space for `snapshot_capacity` × the full exported model bytes, plus staging and
normal rollout data; for example, a 37 GB model needs about 74 GB at capacity two.

Each engine has a separate delivery actor on the source trainer node with its own
two-rank NCCL communicator to one TP1 receiver; training communicators and live
parameters are never used by background delivery. This keeps bucketed transfer
but adds host copies and per-engine CUDA contexts compared with synchronous export.
The initial snapshot is delivered and checked against the startup serving-weight
audit before training admission opens.

Publication on each engine closes admission, waits for owned requests, loads the
snapshot, flushes caches and checks the engine's version before reopening. There
is no pause, retract or abort on the normal path. Admission reserves one optimizer
step of lag headroom, and completed groups still pass the usual consumption-time
lag check. Completed groups share the single FIFO buffer; there is no per-version
queue or newest-policy priority. Groups are internally homogeneous, but an
optimizer batch may contain several eligible policy versions.

## Failures, checkpoints and lifecycle

Snapshot versions are delivered in order per engine. Partial updates and wrong
versions make the engine unavailable and propagate a terminal error. There is
**no automatic per-engine retry or recovery**: the run fails and a fresh process
must restore a committed checkpoint.

Native checkpoints do not quiesce inference or join rolling publication; see
[checkpoint resume semantics](operations.md#checkpoints-while-inference-continues).
Evaluation stops producer submission, drains already-owned work and joins
publication; the completion queue temporarily expands by the number of owned
groups so it cannot block that join. Resume regenerates outstanding prompts at
restored weights; it does not restore partial decodes, KV/KDA state, publishers or
in-memory queues. Shutdown joins publishers and closes their communicators before
engine disposal.

While deliveries overlap, the integration holds the inference controller's update
lock and pauses its health checks. Any evaluation, offload or diagnostic operation
that needs the controller lock must first quiesce the publisher. After a delivery
failure, health checks stay paused so incomplete weights are never treated as a
recovered engine.

## Monitoring

`engine_drain.jsonl` records group/request ownership, actual versions, decode
tokens, grade completion, per-engine drain/update/reopen transitions, snapshot
retention and delivery time/bytes. Overlapping stages must not be summed as
elapsed wall time. Analyze a completed run with:

```bash
python -m scripts.miles.analyze_engine_drain /path/to/run --output /tmp/drain-audit.json
```

The audit checks request-attempt uniqueness and actual response/group versions,
and reports independent engine progress. It does not certify numerical equality,
consumed lag, learning or recovery.

## Freshness

Independent drain removes the fleet barrier but does not minimize policy age. It
drains every engine for each publication, delivers every captured version in
order and consumes completed groups FIFO, which can delay current-policy batches.
Possible improvements, not implemented here, include closing admission on a
subset of engines during the next optimizer step, preferring newest-policy groups
in both dequeue and backpressure, and letting a slow engine skip intermediate
snapshots. When comparing schedules, look at time to the first useful
current-version batch, the consumed-version distribution, useful trained tokens
and rejected completed work, rather than generation occupancy alone.

## Relation to mixed-policy refresh

`core.publication_mode = "refresh"` preserves sampled tokens and behavior
probabilities, rebuilds serving state under new weights and records policy-span
metadata, so a single response can span several policy versions. `engine_drain`
instead finishes each request on its admitted weights and never mixes policies
inside a response. See the [publication-mode guide](grpo.md#publication-modes)
and [mixed-policy refresh](async-pipeline.md#mixed-policy-refresh).
