# Operations and troubleshooting

## Establish completion

Use `python -m open_instruct.miles status run.toml`, then inspect all current
Beaker attempts, taking the latest for each task and replica rank. A zero exit
status, completed workflow state, expected update/eval/checkpoint counts and
retained audit evidence together establish what
finished. A run appearing in W&B or generating responses does not prove updates,
weight publication or clean shutdown succeeded.

| Artifact | What to inspect |
|---|---|
| Local launch receipt | Submitted specification, image ID, submitter revision, allocation and experiment |
| `workflow.json` under output.root | Preparing/training/complete/failed state; a killed job can leave stale state |
| `prepared/` | Model descriptor, immutable data preparation and verifier provenance |
| `rollouts/` | Per-collection samples, behavior versions, rewards and generations |
| `checkpoints/` | Completed native optimizer/model checkpoints for resume |
| `wandb/` | Offline W&B files or online tracking state |
| `driver_timing.jsonl` and contract reports | Stage timings, policy agreement, replay and optimizer/publication diagnostics |
| Beaker `/output` results | run.log, submitted-run.json and bounded JSON/log copies; large tensors stay on WEKA |

Use the exact artifact paths reported by the run; low-level/historical harnesses
can choose different subdirectories. `status` reports attempts and config identity,
not live WEKA metrics. Preserve receipts or set MILES_LAUNCH_RECEIPTS when moving
between laptop and session.

## Interpret measurements

Report rollout/evaluation latency, scoring, optimization, publication, checkpoint,
and orchestration costs separately. Async stages overlap: their durations cannot
simply be summed to derive wall time. Separate cold compilation/restore from warm
steady state, and compare tokens/second and GPU-seconds as well as update time.
Record generated length, cap fraction, mixed-reward groups, reward, lag and TIS
clipping to distinguish a workload change from a kernel speedup.

Centered advantages can make the scalar PPO loss zero at unchanged weights even
when its gradient is nonzero. Check mixed-reward groups, gradients and parameter
changes together. Total gradient norms can include router auxiliary losses; they
do not prove that a batch with identical rewards supplied a policy gradient.

The policy-agreement guard named max_train_rollout_logprob_abs_diff measures the
**mean absolute active-token gap**, despite its legacy name. Replay diagnostics
check supplied expert IDs; native route agreement is a different experiment.
With mixed-policy refresh, retained token spans identify historical behavior
versions, while routes describe the final forward that rebuilt the route table.
Inspect `refresh_scores` for mixed-response counts and historical-prefix versus
latest-forward gaps/TIS clipping; those gaps include actual policy age.
See [implementation contracts](core.md) for exact semantics and evidence limits.

Core MoE runs report router load every optimizer update, including with auxiliary
losses and expert-aware packing disabled. W&B keys under `train/moe/` include
`max_expert_load` (largest assignment count for any layer/expert), `dead_experts`
(total layer/expert pairs receiving zero assignments this update),
`dead_experts_max_per_layer`, `load_cv_mean`, `load_cv_max`, and
`max_mean_load_ratio`. CV is population standard deviation divided by mean load;
zero means uniform. The maximum/mean ratio is one for uniform nonempty layers.
Both normalized metrics are defined as zero for an empty layer.

Counts are aggregated over the complete optimizer batch before each layer's
statistics are computed. `replica_load_cv_max`, `replica_max_mean_load_ratio`,
and `replica_dead_experts_max_per_layer` also expose the worst layer within any
complete expert replica, since pooling replicas can hide local imbalance.
The `router_load` contract/log event includes each layer's global statistics and
assignment total. These are training **dispatch counts**, not gate-weight mass
or a separate evaluation of current router preferences: replayed expert choices
are counted when replay is enabled. All model tokens, including prompts and the
per-document tail row, count. “Dead” means unused in this update, not permanently
inactive. These update totals do not measure worst individual microbatch load.
The implementation reuses Core's forward counters, excludes scoring and backward
recomputation, and gathers only small layer/expert histograms.

## Rollout transport recovery

Refresh sends each generation request once. If a connection fails directly, or
MILES returns its specific `503 {"detail": "Rollout worker unavailable"}` response
for a failed backend `/generate` transport, the producer discards the affected
prompt group and requeues its pristine prompts. The pinned group generator
cancels and joins sibling sample tasks before returning the error. Completed
responses from the failed group are not reused; regenerated responses carry their
own current behavior-version metadata. Other active groups continue normally.

The existing budget permits eight consecutive group transport failures; the
ninth fails the run. A completed group resets that streak. Unknown HTTP errors,
invalid samples/provenance and generation deadlines remain fatal. This is not a
generic retry policy for all 5xx responses, and does not hide a persistently
unavailable service. Server-side work whose HTTP connection was lost may already
have executed; no exactly-once generation guarantee is implied.

`pipeline_occupancy.jsonl` includes `transport_requeued_groups` and
`consecutive_transport_failures`; warning logs identify the affected group and
exception. A router backend disconnect was previously transformed into an
`HTTPStatusError` that bypassed the direct-transport retry predicate. Both sides
of that mismatch predated the September 22 upstream migration.

## Failure triage

| Symptom | Next check |
|---|---|
| Configuration rejected | Error field/context, TOML types, conflicting aliases; rerun with --debug |
| Queued job | `beaker job events JOB_ID`; use the scheduler's reason and latest attempt |
| One node running, another queued; rendezvous timeout | `beaker experiment get EXPERIMENT_ID --format json`: check shared `execution.replicaGroupID`, expected ranks and `execution.spec.leaderSelection`, then inspect events for every replica |
| Import/model failure | Selected image provenance, lock, architecture/tokenizer descriptor |
| Out of memory | Trainer resident state, optimizer initialization, pack/context budget, actual KV/recurrent pools and graph capture |
| Engine unavailable | Per-engine logs, health/recovery events, published policy version and Ray actor state |
| Policy agreement/replay failure | Exact checkpoint/template, token alignment, precision, replay fields and version provenance |
| Save/resume failure | Completed-checkpoint marker, original run specification, output-root ownership and disk capacity |
| Slow shutdown | Compiler-cache phase records and cancellation result; do not attribute all delay to archive compression |

For distributed startup failures, first check the
[scheduling contract](launching.md#distributed-scheduling-contract). Positive
minimum runtimes on separate tasks do not make a replica group. A free-GPU
snapshot does not establish why a group waits or whether Beaker can preempt
eligible work. Do not substitute idle-node polling or a longer application
rendezvous timeout for correcting an incorrectly submitted group. If the group
is correct, use events and logs to distinguish quota/capacity, image startup,
and node failures.

Do not bypass contract failures merely to finish a run. A healthy HTTP process is
not proof that it has the current weights. The runtime's health/recovery support
does not establish recovery of every Core trainer failure.

Compiler-cache publication is best effort. It stages locally, limits publishers
per node and uses a shared bounded publication wait; timeout must not turn completed
training into failure. See [cache guide](compiler-cache.md). Report metadata I/O
is outside that publication wait, so it is not an absolute shutdown deadline.

## Checkpoints while inference continues

Native trainer checkpoints are saved between optimizer updates without draining
inference, including mixed-policy refresh and rolling engine publication. The
driver saves the dataset cursor and pending-prompt ledger, writes the trainer's
model/optimizer/scheduler/RNG state and policy clock, then commits the checkpoint
with the cursor checksum. It does not consume another training batch during this
sequence. Only a checkpoint with its completion marker is resumable.

The async data source snapshots its cursor and pristine pending prompts under one
lock. This briefly serializes prompt admission/bookkeeping with the cursor write;
it does not wait for HTTP requests, generation, reward services, or the completed
queue to empty. Live requests and buffered responses remain usable in the running
job. On resume, **all unconsumed groups in that snapshot are regenerated** under
the restored policy, including completed-but-unused responses. Request caches,
partial responses and historical behavior log-probabilities are not restored.
This preserves prompt accounting and trainer state, not an identical uninterrupted
sampling trajectory. Work admitted after the snapshot is reached again through
the restored dataset cursor.

Evaluation on shared engines, final export and shutdown still have their own
lifecycle boundaries. Removing the checkpoint drain does not qualify those paths
for nonblocking operation. The native trainer write is still synchronous with
training; this change allows inference to continue during it.


## Evaluation admission in mixed-policy refresh

With `core.publication_mode="refresh"`, evaluation closes admission to training
HTTP generation. Samples still waiting for a serving slot stay parked in their
original groups; only already-admitted generation calls must finish before
evaluation uses the engines. Evaluation has separate admission and generation
state. Successful evaluation reopens the training gate without resampling parked
prompts or changing completed siblings. Reward verification that has already
started can finish while the generation gate is closed. An evaluation failure
leaves training paused and propagates to teardown.

Final export and shutdown use the same generation boundary. Shutdown cancels
unused parked work; checkpoint recovery still regenerates unconsumed prompts
from the saved ledger as described above. Ordinary weight refresh continues to
preserve live requests and does not invoke this evaluation drain.

`pipeline_occupancy.jsonl` records `generation_admission_paused`,
`generation_admission_waiters` and `generation_active_calls` alongside existing
HTTP occupancy. `pipeline_lifecycle.jsonl` records `generation_paused` and
`generation_resumed`. Admission waiters include calls parked outside the HTTP
semaphore; they are not requests already running on SGLang. The drain timeout
still bounds active calls, including requests already submitted to a server but
waiting there, so long responses can still delay evaluation.

This behavior requires an application image built with the admission-gate change;
older immutable images continue to drain all producer-owned generations. For a
structured run, omit `async.async_max_concurrent_samples` to use the existing
automatic bound: the larger of one rollout collection and two waves of serving
slots, rounded to whole prompt groups. Measure trainer wait and serving occupancy
before increasing that budget: more queued work does not add inference capacity.
