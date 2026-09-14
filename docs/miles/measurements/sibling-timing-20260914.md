# Sibling timing instrumentation

This measures where siblings become staggered, without changing FIFO, producer
admission, router assignment, engine scheduling, or the policy-age limit.
It applies to `core.publication_mode = "refresh"` when the existing
`core.pipeline_observation_interval` is positive. The throughput qualification
profiles already enable that observation setting. SGLang metrics must be enabled
to obtain its engine timestamps; missing fields remain missing.

## Boundaries

Each fresh group attempt gets a UUID before its sibling tasks are created.
Each response records the following boundaries:

| Boundary | Source | Interpretation |
|---|---|---|
| Group created | Producer wall + monotonic clock | Before sibling tasks compete for HTTP admission |
| Admission acquired | Entry to refresh generator | Immediately after MILES' per-response semaphore, before request preparation |
| Engine received | SGLang `request_received_ts` | Engine API receipt |
| Engine first forward | SGLang `forward_entry_time` | First scheduler forward-entry boundary; not a CUDA event timestamp |
| First prefill finished | SGLang `prefill_finished_time` | First prefill completion boundary |
| Engine finished | SGLang `request_finished_ts` | Engine API marks response finished; includes output handling |
| Response received | Producer HTTP helper returns | Complete response body received and parsed |
| Call finished | Refresh generator exits | Includes response/provenance validation; error class retained on failure |
| Group finished | Group generation/reward function exits | All siblings/rewards settled, before insertion into completed queue |

The engine's first-forward and first-prefill timestamps are assigned only once
in the pinned implementation, so ordinary retractions retain those initial
boundaries. Its raw `queue_time` can reflect a later retraction; we do not use
that as initial queue time. Engine first-token time is **not** exposed as a
separate measurement here. Prefill finish is labeled as prefill finish.

Producer durations/skews use a single monotonic clock. Engine timestamp skews
across hosts assume synchronized wall clocks; negative within-request durations
are counted as invalid rather than clamped to zero. This does not qualify
cross-host clock accuracy or measure GPU kernel occupancy.

## Output

`checkpoints/sibling_timing_<hex>.jsonl` shards contain a row for every observed
group attempt, including failed/cancelled attempts and groups later discarded
by stale filtering. Each row contains per-sibling boundaries and first/last/skew
summaries, group outcome, request IDs and response lengths. No prompt, generated
text, routes or log-probability arrays are copied into these artifacts. Retry
attempts receive fresh UUIDs. Artifact write failures are logged and do not
interrupt generation.

`rollout_flow.jsonl` records `sibling_group_attempts` for consumed batches. This
joins consumption to all-attempt timing records; an unconsumed attempt alone
does not establish whether it was stale, filtered, failed, or a terminal leftover.

W&B receives `rollout/siblings/consumed/` summaries (mean, p95, max and coverage
in groups) for admission/engine-start/completion skews, admission wait, HTTP
duration, engine initial wait and engine execution duration. Those metrics cover
**consumed groups only**. Partial groups and missing timestamp coverage are
explicit. Use the all-attempt JSONL for investigating selection bias and stale
work rather than looking only at the W&B summaries.

## Qualification and deployment

Thirty-nine combined timing/HTTP runtime tests passed. CPU tests cover staggered admission versus engine execution, absent/invalid engine
timestamps, attempt resets, failures and cancellation-compatible exception
propagation, artifact write failures, consumed-group coverage and logging. The
refresh hooks and real HTTP diagnostic code are exercised in the pinned runtime
image with its optional Megatron debug plugin disabled on the CPU-only test host.
Real-engine sibling timing still needs qualification in a subsequent GPU run.

The already-running 200-update experiment
[01M2F7N19DQ3YJMRJAXJ1K4H59](https://beaker.org/ex/01M2F7N19DQ3YJMRJAXJ1K4H59)
is frozen at source revision `455d5cd0c`: it has correlated HTTP request logs and
60-second keep-alive, but **does not contain this later instrumentation**. It is
not restarted or hot-patched. The next committed-source qualification can collect
these measurements as part of its normal workload.


## Preliminary admission-only observation from the current run

The correlated HTTP logs already deployed in the 200-update run provide a partial
measurement before the richer instrumentation lands. At the 06:28:52 UTC log
snapshot, 368 groups had all four submission and response-completion records
available to the parser (1,532 parsed submissions, 1,503 completions overall).

| Gap within a group | Median | p95 | Maximum |
|---|---:|---:|---:|
| First to last HTTP admission/submission | 0.3135 s | 1.320 s | 6.958 s |
| First to last response completion at producer | 8.7325 s | 39.995 s | 52.389 s |

These are initial-version requests while the first trainer update is cold. They
are not a steady-state measurement, and incomplete groups are excluded. The
producer submission event is after its semaphore, not group creation or engine
first forward. This subset suggests that most observed sibling separation occurs
after HTTP admission; engine queue delays, response lengths and response handling
remain confounded. It does not establish the cause of later stale drops.
