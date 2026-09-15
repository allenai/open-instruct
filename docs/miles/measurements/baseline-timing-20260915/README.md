# Why the broad baselines have not reached 100 updates

Evidence snapshot: September 15, 2026, 17:54 UTC. This is a timing diagnosis,
not a completed learning comparison. The workload and running jobs were not
changed during this investigation.

[Timing overview](timing-overview.png) · [Stage totals](stage-summary.json)

Raw timing, pipeline occupancy and device samples were copied by a read-only
[Saturn job](https://beaker.org/ex/01M2K30940SGHXVA2PBM0AFAPK), result dataset
`01M2K3094ACG5YVCXDH1QBZC1Q`. Collection used committed source `3ef0b415d`.
The collector retains complete JSONL records at a bounded snapshot size and
never opens checkpoint tensor shards. Local raw logs and artifacts are under
`/tmp/baseline-timing-20260915/`; the Beaker result is the retained source.

## Actual update costs

These are *awaited driver stages*, so async generation overlap is excluded from
rollout wait. Trainer calls include scoring/audits, forward/backward, optimizer
work and compilation. Do not add per-request grading times to this table: those
requests overlap. Checkpoints are amortized across the measured updates.

| Arm / window | Rollout wait | Trainer call | Weight publication | Amortized checkpoint | Total excluding evaluation |
|---|---:|---:|---:|---:|---:|
| Dense broad, completed updates 1–7 across two attempts | 2,410.5 s | 163.2 s | 1.8 s | 5.6 s | **43.0 min** |
| MoE broad, warm updates 20–24 before preemption | 380.1 s | 44.1 s | 3.0 s | 21.2 s | **7.5 min** |
| Core GSM8K, resumed updates 51–129 | 402.5 s | 10.5 s | 1.7 s | 1.4 s | **6.9 min** |

The dense broad run's two completed updates after restart are slower still:
about 50 minutes waiting plus 3.2 minutes in the trainer call on average.
One of those trainer calls includes the restart scoring check.
At the all-seven-update rate, 100 dense broad updates would take approximately
**72 hours of allocated run time**, before startup, evaluation and interruptions.
It cannot reach 100/day with its present configuration. The MoE's measured warm
rate projects about **12.5 hours for 100 updates**, excluding those extras. This
five-update window is an estimate, not a guaranteed steady-state throughput.

Actual checkpoint writes are about 39 seconds for dense and 105 seconds for MoE.
Core GSM8K's two blocking 512-question evaluations took **71.3 and 58.3 minutes**.
Evaluation has not yet contributed to the broad runs' training delay: neither
has reached its first 50-update evaluation. Startup-to-first-publication costs
roughly 15–21 minutes in the broad attempts, including managed-service startup
before the driver timers begin. Neither weight synchronization nor checkpoint
writing explains the broad run's multi-hour delay.

## Dense is limited by policy inference

The seven completed dense updates averaged **2.60 million response tokens** each.
Engine gauges averaged about **215 output tokens/second per engine**, with five
engines. The simple throughput estimate is 2.60M / (5 × 215) = **40.4 minutes**,
which closely matches the 40.2-minute measured rollout wait. Gauge averages are
sampled measurements, not a precise conservation-of-tokens accounting.

Over completed-cycle windows, the observations are:

- 39.99 of 40 HTTP admission slots occupied, with about 967 requests waiting
  outside those slots. These are pending generation calls, not completed samples.
- 7.94 active requests per engine, versus a configured maximum of eight;
  engine-side queued requests average 0.03.
- Completed training queue empty 95.4% of sampled time.
- Inference GPU utilization averages 50.9%, trainer GPU utilization 5.4%, and
  judge GPU utilization 3.8%. These are sampled `nvidia-smi` device utilization,
  not SM occupancy or warp-lane utilization.
- Inference memory averages 162.2 GiB of 268.6 GiB per device. Mean occupied KV
  token-pool fraction is 26.7%; sampled p95 is 40.1%.

The dense profile deliberately retained eight active requests and disabled
CUDA decode graphs when the response budget increased to 32K. Those were
conservative capacity settings, not a qualified high-throughput configuration.
The next serving experiment should hold the recorded workload and model fixed,
compare 8 versus 16 active requests with matching HTTP admission, then qualify
decode graphs. Keep the existing KV token budget initially; measure retractions,
long-request completion, memory peaks and token throughput. Thirty-two worst-case
32K requests cannot all fit in this dense model's KV pool simultaneously.
Revisit engine count only after measuring throughput under the better settings.

The broad batch also differs substantially from GSM8K: **256 versus 64 samples**,
and roughly **2.5–3.5 million versus 160,000 response tokens per update**. Step
counts alone conceal a roughly 16–22× difference in tokens. Dense and MoE also
have very different active model computation; equal GPU counts need not produce
similar inference throughput.

## MoE repeatedly pays its compilation warmup

The logs explicitly report `status: miss` when resumed trainer and serving
workers restore their Triton caches. Driver trainer calls fall from roughly
23–30 minutes on the first call to below a minute after warming. Both the fresh
and resumed processes show this pattern in the plot.

The source explains a persistence gap: `startup_cache.finish()` publishes only
when `success` is true, after worker disposal. Failures skip publication, and
preemption can terminate the process before teardown. Thus a long run can spend
hours compiling without producing a reusable shared generation of its cache.
Local cache directories are ephemeral `/tmp/core-triton-*` paths.

Relative to a late-attempt training-time baseline, the first 24-update attempt
spent about **82 extra minutes inside optimizer-training timing**. That is an
estimated warmup penalty, not a profiler measurement attributing every second to
one compiler. Full driver trainer calls also include scoring checks. Logs include
TileLang and FlashAttention/CUTLASS activity; current managed cache publication
covers Triton, so preserving Triton alone must not be assumed to eliminate every
cold-start cost.

The fix should publish an immutable cache snapshot during successful progress,
with bounded background I/O and retention of the worker's mutable local cache.
Do not reuse `publish_worker()` unchanged while the actor is live: it deletes
the local directory after publication. Qualification must kill/restart a worker
and verify restore hits, unchanged outputs, fewer compiler misses, and lower
first-update cost. An exit-only persistence test is insufficient here.

## Interruptions and lost work

- The early MoE attempt ran about **2 h 15 min**, reached five updates, then failed
  in the old checkpoint-drain path. The saved timers show a 900-second checkpoint
  drain timeout, another 900-second shutdown drain, and 120 seconds of failed
  rollout disposal. No completed checkpoint preserved those five updates.
- The replacement MoE ran four hours and reached update 24, was preempted, then
  waited about **3 h 12 min** until its trainer restarted. It restored checkpoint
  20, so four completed but unsaved updates were replayed. It has reached update
  34 in the snapshot. This preemption was scheduler allocation balancing, not an
  application crash.
- The current one-node dense run completed five updates, was preempted after four
  hours, waited about **3 h 5 min**, and restored checkpoint 5. Its current seven
  updates do not mean it had been training continuously for the whole elapsed day.
  An earlier two-node attempt also exited unsuccessfully after about 3 h 9 min.
- Core GSM8K was preempted after update 50 and resumed about **3 h 25 min** later.
  It has reached update 129 in this snapshot.
- Original Open Instruct reached update 145 before preemption. Its last save was
  125. It remained stopped about **7 h 37 min** before the replacement allocation
  started: a recovery/monitoring gap, not trainer compute. The new explicit
  native-resume wrapper enables automatic preemption recovery; its job is now
  running. The unsaved 20-update ledger was archived rather than counted twice.

The new Core broad attempts automatically recovered. The earlier checkpoint
failure and the original wrapper's refusal to reuse a run directory were separate
implementation problems; neither should be mislabeled as ordinary queue time.

## Remaining grading issue and priorities

Code execution sometimes receives HTTP 503 from the configured AWS API endpoint.
The adapter allows eight retries plus exponential backoff and a timeout per
attempt. Observed failed calls take **517–530 seconds** before zero reward.
Those are concurrent request durations, so summing them would overstate elapsed
wall time. They can nevertheless delay the last sibling in a prompt group.
The worker no longer fails the entire run, but this is not an acceptable total
latency budget. Use a bounded request/retry budget and retain error-rate metrics;
blindly timing out an `asyncio.to_thread` wrapper would leave its thread running.

Priority order from these measurements:

1. Qualify higher dense inference concurrency and decode graphs on the same
   recorded workload; size the inference allocation from the resulting rate.
2. Preserve compiler caches before preemption, with a restart-based acceptance
   test and separate visibility into each compiler family.
3. Bound grading retry latency and report service failures separately from actual
   model failures. Keep those semantics matched in compared runs.
4. Keep automatic recovery enabled and choose scheduling/runtime protection using
   the measured duration. Avoid repeated ad hoc restarts that lose warmup and
   unsaved work.

Further trainer kernel tuning is not the first priority for the dense broad run.
Changing batch size or response cap could make the benchmark cheaper, but that
would be an explicit new comparison recipe, not a performance fix to these runs.
