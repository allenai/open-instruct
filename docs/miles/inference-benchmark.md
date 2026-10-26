# Fixed-policy inference benchmarks

The experimental `[benchmark]` workflow runs the MILES asynchronous producer,
router and SGLang engines without creating trainer workers. It consumes complete
prompt groups, checks response/logprob/replay alignment and discards them after
recording measurements. Weights stay at the initial checkpoint, version `0`.
This measures inference capacity, not training throughput, policy-lag safety or
publication cost. No new serving defaults are qualified merely by enabling it.

Copy a run file to `runs/` and keep the model, data, sampling and serving options
of interest. Retain trainer settings for reference; the allocation plan assigns
them zero GPUs. Use disaggregated TP1 inference with one, two or eight GPUs on one
node, and set `launch.gpus_per_replica` to that GPU count. Remove evaluation
configuration and named judge bindings, disable checkpoint saves/HF export, and
set `launch.auto_resume=false`. Each preempted attempt is a separate measurement.

```toml
[benchmark]
warmup_seconds = 120
measure_seconds = 300
collection_timeout = 900
drain_timeout = 120
capture_samples = 8
reward_mode = "none"

[training]
save_checkpoints = false
filter_zero_std_groups = false

[output]
root = "/weka/oe-training-default/YOUR_USERNAME/open-instruct/runs/inference-benchmark"
export_hf = false

[async]
fully_async = true
```

These are additions/edits to a complete run, not a standalone configuration.
`reward_mode="none"` substitutes a constant reward and requires reward-based group
filtering to be disabled. `"configured"` retains the run's reward computation and
filtering for a second, separate rollout-delivery measurement. The initial
implementation does not allocate managed judges or background evaluators.
Both modes retain ordinary behavior logprobs and requested expert-routing payloads.

To compare several settings serially on the same allocation, add named variants:

```toml
[[benchmark.variants]]
name = "control-before"
overrides = []

[[benchmark.variants]]
name = "prefill-4096"
overrides = ["miles.sglang_chunked_prefill_size=4096"]

[[benchmark.variants]]
name = "control-after"
overrides = []
```

Override an option in the section where it is already declared to avoid conflicting
aliases. Each variant starts a fresh Python/Ray/server process and gets a separate
output directory. The entire panel keeps the same allocation, checkpoint and data;
variants may change serving and queue controls only. A failed case stops the panel.
The Beaker timeout must cover all cases, startup and drain included.

Use the normal `plan`, `validate`, and committed-image `run` commands. Planning
reports only inference GPU allocation and identifies the benchmark explicitly.
Inside the pinned image, `train` dispatches to the benchmark workflow when this
table is present. Plain `debug_rollout_only=true` remains unsupported for training.

## Measurement contract

The producer runs continuously across warmup, measurement and collection
boundaries. Warmup ends at the first complete collection after its duration;
measurement similarly ends at a collection boundary. A collection timeout fails
the attempt. Actual durations, including response checks and consumer accounting,
are recorded rather than substituting the requested duration in the denominator.

The primary rate is **response tokens in complete delivered groups / measured wall
seconds**; the GPU-normalized rate divides by allocated inference GPUs. Prompt
tokens are not counted as output throughput. Startup/cache/graph capture and final
drain are reported separately. Configured rewards and filtering, if enabled, are
inside the measured collection time.

Queue snapshots at boundaries identify completed but unconsumed work. Partial
groups and unfinished requests can also span a boundary: a delivered-token rate
is not an exact count of GPU tokens executed during that interval. Inspect the
engine and pipeline observations, use windows spanning several request lifetimes,
and repeat finalists. A successful shutdown does not retroactively count its
leftovers as delivered work.

Artifacts under `output.root`, also collected into the Beaker result:

| Artifact | Contents |
|---|---|
| `benchmark-summary.json` | Measured delivered throughput, GPU denominator, startup and drain time |
| `benchmark-batches.jsonl` | Complete-collection timing, counts and producer metrics |
| `benchmark-responses.jsonl` | Prompt/response lengths, finish status and replay payload size |
| `benchmark-captures/*.json` | Token IDs, logprobs and route hashes for the first configured sample indices; separate files keep long-response captures within the result collector's per-file limit |
| `benchmark-boundaries.json` / `benchmark-shutdown.json` | Retained queue/work counts at measurement boundaries and shutdown |
| `benchmark-server-start.json` / `benchmark-server-end.json` | Engine identities and resolved server information |
| `benchmark-model-config.json` | Checkpoint architecture/dtype configuration for comparing model shapes |
| `benchmark-engine-counters.json` | Raw engine metrics at measurement boundaries, with scrape timestamp bounds |
| `benchmark-resources.jsonl` | GPU identity/activity/memory and consumer CPU/RSS samples |
| `checkpoints/*occupancy.jsonl` | Existing MILES pipeline and per-engine observations; this directory contains no trainer checkpoint |

The payload checks detect missing/nonfinite logprobs, token/replay misalignment,
unfinished responses and unexpected policy versions. They do **not** establish
numerical equivalence to a baseline or Core. Qualification additionally needs
same-prefix probability comparisons, cache/chunk boundary cases, and control
repeats. Preserve image, runtime lock, prompt manifest and resolved serving mode
with those comparisons.

Analyze a completed case with `python -m scripts.miles.analyze_inference_benchmark
RUN_DIRECTORY --output analysis.json`. The analyzer recomputes collection totals,
checks response accounting, and aligns length distributions, GPU measurements and
queue/engine observations to the measured window. Missing observation coverage
remains unknown rather than becoming zero utilization.
Resource-query timeouts and OS errors are recorded as failed samples and retried;
CPU counters remain available when that sample's GPU query fails. Cancellation
still joins the active query subprocess. Inspect coverage before treating an
observed memory maximum as the peak for the measurement. Older images can stop
resource sampling after the first error even when generation completes normally.
When counter snapshots are available, the report also computes native execution
token rates, prefill work and forward-time deltas. These include work on unfinished
responses and complement delivered throughput. Decode counters exclude the first
tokens selected during prefill; they are not identical to response token counts.
Scrape windows differ slightly from collection boundaries and retain their timing
bounds. Missing/reset counters remain unknown. Forward counters do not account for
every part of engine wall time, so their residual is not automatically a CPU stall.
The pinned runtime only fills forward-time counters with
`SGLANG_ENABLE_METRICS_DEVICE_TIMER=true`; it defaults to false. Keep that setting
fixed within a comparison and measure its overhead before using timed profiles
as throughput evidence.

Preserve full results rather than only summary tables. Run `python -m
scripts.miles.archive_inference_benchmark EXPERIMENT_ID LOCAL_DIRECTORY` after
finalization to download every attempt's Beaker result, full job log and metadata,
with a SHA-256 file inventory. Failed and preempted attempts remain separate.
The benchmark collector retains JSON/JSONL/log reports without the ordinary
32 MiB per-file ceiling; `result-collection.json` records copied and size-skipped
reports. Older images still use that ceiling and need a separate WEKA recovery
for omitted files. Weight trees are not result artifacts. Ordinary throughput runs
retain routing hashes; diagnostic replays additionally retain compressed routing
matrices under `benchmark-replay-routes/`. Token/logprob captures are the configured subset, while response
accounting and timing cover the recorded workload. Keep workload provenance and
launch receipts alongside the archives. Future fits should condition on model
architecture, GPU, runtime revision, pools, scheduling, fleet size and observed
length distributions; do not pool different shapes into a universal admission rule.

For a separate same-prefix diagnostic, copy the benchmark run to a new output
directory, set `benchmark.replay_reference` to the absolute path of a completed
case, and set `miles.sglang_enable_custom_logit_processor=true`. It replays the
captured token sequences twice by default (`benchmark.replay_repeats`), retaining
raw model logprobs and route hashes in `benchmark-replay-*.jsonl`. Foreground
scores are appended as requests
complete so interrupted attempts retain completed work; incomplete workflows are
still rejected by the comparison tool. Repeat controls
and candidates on the same reference. This path checks frozen-token reproduction
and payload lengths; statistical comparison is still required. It does not
produce a throughput number: forced token selection synchronizes CPU and GPU.
`python -m scripts.miles.compare_inference_replay CONTROL CANDIDATE --output
comparison.json` checks reference identities and computes paired prompt-group
bootstrap intervals, retaining siblings and repeated waves in each cluster.
Reports also separate matching replay waves and response-position bands (0–1K,
1K–8K, 8K–16K, 16K–32K and beyond when present). Position bands include only
trajectories reaching them, so inspect their prompt counts; they are conditional
diagnostics and their intervals are not adjusted for multiple comparisons.
When routing matrices are present, comparison reports ordered expert-slot and
response-conditioning token disagreement, including per-layer rates, rather than
only whether any route changed in an entire trajectory. It separately measures
expert membership changes, so permuting the same top-k experts is distinguishable
from selecting different experts. Known Olmo dense layers are excluded from the
denominator; missing layout metadata is reported explicitly. Stored matrix hashes
are verified before comparison. For Olmo MoE payloads with model metadata, the
comparison also rejects negative/out-of-range expert IDs, duplicate selected
experts and a top-k dimension that disagrees with the model. These integrity
checks include earlier prompt positions while excluding dense-layer padding.
`expert_id_contract_checked` is false when the model lacks the expert counts
needed for the complete check; matching hashes alone do not validate expert IDs.
Its ratios and logprob differences are relative to the serving control, not Core;
the intervals do not cover machine/day variability or establish learning quality.
If the capture set is smaller than the admission being tested, set
`benchmark.replay_copies` to repeat those trajectories within each wave (default
one). Use the same copy count for control and candidate. Copies retain their
original prompt cluster and do not increase the independent sample count. This
deliberately changes prefix sharing for the diagnostic; inspect occupancy because
short trajectories finish first and batch size falls during each wave.

For sustained admission through long foreground trajectories, use
`benchmark.replay_background=true` instead. It maintains background streams equal
to the configured fleet HTTP capacity, cycling through the same saved trajectories
through the MILES router. They use the same forced-token hook but omit logprob and
route output. Only foreground trajectories enter agreement statistics; background
completions and interruptions are saved separately. After each wave, replenishment
stops and the private fleet is repeatedly aborted until every background HTTP
response has returned, bounded by `benchmark.drain_timeout`. This prevents queued
router requests from spilling into the next wave. Compare identical background
protocols and inspect measured occupancy; this diagnostic load is not a throughput
workload or additional independent statistical evidence.

For an additional fixed Core reference, copy the same run to another output,
retain `benchmark.replay_reference`, set `benchmark.score_reference=true`, and
use one GPU with no variants. This performs forward-only BF16 scoring with the
pinned MILES/Core model preparation and configured attention backend. It creates
no trainer, optimizer or serving engine. Each original capture is scored once,
with natural routing and one causal document per request at EP1; copies and
repeated replay waves share that reference. The scorer chunks the vocabulary head
to bound memory and retains `benchmark-core-reference.jsonl` and source/backend
provenance. Use `compare_inference_replay ... --core CORE_RUN_DIRECTORY` to report
both serving arms' errors and prompt-clustered paired changes against it. This
is a numerical reference, not a measurement of packed, expert-parallel training
or a guarantee of learning quality. Serving-to-serving agreement alone cannot
determine whether a difference moved toward or away from this reference.

To isolate agreement under the trainer's routing-replay contract, optionally set
`benchmark.score_route_reference` to a completed serving replay directory and
`benchmark.score_route_repeat` to its zero-based wave (default zero). This requires
`score_reference=true`, the same checkpoint and capture hash, and exactly one
route record per original capture (`replay_copies=1`). The diagnostic validates
the saved matrix hashes, shape and expert-ID range before passing each matrix to
the pinned MILES/Core replay context. It retains the full input sequence; native
trainer code supplies each document's unscored final-token route padding.
No serving hot-path checks are added. The summary records `routing=serving_replay`,
and each score vector records its source route hash. Compare these scores with
that exact source wave; they are not an independent natural-routing reference
for comparing two serving arms, so the comparison command rejects them as `--core`.
This remains an EP1 diagnostic, without packing, training or mid-sequence refresh;
it cannot replace the production qualification run.
Each replayed score vector also records `route_context_setup_host_seconds`:
elapsed host time for native route padding/conversion and context entry, including
its existing GPU range/uniqueness predicates and synchronization. This excludes
file loading, the initial route upload and model forward; it does not isolate
individual check costs or predict packed expert-parallel trainer overhead. The
diagnostic adds no CUDA synchronization for this timer.

## Selecting configurations

Start with one engine and memory-feasible pools. Couple admission with decode
graph coverage, then independently tune HTTP permits and producer headroom.
Compare prefill chunks and mixed-chunk scheduling on realistic prompt/response
lengths. Keep sampling, response/context limits and arithmetic unchanged.
Check the resolved arithmetic mode: prefill graphs make this hybrid model
ineligible for automatic Core-compatible rounding, so enabling them is not a pure
scheduling comparison under `auto`. An experimental graph comparison must select
the same explicit arithmetic mode in all arms and qualify runtime compatibility
and numerics; the initial admission study keeps prefill graphs disabled.
Also check `internal_states[].effective_max_running_requests_per_dp` in the engine
snapshot. With the pinned standard overlapping `extra_buffer` cache, the recurrent
pool reserves five state slots per running request; insufficient slots reduce
effective admission with a server warning. Other cache/overlap modes can have
different ratios. Changing recurrent-state precision needs a separate numerical
qualification.
Only advance finalists to two and eight engines; report per-engine imbalance and
fleet throughput divided by engine count times the single-engine baseline.

With `use_miles_router=true` (required for the qualified refresh/replay path), the
pinned router selects the healthy engine with the fewest active requests. Native
SGLang `sglang_router_policy` and cache/balance thresholds do not change this
router. Its `miles_router_max_connections` defaults to fleet HTTP capacity; an
explicit smaller value can introduce another queue. Planning warns about both
ineffective native policy overrides and this extra connection limit.

A short-response lifecycle smoke test is useful before this procedure, but it
cannot establish a long-context serving default. Treat H100 as a separate memory
and throughput qualification rather than extrapolating B300 capacities.

For an illustrative mixed corpus when a production manifest is unavailable,
`python -m scripts.miles.prepare_inference_workload --model CHECKPOINT --output
NEW_DIRECTORY --task math=1024 --task gsm8k=512 --task ifeval=512` prepares pinned
tasks using the checkpoint's tokenizer/template. It explicitly excludes prompts
above 2048 tokens, retains source identities and exclusion records, and never
truncates prompts. Point `data.prompt_data` and `data.reward_config` at its prepared
outputs. This subset is a benchmark workload, not a production data recommendation;
qualify the chosen settings again on the intended production length distribution.

For a separately labeled long-prefill workload, use
`python -m scripts.miles.prepare_inference_continuations --model CHECKPOINT
--output NEW_DIRECTORY --reference COMPLETED_CASE` (repeat `--reference` for
additional cases). This takes natural prefixes from saved trajectories, targeting
8,192 prompt tokens with at least 2,048 original continuation tokens remaining.
It searches backward by at most 32 tokens for an exact decode/encode boundary,
never pads, deduplicates identical prefixes, and retains source hashes, exclusions,
actual lengths and original-prompt clusters. These continuations are useful for
prefill mechanics; siblings and repeated source prompts are not independent data.
The generated verifier deliberately rejects reward computation: use only
`benchmark.reward_mode=none`.
Check corpus size against outstanding work: a small repeated corpus can make
cache-on measurements almost entirely prefix hits. Include a separately qualified
cache-off comparison to measure uncached prefill, and retain the observed cached
and uncached input counters.

Measure response-length sensitivity separately on the same immutable short
prompts, for example 2K/8K/32K response caps with sufficient context. Start with
the same admission and queue settings, then check a neighboring admission to see
whether the knee moves. Retain truncation rates and actual distributions; cap
changes also change which trajectories finish naturally. Long-prefill comparisons
need their own frozen-prefix numerical checks because moving tokens into prefill
changes the execution path.
