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
| `benchmark-captures.jsonl` | Token IDs, logprobs and route hashes for the first configured sample indices |
| `benchmark-boundaries.json` / `benchmark-shutdown.json` | Retained queue/work counts at measurement boundaries and shutdown |
| `benchmark-server-start.json` / `benchmark-server-end.json` | Engine identities and resolved server information |
| `benchmark-resources.jsonl` | GPU identity/activity/memory and consumer CPU/RSS samples |
| `checkpoints/*occupancy.jsonl` | Existing MILES pipeline and per-engine observations; this directory contains no trainer checkpoint |

The payload checks detect missing/nonfinite logprobs, token/replay misalignment,
unfinished responses and unexpected policy versions. They do **not** establish
numerical equivalence to a baseline or Core. Qualification additionally needs
same-prefix probability comparisons, cache/chunk boundary cases, and control
repeats. Preserve image, runtime lock, prompt manifest and resolved serving mode
with those comparisons.

## Selecting configurations

Start with one engine and memory-feasible pools. Couple admission with decode
graph coverage, then independently tune HTTP permits and producer headroom.
Compare prefill chunks and mixed-chunk scheduling on realistic prompt/response
lengths. Keep sampling, response/context limits and arithmetic unchanged.
Only advance finalists to two and eight engines; report per-engine imbalance and
fleet throughput divided by engine count times the single-engine baseline.

A short-response lifecycle smoke test is useful before this procedure, but it
cannot establish a long-context serving default. Treat H100 as a separate memory
and throughput qualification rather than extrapolating B300 capacities.
