# Capacity dashboard and trainer tuning

The capacity report separates **Trainer**, **Inference**, **Pipeline**, **Drops
and freshness**, and **Warmup and observation coverage**. Its analysis runs are
postprocessed from completed-run artifacts and have job type `capacity-analysis`;
they never rewrite the original training history. Runtime Core trainer metrics
also include the explicit phase rates listed below on subsequent launches.

The [prepared measurements](results/capacity-dashboard-20260913.json) include
the 2T/4I and 2T/6I baselines, EP8, and the 2T/2I concurrency-32 follow-up.
The report has 28 panels; W&B publication is pending explicit approval following
an automatic approval-review rejection. No original training runs were modified.

## What the rates mean

| Metric | Numerator and denominator |
|---|---|
| `train/model_tokens_per_gpu_second` (live Core) | Globally summed model-input tokens / slowest-rank optimizer-step seconds / trainer GPUs. Includes prompts and actual input padding; excludes the preceding standalone scoring pass. |
| `train/active_response_tokens_per_gpu_second` (live Core) | Global active loss-mask tokens / the same optimizer-step seconds / trainer GPUs. |
| `trainer/model_tokens_per_gpu_second` (analysis) | Same global-count convention, using retained per-rank optimizer records and the slowest rank. The recorded boundary is slightly earlier than the live step-summary boundary. |
| `trainer/scoring_model_tokens_per_gpu_second` | Sum of scoring input tokens across ranks / slowest rank's scoring duration / trainer GPUs. |
| `inference/observed_decode_tokens_per_gpu_second` | Sum of observed SGLang generation-throughput gauges / inference GPUs. Gauges can include work not ultimately consumed. Omitted if engine/rank identity is ambiguous or coverage is insufficient. |
| `inference/useful_response_tokens_per_gpu_cycle_second` | Consumed response tokens / awaited normal-cycle seconds / inference GPUs. This is serving-cost attribution, not a measurement of active decode speed. |
| `pipeline/useful_response_tokens_per_allocated_gpu_second` | Consumed response tokens / normal-cycle seconds / all allocated GPUs, including any unused reservations. |

Normal-cycle seconds include collection, score/train, and publication. Generation
runs concurrently; do not add its service time as another serial stage. Evaluation,
checkpointing, export, startup and final drain are excluded from these rates.
These are operational rates, not reward improvement per GPU-hour or model FLOPs.

Existing MILES `perf/tokens_per_gpu_per_sec` and
`perf/effective_tokens_per_gpu_per_sec` divide sample lengths by MILES' rollout
timer and serving GPU count. In async mode that timer must not be assumed to be
the full producer service interval or full training cycle. Keep these existing
metrics, but use the explicitly scoped rates for capacity decisions.

## Signals to watch

| Observation | What to investigate |
|---|---|
| Trainer waits for eligible groups; engines below their running limit; HTTP waiters exist | Admission imbalance, routing, state/token pools, or engine scheduling. |
| Engines reach the configured running limit; substantial pool and memory headroom remain | Increase HTTP slots, running limit and decode graph capture together; compare throughput and latency. |
| Engines finish bursts and the completed FIFO stays full | Trainer backpressure. Faster inference may permit fewer inference GPUs rather than higher training throughput. |
| More producer/queue capacity raises stale drops | Reduce ahead-of-training work; inspect age and length bins before changing the lag contract. |
| Kernel activity is high but tokens/sec/GPU improves with batching | Activity alone did not establish compute saturation. |
| Trainer is supplied but kernel activity is low | Profile gaps, scoring, collectives, dispatch and small kernels. A queue change cannot explain all time inside the training call. |
| Memory appears free but requests cannot enter | Explicit token/recurrent-state pools can bind independently of total free device memory. |

HTTP admission is a **fleet-wide client semaphore**, computed as configured
`sglang_server_concurrency` times engine count. It bounds outstanding generation
requests, including time in the router/server and response processing; it is not
a strict per-engine partition. The router selects an engine. Each engine then
applies its `sglang_max_running_requests` limit and effective memory/state limits.

For the concurrency-32 follow-up, both settings are 32; two TP1 engines therefore
have 64 global HTTP slots. `sglang_cuda_graph_backend_decode="full"` and
`sglang_cuda_graph_max_bs_decode=32` request capture through batch 32. Prefill graphs
remain disabled for the qualified refresh path. Explicit graph JSON overrides
convenience flags, so inspect the resolved plan and actual engine startup logs.

## Trainer knobs supported by this adapter

| Knob | Purpose and constraint |
|---|---|
| Trainer GPUs and `expert_parallel_size` | Change expert distribution and data-parallel work; EP must divide the trainer world and model expert geometry. Compare at fixed optimization batch. |
| `global_batch_size` | Samples per optimizer update. Changes update frequency and RL statistics as well as efficiency; it is not a neutral performance control. |
| `sequence_packing`, `core.packing_max_tokens` | Combine samples into packs with independent sequence boundaries. Can reduce small-work overhead; replay, masks, positions and memory must remain correct. Qualify separately from this unpacked basket. |
| `activation_recompute` | Trade activation memory for recomputation. Turning it off may save work if the full workload fits; memory snapshots and long responses matter. |
| `trainer_flash_attention_version` | Choose a supported attention implementation for the architecture/runtime. Keep numerical validation and compilation warmup visible. |
| `runtime.row_specialization`, compiler cache | Dynamic rows avoid data-dependent specialization churn; persistent cache reduces repeated startup cost. Neither establishes steady-state compute saturation. |
| `core.scoring_pass_required`, scoring check interval | Control the extra pre-update forward only where the adapter's scoring contract permits it. Do not bypass a required async/off-policy score just to improve timing. |
| Replay/parameter/publication diagnostics | Extra validation has a cost. Retain it while qualifying changes; compare production settings afterward. |
| Publication bucket/layout and eval/save cadence | Affect transfer or lifecycle overhead, rather than model forward/backward kernel efficiency. |

The current MoE adapter enforces `micro_batch_size=1` and rejects dynamic
microbatching and trainer TP/CP/PP. It accumulates unpadded samples, or packs them
when packing is enabled. A larger ordinary microbatch is therefore implementation
work, not an available dial. The trainer's low observed kernel activity warrants
profiling; it is not enough evidence to blame any one kernel or collective.

NVML kernel activity is the percentage of sampled time with kernels running.
It does not measure free warp lanes, achieved SM occupancy, tensor-core usage,
or achieved bandwidth. Those require a kernel profiler. Memory is NVML device
usage, including allocator reservations; sampled peaks can miss brief spikes.
Coverage panels distinguish missing observations from zero activity.

## Reproduce or publish a report

The offline reporter currently targets disaggregated qualification runs with one
optimizer step per collection. Shared colocated GPUs need phase-based role
attribution and are rejected rather than mislabeled. It requires completed artifacts, including driver timings,
per-rank contracts, rollout flow, plan and optional pipeline/engine/GPU samples.
The basket starts GPU sampling; arbitrary workflow runs need those samples
collected too if hardware panels are desired. Missing samples leave gaps.

```bash
uv run --with wandb-workspaces python -m scripts.miles.publish_capacity_report \
  --run 2t4i-c8-b128=/path/to/downloaded/run \
  --output /tmp/capacity-report
```

Inspect the prepared JSON, then add `--publish` to create the analysis runs and
report in W&B. `--group`, `--entity` and `--project` select its destination.
Use `--report-url` when adding a newly completed run to an existing capacity report;
the report selects that group's analysis runs. Run IDs are deterministic and
`resume=never` refuses to append duplicate history on an accidental repeat.
Main panels start after six updates by default; the warmup section retains all
updates. Inspect timing traces before interpreting a short window as steady state.
