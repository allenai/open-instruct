# Throughput profiles and qualification

The four new starter files are **candidates under qualification**, not measured
optimal ratios. They target Holmes B300 hardware for the full SFT model used in
the GSM8K comparison. Other model sizes, attention types, GPU memory and networks
can require different settings. Existing examples remain available.

| Profile | Placement | Purpose | Model assumption |
|---|---|---|---|
| [dev](../../configs/miles/examples/dev.toml) | 1 shared GPU | GSM8K mechanics, saving and basic data plumbing | Small MoE; Core optimizer and serving pools fit together. The qualification fixture is random and is not an accuracy model. |
| [tiny](../../configs/miles/examples/tiny.toml) | 1 trainer + 1 inference GPU | Disaggregated mechanics and very small jobs | Same small model fixture for the initial exercise. A full SFT model is not assumed to fit one trainer GPU. |
| [small](../../configs/miles/examples/small.toml) | 2 trainers + 4 inference GPUs | First useful throughput comparison | Full SFT checkpoint, EP2, TP1 engines. Ratio is compared with 4+2. |
| [large](../../configs/miles/examples/large.toml) | 8 trainers + 56 inference GPUs | 64-GPU scale target | Full SFT checkpoint, EP8 trainer node and seven inference nodes; qualification follows smaller checks. |

Small and large currently opt into experimental mixed-policy refresh. They use
radix caching, the qualified extra-buffer KDA strategy, graphs disabled as required
by refresh, direct bucketed publication, and preserved historical behavior scores.
They are not a statement that CUDA graphs, fused export, or packing are undesirable:
those should be compared in subsequent focused measurements rather than silently
combined with a new scheduling experiment.

## Selection rationale

* Dev/tiny use modest admission and short responses. We will establish a working
  configuration once, without a separate optimization campaign.
* Small uses 8 prompts × 4 responses (32 samples) per update. Its four alternatives
  retain the same full SFT weights, prepared GSM8K input, response cap 4096, seed,
  LR, clipping, and age limit. Completion order can still select different prompts.
* Large uses 64 prompts × 4 responses (256 samples) per update. This explicitly
  changes the optimization batch from small; compare useful tokens/second and
  discard rates, not update counts as though learning configurations were equal.
* Producer concurrency derives from engine count and requested admission. The
  completed FIFO holds one collection in the throughput candidates. Neither
  setting is inferred from physical node count alone.
* Requested token and recurrent-state pools cover the configured admission at
  the chosen context limit, subject to runtime GPU fit. Actual SGLang limits must
  be recorded; a configured maximum does not prove it was achieved.
* Weight publication is accepted overhead. Record its fleet-wide latency alongside
  training wait and token-discard fractions; do not hide it inside generation.
* The 64-GPU profile requires multi-node coordinated launch with auto-resume
  disabled until that restart topology is qualified. Checkpoints remain available.

## How the limits interact

| Control | What it limits | External constraint and tuning signal |
|---|---|---|
| Trainer GPUs / EP | Model and optimizer distribution, training time | Expert divisibility, model memory, collective bandwidth; compare training time at the same optimization batch. |
| Inference GPUs / engine TP | Independent engines and model fit | GPU memory and interconnect; increasing TP reduces engine count at a fixed GPU budget. |
| Producer sample budget | Unfinished generation-and-reward work | Fleet service time, grading latency and group stragglers; enough headroom to refill engines, then watch age and discarded tokens. |
| HTTP concurrency per engine | Global generation semaphore, scaled by engine count | Too low leaves serving slots empty; too high moves waiting work into the serving system without creating GPU capacity. |
| Running requests per engine | Requested decode batch admission | Effective token pool, recurrent-state pool and GPU memory can cap it further. Record the engine's resolved limit. |
| Token/context and recurrent-state pools | Capacity for active contexts and cached prefixes | Model geometry, dtype, radix strategy, overlap scheduling and memory headroom. KDA state slots are not necessarily one per request. |
| Completed-buffer factor | Whole ready groups waiting for consumption | Trainer service rate and allowed policy lag; a larger queue absorbs bursts but cannot repair a sustained rate mismatch. |
| Collection / optimization batch | Responses collected and samples per optimizer step | Trainer divisibility, memory, desired RL statistics; changing these is an optimization change, not just a throughput tweak. |
| Allowed policy lag | Which completed groups remain eligible | Current trainer version versus oldest sampled token version; raising it accepts more off-policy data rather than making generation faster. |
| Publication interval | How often serving receives current weights | Collective transfer and re-prefill latency versus policy freshness; keep this cost visible. |
| Save / eval cadence | Interruptions outside normal training cycles | Checkpoint I/O, evaluation size and draining outstanding work; compare total run time separately. |

Start with model fit and the desired optimization batch, then size engine
admission from memory. Use the topology-derived producer budget as a starting
point. If training waits and engines have spare capacity, increase admission or
backfill; if engines are already busy, compare more inference GPUs. If completed
work ages out while training stays busy, reduce ahead-of-training work before
loosening the lag limit. Report dropped tokens as well as samples: a small sample
fraction can hide substantial wasted long-response work. Keep the length/age
breakdowns alongside the aggregate so this tradeoff stays visible.

Persistent compilation caching reduces repeat startup cost; it does not increase
engine admission or buffer capacity. Keep it enabled for normal examples, while
recording cold and warm timings separately. Cache-off comparisons should use
separate run identities and private cache locations, not delete shared caches.

## Nine-case basket

| Order | Case | Question |
|---|---|---|
| 1 | dev | Does one-GPU GSM8K plumbing and small checkpoint saving work? |
| 2 | tiny | Does separating trainer and serving work without changing the model? |
| 3 | small-2t4i-group | Initial six-GPU baseline, 16 requested engine slots each |
| 4 | small-4t2i-group | Is compute or serving the limiting side at six GPUs? |
| 5 | small-2t4i-sample | Does per-sample backfill reduce sibling-straggler idle time? |
| 6 | small-2t4i-c8 | Does lower engine batch concurrency improve useful throughput/latency? |
| 7 | bridge-2t6i | Does additional inference still help the two trainers? |
| 8 | bridge-8t8i | Qualify inter-node publication and the larger training batch |
| 9 | large-8t56i | Measure 64-GPU behavior once the smaller boundaries work |

Failures are findings, not reasons to silently change a case. Any repair gets a
new run identity and records the changed setting. A result that needs a different
model must be labeled as a different qualification. We will refine ratios based
on the measured cases, not advertise the starting ratio as optimal.

The runner uses `plan/validate/train` workflow configuration, an immutable base
image plus a checksum-verified committed source archive, separate WEKA roots,
urgent Holmes placement, and W&B group `throughput-profiles-20260913-v1`.
CPU-only WEKA analysis runs on Saturn. Source inputs and earlier experiments are
not modified. The small/large benchmark disables saves and eval to isolate normal
cycles; the examples retain practical save/evaluation cadence.

Each performance case initially runs 12 updates and reports the window after
three warmup updates. Report per-step timings too: three warmups do not establish
that all compilation has stopped. Dev/tiny run four updates and are mechanics
checks. A successful workflow and complete, non-skipped optimizer sequence are
required before producing a success report. This is performance qualification,
not evidence of comparable learning quality or full resume equivalence.

## Warnings, validations, and evidence

`plan` now reports `runtime.throughput` as well as `runtime.async_capacity`.
Structured `validate` and driver startup print the advisories. Whole inference
engines, positive publication intervals, and valid prefill chunk values are hard
checks. Token-pool coverage, graph coverage, resident colocation memory, reference
model cost, frequent saves/evals, diagnostics and group-straggler backfill are
warnings because intentional small/development configurations remain valid.

Read the [queue guide](async-pipeline.md) for exact producer, semaphore, engine and
completed-buffer semantics. `rollout_flow.jsonl` retains response lengths, useful
tokens, mixed-response counts, and queue discard metrics independently of W&B.
`driver_timing.jsonl` distinguishes checkpoint draining from writing checkpoints.
The analyzer reports warm consumer-wait fraction, useful response tokens/second,
discarded-token fraction, and stage medians, with all lifecycle timings separate.
Its normal-cycle sum is not an engine GPU utilization measurement: generation
runs concurrently with training. Saved evidence and Beaker links will be added
here as cases finish. Qualification is limited to the model and workload exercised.

## Results recorded September 13

[Machine-readable measurements](results/throughput-profiles-20260913.json) retain
source/image provenance and per-update records. Initial mechanics results:

| Case | Result | Warm median generation wait / training / publication | Evidence |
|---|---|---|---|
| Dev: 1 GPU colocated | Four updates and checkpoint saves passed | 2.05 / 0.38 / 0.09 s | [Beaker](https://beaker.org/ex/01M2DZWJCXVY30Q53BWAPVWJ9V) |
| Tiny: 1+1 disaggregated | Four updates and checkpoint saves passed | 2.78 / 0.42 / 0.10 s | [Beaker](https://beaker.org/ex/01M2DZWK3SAWW7AD47A3PP61PQ) |

Both used the same random small conventional MoE geometry and eight prepared
GSM8K questions, with four prompts × two responses per collection and a 256-token
response cap. These prove data/trainer/publication/save mechanics; they do not
measure GSM8K learning or qualify full SFT model fit. All four optimizer steps
were present and not skipped. Resume and export are not part of these two runs.
The modest timing difference is not an optimization finding from three warm
collections. Use dev as the minimal exercise, and tiny when the disaggregated
boundary itself matters. Full-model ratio and scale qualification remains in
progress.

The first full-model 2+4 and 4+2 attempts each completed one optimizer update
and published version 1, then failed when consuming the next collection. Refresh
had incorrectly reused `core.engine_drain_timeout=180` as a whole generation
request deadline. Slow responses timed out in the background while the cold
first training step ran. These are failed qualifications, not throughput results:
[2+4](https://beaker.org/ex/01M2DZWM5KP9V005R2BEX0ZK93),
[4+2](https://beaker.org/ex/01M2DZWNXMG5KSDA9SSM8HQWXT).

Generation now has a separate `core.refresh_request_timeout` (1800 seconds),
covering serving queue time, decoding and refresh pauses. The examples set
`core.engine_drain_timeout=900` for save/eval/shutdown; transfer time remains
bounded separately. On timeout the error identifies the request and relevant
setting. Regression tests verify that a short drain budget cannot cancel an
ordinary request and that the generation deadline cancels its HTTP task.

A separate c8 attempt was canceled because its host reported zero temporary-disk
space. Subsequent basket submissions exclude that host. Attempts stopped for
these known issues are excluded from performance comparisons; retries have new
identities and keep the same model/data/optimization settings.
