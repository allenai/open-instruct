# Packed trainer and inference concurrency sweep

Follow-up to the [unpacked c32 result](throughput-c32-20260913.md). This is a
qualification campaign, not yet a measured recommendation.

All arms use the same full-SFT KDA/latent-MoE checkpoint and GSM8K fixture, EP2
trainers plus two TP1 inference engines, 128 responses per optimizer update,
6,144-token context, 4,096-token response cap, dynamic rows, replay, recomputation,
FIFO groups, lag two, TIS, and the previous objective. Packing is enabled with a
6,144-token budget. It changes auxiliary-loss aggregation across samples in each
pack; this is a throughput comparison, not an exact gradient-parity claim.

| Setting | c32 | c64 | c128 |
|---|---:|---:|---:|
| Per-engine HTTP admission and running requests | 32 | 64 | 128 |
| Decode graph maximum batch | 32 | 64 | 128 |
| Producer sample admission | 512 | 512 | 512 |
| Completed queue capacity, samples | 128 | 128 | 128 |
| Full-attention token pool per engine | 786432 | 786432 | 786432 |
| Recurrent-state slots per engine | 1024 | 1024 | 1024 |
| Updates | 24 | 24 | 24 |

C32 and c64 launched first. The user then requested c256 immediately alongside
c128, continuing to double until inference throughput stops improving or memory
becomes the limit. C256 uses 1,024 producer samples, 2,048 recurrent-state slots,
and 1,572,864 full-attention token slots; HTTP/running/graph limits are all 256.
The pool and producer increases are necessary supporting changes, not isolated
single-variable comparisons. Keep the completed queue at 128 samples and record
backpressure; use sustained inference-only measurement if the trainer hides the
serving ceiling. If packed
trainer demand outstrips inference, use the measured serving curve to adjust the
GPU ratio. If serving is throttled by a full completed buffer, higher concurrency
is not evidence of greater useful throughput. Do not enlarge the lag budget just
to hide excess production. Inspect warmup before selecting a steady window; six
excluded updates is an initial analysis convention, not a claim compilation ended.

## Accounting and acceptance

The previous c32 trial's last periodic observation had 32 completed groups
(128 responses), its full configured capacity, with admission closed for shutdown.
The shutdown drain may grow the queue beyond that last periodic sample.
It also had 32 producer-owned groups, which were not classified as ready versus
unfinished in those old records. Zero *stale dequeue* drops therefore did not mean
zero excess generation. A bounded FIFO applies backpressure once full; its level
need not keep increasing when serving capacity exceeds consumption.

New `pipeline_lifecycle.jsonl` records exact shutdown-start and shutdown-complete
boundaries. Periodic observations now count buffered and producer-ready groups,
samples and response tokens, cumulative/current buffer insertion wait, and
completed but unqueued responses discarded at final shutdown. These final unused
responses are distinct from stale queue drops. Snapshotting never consumes the
queue or resets its counters. Lifecycle files are retained in Beaker results and
included in basket analysis. Failed/cancelled runs may lack a terminal boundary;
an absent record is not a zero balance.

Review all-rank optimizer steps, packed token accounting and router replay;
pack count/fill, model/active tokens per second per trainer GPU; score, train and
publication time; serving running/queued requests, token/state pools, decode rate,
GPU memory and activity; queue occupancy, blocked insertion, stale drops, age,
length, and terminal unused work. Keep historical behavior probabilities and lag
filters unchanged. Re-profile recommendations after packing; do not extrapolate
unpacked trainer utilization to the packed trainer.
