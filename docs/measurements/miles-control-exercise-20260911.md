# Core scoring, scheduling and configuration exercise

This campaign uses the committed runtime containing Core `307d20590`. It reuses
our immutable full-SFT GSM8K preparation and initial HF checkpoint under
`/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1`.
It is a performance and interoperability exercise, not a learning comparison.

One image launches three allocations, at most eight B300 GPUs, on Holmes in
`ai2/open-instruct-dev`, urgent priority with one-hour minimum runtime:

| Allocation | Work | GPUs |
| --- | --- | --- |
| scores | Static then dynamic row specialization, independent processes/caches, same retained batches 5–9, first and repeated scoring | 2 |
| scheduling | 24 synchronous then 24 asynchronous updates, same hardware, fresh model and private caches per arm | 3 |
| controls | Four asynchronous updates with grouped options and independent audits | 3 |

The scoring arm checks the requested selector reaches every routed-expert
module and requires exact per-token scores between modes and on repeats. It
retains JIT misses, compiler artifact writes, wall/stream timings and token counts.

The scheduling pair differs only in async enablement, buffer settings, and the
allowed policy lag (zero vs one update). Both use rollout log probabilities,
dynamic rows, EP2 training, one resident SGLang GPU, full decode graphs, four
concurrent requests, 4 prompts × 4 responses, and a 4096-response-token limit.
Caches are cold and isolated per arm; report startup and the first four updates
separately from the remaining 20. This does not qualify persistent cache reuse.
Async completion order can change consumed prompts, so compare response-token
throughput as well as update time. Async generation wait is consumer stall time,
not total inference work; overlapping phases must not be summed as serial work.

The controls arm exercises TOML roundtrip, `plan`, native `validate`, the actual
`train` CLI and `--set`; TIS clipping, buffer factor 2/drop policy, cosine LR with
warmup, entropy loss, concurrency/graph batch size 8, smaller publication
buckets, offline W&B, initial/final eight-question heldout evaluation, one native
checkpoint, and a final exact current-weight publication check. Combined changes
establish interoperability only; they do not isolate each knob's performance.
No reference-KL, replay, failure injection, or checkpoint-resume claim is made.

All live arms retain samples and independently check prepared identity, prompt
token hashes, rewards, full groups, no repeated consumption, per-group policy lag,
rank-local behavior versions, optimizer steps, and publication sequence. The
controls arm also checks saved manifests/cursors/shard presence and heldout results.

Launch after committing:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_control_exercise.sh
```

Artifacts are under `control-exercise/$BEAKER_EXPERIMENT_ID` on WEKA, with
reports and logs copied to the Beaker result. Scoring tensor results are small;
large native checkpoints remain on WEKA. A job exit alone is not an audit pass.

Submitted experiment: [01M28PAQW8YCJ6P8M9F2343PYH](https://beaker.org/ex/01M28PAQW8YCJ6P8M9F2343PYH).
Source: `d8beffb81`; immutable image: `01M28PAA65C3MQ0QWZRP3KK6Z7`.
All three tasks were placed on `holmes-cs-aus-504`, using disjoint GPUs on the
same host. Shared host/IO/fabric contention may affect the overlapping startup
and early measurements; this is not an isolated hardware benchmark.

Prelaunch validation: 30 runtime parser/audit tests and four paired KDA/latent
GPU training parity cases passed; formatting/static checks passed. The scoring
recipe guard subsequently passed 10 tests including an injected unrelated
configuration difference. Results below must come from completed remote audits.
