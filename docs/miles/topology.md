# Topology and capacity

Start with [structured examples](../../configs/miles/examples/README.md) and inspect
`plan` before allocating GPUs. The full async starter is **one EP8 trainer node
plus eight TP1 rollout engines on a second node**, not the historical EP2 + one
engine study. Recommended configuration and measured qualification are distinct.

| Control | Meaning |
|---|---|
| `trainer.gpus` | Trainer GPUs per trainer node |
| `trainer.trainer_num_nodes` | Number of trainer nodes; multiply by trainer.gpus for world size |
| `trainer.expert_parallel_size` | Core expert process group; not inference TP |
| `inference.gpus` | Total rollout GPUs; divided by GPUs per engine for engine count |
| `inference.rollout_tensor_parallel_size` | GPUs per rollout engine |
| `launch.gpus_per_replica` | Physical GPU allocation per Beaker task/node |
| `inference.placement_mode` | `colocated` shares GPUs; `disaggregated` separates pools |

Trainer TP, PP and CP must remain one in this adapter. Expert parallelism is
supported; inference TP is a separate engine setting. Router replay does not
change those parallelism limits.

Core keeps the trainer resident. The tiny colocated example is a development
check; full-model optimizer/engine coexistence has not been qualified by it.
Multi-node jobs use explicit tasks with disjoint hostname pools; judge GPUs
belong to their own fixed-weight service. Plan reports unused slots. Consult
[managed judges](managed-judges.md) for current placement restrictions.

## Collections, optimization and async

The baseline collects **8 prompts × 8 responses = 64 samples**, with global batch
64 for one optimizer update per collection. Smaller global batches introduce
multiple optimizer updates and change policy lag requirements. Core microbatch
size remains one; optional [packing](sequence-packing.md) groups original samples
into document-isolated forwards within an optimizer partition.

Async requires resident disaggregated engines. The starter uses staleness one,
buffer factor two, retry of unused groups, group submission, and TIS with
trainer-scored old log probabilities. Staleness counts optimizer steps, not elapsed
seconds or merely collections. Multiple optimizer steps per collection consume
lag allowance. Replay, auxiliary loss and reference KL are independent settings.

Eight engines do not each get 64 requests from a 64-response collection. Admission
is a ceiling, not guaranteed occupancy; async buffering can provide additional
work. Change collection/global-batch sizes deliberately when studying utilization.
Do not infer an eightfold speedup from eight rollout GPUs.

## Serving and memory

Set client concurrency, engine maximum running requests, decode graph size,
full-attention token capacity and recurrent-state cache capacity together. The
starters expose concurrency 64; shared-engine held-out evaluation uses those same
limits. Context includes prompt plus response; response caps alone do not bound
prompt memory. Trainer pack budget and serving context are different controls.

Measure actual allocated KV/recurrent pools, memory after optimizer creation and
graph capture, retractions, response tails, tokens/second and tokens/GPU-second.
For prefix-cache policy see [run controls](run-controls.md); do not assume ordinary
radix caching is interchangeable with KDA recurrent-state caching.

## When radix caching is useful

Radix caching is most promising for **long, repeatedly reused prompt prefixes**:
shared system instructions, few-shot examples, repeated context, or multiple
responses to the same prompt. Long prompts alone are insufficient if their token
prefixes differ. Reuse also depends on requests reaching an engine with matching
cached state, cache capacity/eviction, and invalidation when policy weights change.
A cache-aware router can help request placement; it cannot create shared prefixes.

The benefit is avoiding repeated prefix processing. For short prompts followed by
long generated responses, that can be a small part of the workload. Judge latency,
training, publication or decode can still determine end-to-end cadence. Compare
warm update wall time and trainer data waits alongside cache hits and engine
throughput; a higher engine throughput number alone does not establish a faster run.

A preliminary 20-update comparison reported on September 12 found about **45 cached
tokens per sample versus roughly 3,600 generated tokens**. Warm updates 3–20 took
31.5 minutes with cache off and 32.0 minutes with radix plus cache-aware routing:
no measurable end-to-end gain in that comparison, despite cache hits and increased
reported engine throughput. This is a training-phase observation; both arms lost
final evaluation to a code-service error. Matching mean trainer/behavior log-prob
gaps (0.0241) do not establish tokenwise numerical identity. The full experiment
report and mixed-chunk result were still pending when this guidance was added.

Treat radix caching as workload-dependent tuning, particularly worth measuring
when substantial prefixes repeat. Keep the example defaults until a representative
comparison supports changing them. See [run controls](run-controls.md) for the
radix switch and retain the model-specific KDA recurrent-cache requirements.

The current evidence includes B300 EP1/EP2 numerical and lifecycle checks,
full-SFT async/admission runs and small multi-node exercises. EP8 packing throughput,
full-model colocation, new architectures and other GPU types require their own
qualification. See [measurements](measurements/index.md); old EP2 timing is not an
EP8 capacity claim. FlashAttention backend names alone do not establish H100 support.
