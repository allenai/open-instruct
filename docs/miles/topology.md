# Topology and capacity

Start with a [maintained example](../../configs/miles/examples/README.md) and
inspect `plan` before allocating GPUs. The
[generated recipe tables](configuration.md#example-recipes) show current
allocations. This guide explains how those counts map to physical nodes.

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
Multi-node jobs fill spare GPUs on the final trainer node with whole inference
engines, keeping trainer ranks first in GPU order. Distributed launches require
a native Beaker replica group and full eight-GPU nodes; see the
[scheduling contract](launching.md#distributed-scheduling-contract). Judge GPUs belong to their
own fixed-weight service. Plan reports unused slots. Consult
[managed judges](managed-judges.md) for current placement restrictions.

## Collections, optimization and async

Collection size and optimizer batch size determine how many updates each
collection produces; see [batch geometry](workflow.md#collection-and-policy-settings).
Core microbatch size remains one. Optional [packing](sequence-packing.md) groups
original samples into document-isolated forwards within an optimizer partition.

Async requires resident disaggregated engines. Staleness counts optimizer steps,
not elapsed seconds or collections. Multiple optimizer steps per collection
consume lag allowance. See [policy lag and TIS](async-pipeline.md#policy-lag-and-tis)
for the current refresh settings. Replay, auxiliary loss and reference KL are
independent settings.

Eight engines do not each get 64 requests from a 64-response collection. Admission
is a ceiling, not guaranteed occupancy; async buffering can provide additional
work. Change collection/global-batch sizes deliberately when studying utilization.
Do not infer an eightfold speedup from eight rollout GPUs.

## Serving and memory

Set client concurrency, engine maximum running requests, decode graph size,
full-attention token capacity and recurrent-state cache capacity together. Use
[admission sizing](throughput-profiles.md#size-engine-admission-from-memory) to
choose compatible limits; shared-engine held-out evaluation uses those same limits.
Context includes prompt plus response; response caps alone do not bound prompt
memory. Trainer pack budget and serving context are different controls.

Measure actual allocated KV/recurrent pools, memory after optimizer creation and
graph capture, retractions, response tails, tokens/second and tokens/GPU-second.
For prefix-cache policy see [run controls](run-controls.md); do not assume ordinary
radix caching is interchangeable with KDA recurrent-state caching.

For length-specific budgets, controls and qualification evidence, see
[long sequences](long-sequences.md).

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

The [archived radix-cache comparison](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/radix-cache-ab-20260912.md)
records the workload, timing and failure limits of that experiment. Its
historical default suggestion does not override the current example TOMLs.

Treat radix caching as workload-dependent tuning, particularly worth measuring
when substantial prefixes repeat. Keep the example defaults until a representative
comparison supports changing them. See [run controls](run-controls.md) for the
radix switch and retain the model-specific KDA recurrent-cache requirements.

The current evidence includes B300 EP1/EP2 numerical and lifecycle checks,
full-SFT async/admission runs and small multi-node exercises. EP8 packing throughput,
full-SFT MoE colocation, new architectures and other GPU types require their own
qualification. See [measurements](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/index.md); old EP2 timing is not an
EP8 capacity claim. FlashAttention backend names alone do not establish H100 support.
