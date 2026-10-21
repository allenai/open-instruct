# Multi-node runs and named GPU judges

Distributed disaggregated runs require a Beaker replica group on distinct physical
nodes. Trainer GPU count, rollout GPU count and judge GPU count are independent.
See the [scheduling contract](launching.md#distributed-scheduling-contract)
before submitting a multi-node run. Short execution checks do not establish
mixture learning quality.

Start from [medium.toml](../../configs/miles/examples/medium.toml), copying it to
`runs/` before changing policy, dataset, verifier and judge inputs. Run `plan` and
`validate` on that personal file, inspect the rendered replica group, then submit
through the [committed-image launcher](launching.md). The example paths are
placeholders, not downloadable prepared inputs. CPU-only preparation needing
WEKA belongs on Saturn; GPU placement follows the selected run topology.

## GPU ownership

`launch.gpus_per_replica` is the physical GPU allocation per replica. For a run
that fits on one node, only the requested GPUs are allocated. Larger disaggregated
runs place trainer ranks first, fill spare GPUs on the final trainer node with
whole rollout engines, then add as many rollout nodes as needed. These engines
use separate GPUs from the trainer; this is disaggregated placement.
Managed judges fill spare space on the final rollout node, then additional nodes.
All replicas have the same allocation size; the plan reports any unused GPUs.

| Request | Allocation at 8 GPUs/replica |
| --- | --- |
| 8 trainer + 7 rollout + 1 judge | 2 nodes, 16 GPUs, no unused GPUs |
| 4 trainer + 11 rollout + 1 judge | 2 nodes, 16 GPUs, no unused GPUs |
| 8 trainer + 23 rollout + 1 judge | 4 nodes, 32 GPUs, no unused GPUs |
| 8 trainer + 8 rollout + 1 judge | 3 nodes, 24 GPUs, 7 unused GPUs |

The last case deliberately keeps all eight rollout GPUs. A later heterogeneous
allocation launcher could avoid those unused GPUs. Increasing inference capacity
is already expressible with `inference.gpus`; engines must fit within a node and
tensor parallelism must divide the node capacity. Multi-node colocation, separate
evaluation GPU pools, cross-node serving TP and automatic coordinated restart
remain unsupported. Restarting a run that carries a managed judge is untested;
multi-node resume has been tested only without a judge in the allocation.

The launcher submits one task with native replicas, leader selection and
synchronized start. Multi-node launches require full eight-GPU nodes; partial-node
allocations are rejected because replicas could share a physical host. Optional
hostname filtering uses one shared allowlist for the entire replica group.

Before Ray starts, replicas exchange addresses on WEKA and sort them numerically
as MILES sorts placement bundles. Beaker replica zero is not assumed to own the
trainer. Judge devices are excluded from Ray's CUDA mask and advertised GPU count.
The readiness gate checks the exact live-node GPU layout, not just its sum, and
rejects replicas sharing a physical address. This prevents a judge from receiving
policy weights or a trainer rank from landing on a judge device.

## Named services, rubrics and bindings

The schema follows the Megatron implementation: `[judges.NAME]` declares serving, `[rubrics.NAME]`
declares grading, and `[judging.bindings.VERIFIER]` maps prepared verifier names to
both. Two rubrics can use one fixed service. Only bound services consume GPUs.
The medium and large examples bind `general-quality` and `general-quality_ref`.

Managed services use the pinned training image's SGLang and a cached,
revision-pinned Qwen model with the `qwen3-no-thinking` template. Model
weights are prepared before GPU allocation. `max_context_length`,
`max_concurrent_calls`, `timeout`, and grading output/temperature are independent
of policy generation settings. Existing unauthenticated OpenAI-compatible services
can use `mode="external"`, `endpoint`, `model`, and the same context/concurrency
settings; their lifecycle and context enforcement remain externally owned.
Independent vLLM allocations, credentialed external judges and custom rubric files
available in the Megatron implementation are not supported yet; unsupported keys fail.

For long policy responses, budget the complete grading request in **judge**
tokens: question, candidate answer, optional reference, rubric, chat framing and
the judge's output. A 32K policy response can exceed 40K judge tokens because the
tokenizers differ. Managed Qwen/Qwen3-32B can explicitly request
`context_extension="qwen3-yarn-128k"` with `max_context_length=131072`. This uses
Qwen's documented factor-4 YaRN configuration (original context 32768), passed as
a serving override without rewriting prepared weights. The native default stays
unchanged; unsupported models, double scaling and limits above 131072 fail.
This changes judge computation, so compare runs using the same judge setting and
check long grading requests before using it for a benchmark.

The judge client matches the Megatron implementation. It uses the same rubric
text/digests and answer extraction, keeps `metadata.judge_query` and reference
labels, tokenizes the full grading request plus its output reservation, and
rejects overflow without truncating the request. Incomplete/invalid grades and
exhausted transport errors fail the run. Raw replies, reasoning, rubric/model
identity, retries and latency are retained in `metadata.verifier_diagnostics`.
Named bindings use the sample-aware reward bridge; they never import code selected
by data. Deterministic verifiers keep the existing Open Instruct path. Code uses
the baseline HTTP payload and strict transport error handling.

## Lifecycle and audit artifacts

A unique submission UUID isolates coordination records under
`OUTPUT/cluster/UUID`. Each replica supervises its local Ray/judge process groups
and publishes a heartbeat. Startup and heartbeat timeouts are configured in
seconds through `launch.coordination.startup_timeout` (1200) and
`heartbeat_timeout` (120). Peer failures propagate through shared records and
Beaker's failure/preemption propagation. No Beaker API credential is necessary
for the packed replica group. Liveness uses SGLang's non-generating `/health` mode and three consecutive
failures; actual grading requests retain their own hard deadlines. Normal completion uses a peer acknowledgement before
tearing down Ray. Service failures stop training; no automatic reward-zero fallback.

The driver checks model discovery and known-good versus known-bad answers through
each named binding before training. Inspect `placement-*.json`, `ray-layout.json`,
`judge-canaries.json`, `registry.json`, `complete.json` and `cleanup-*.json`, alongside
Core optimizer/publication/replay contracts and retained rollout metadata.
`driver-RANK.log` and `judge-NAME-RANK.log` stay on WEKA. A successful tiny run must
include finite optimizer steps, policy refresh and actual judge replies; process
exit alone is insufficient. Startup controls do not establish judge calibration
or throughput for a production mixture. The full EP8 judged combination, 32K
judged responses and sustained multi-node performance are untested.

## Policy-engine health

The pinned MILES fork owns the raw-JSON router handling.
Open Instruct uses its native spawn target without a startup hook or subclass.
Generation and health have independent HTTP connection pools: saturating the
generation pool must not prevent probes from reaching healthy engines. Native
regressions cover real HTTP saturation, retirement/re-registration, stale probe
results, retryable 503 responses, cancellation counters and joined shutdown.
Health failure logs include the exception class.

Quarantine remains sticky until explicit worker registration; HTTP recovery alone
cannot prove an engine has current policy weights. Registration epochs reject
late probe results from retired incarnations. Engine replacement is untested,
and this does not enable fault tolerance.
