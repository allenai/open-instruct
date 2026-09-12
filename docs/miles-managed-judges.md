# Multi-node runs and named GPU judges

The researcher launcher now compiles disaggregated runs into replicated Beaker
tasks. Trainer GPU count, rollout GPU count and judge GPU count are independent.
The first exercise is intentionally two updates, not a learning run.

```bash
python -m open_instruct.miles plan configs/miles/qualification/multinode-judges.toml
python -m open_instruct.miles run configs/miles/qualification/multinode-judges.toml
```

The checked-in qualification file names account-specific prepared paths. Prepare
its tiny Dolci subset with `scripts/train/debug/miles_judge_preparation.sh` through
the required build wrapper first. That CPU job runs on Saturn, validates the
cached judge identity/template and checks the function/stdio code endpoints.
GPU jobs run urgent on Holmes in `ai2/open-instruct-dev`, with a positive minimum
runtime. The example disables checkpoint saves and export to bound the exercise.

## GPU ownership

`launch.gpus_per_replica` is the physical GPU allocation per replica. For a run
that fits on one node, only the requested GPUs are allocated. Larger disaggregated
runs reserve trainer nodes first, followed by as many rollout nodes as needed.
Managed judges fill spare space on the final rollout node, then additional nodes.
All replicas have the same allocation size; the plan reports any unused GPUs.

| Request | Allocation at 8 GPUs/replica |
| --- | --- |
| 8 trainer + 7 rollout + 1 judge | 2 nodes, 16 GPUs, no unused GPUs |
| 8 trainer + 23 rollout + 1 judge | 4 nodes, 32 GPUs, no unused GPUs |
| 8 trainer + 8 rollout + 1 judge | 3 nodes, 24 GPUs, 7 unused GPUs |

The last case deliberately keeps all eight rollout GPUs. A later heterogeneous
allocation launcher could avoid those unused GPUs. Increasing inference capacity
is already expressible with `inference.gpus`; engines must fit within a node and
tensor parallelism must divide the node capacity. Multi-node colocation, separate
evaluation GPU pools, cross-node serving TP and automatic coordinated restart
remain unsupported. Set `launch.auto_resume=false` for multi-node/managed runs.

Before Ray starts, replicas exchange addresses on WEKA and sort them numerically
as MILES sorts placement bundles. Beaker replica zero is not assumed to own the
trainer. Judge devices are excluded from Ray's CUDA mask and advertised GPU count.
The readiness gate checks the exact live-node GPU layout, not just its sum, and
rejects replicas sharing a physical address. This prevents a judge from receiving
policy weights or a trainer rank from landing on a judge device.

## Named services, rubrics and bindings

The schema follows olmo-miles: `[judges.NAME]` declares serving, `[rubrics.NAME]`
declares grading, and `[judging.bindings.VERIFIER]` maps prepared verifier names to
both. Two rubrics can use one fixed service. Only bound services consume GPUs.
The qualification file demonstrates `general-quality` and `general-quality_ref`.

Managed services currently use the pinned training image's SGLang and a cached,
revision-pinned Qwen model with the qualified `qwen3-no-thinking` template. Model
weights are prepared before GPU allocation. `max_context_length`,
`max_concurrent_calls`, `timeout`, and grading output/temperature are independent
of policy generation settings. Existing unauthenticated OpenAI-compatible services
can use `mode="external"`, `endpoint`, `model`, and the same context/concurrency
settings; their lifecycle and context enforcement remain externally owned.
Independent vLLM allocations, credentialed external judges and custom rubric files
from the broader olmo-miles surface have not been ported yet; unsupported keys fail.

The judge client was ported from olmo-miles `afbdd6f`. It uses the same rubric
text/digests and answer extraction, keeps `metadata.judge_query` and reference
labels, tokenizes the full grading request plus its output reservation, and
rejects overflow without truncating evidence. Incomplete/invalid grades and
exhausted transport errors fail the run. Raw replies, reasoning, rubric/model
identity, retries and latency are retained in `metadata.verifier_diagnostics`.
Named bindings use the sample-aware reward bridge; they never import code selected
by data. Deterministic verifiers keep the existing Open Instruct path. Code uses
the baseline HTTP payload and strict transport error handling.

## Lifecycle and evidence

A unique submission UUID isolates coordination records under
`OUTPUT/cluster/UUID`. Each replica supervises its local Ray/judge process groups
and publishes a heartbeat. Startup and heartbeat timeouts are configured in
seconds through `launch.coordination.startup_timeout` (1200) and
`heartbeat_timeout` (120). Peer failures propagate through shared records and
Beaker's failure/preemption propagation. No Beaker API credential is necessary
for the packed replica group. Normal completion uses a peer acknowledgement before
tearing down Ray. Service failures stop training; no automatic reward-zero fallback.

The driver checks model discovery and known-good versus known-bad answers through
each named binding before training. Inspect `placement-*.json`, `ray-layout.json`,
`judge-canaries.json`, `registry.json`, `complete.json` and `cleanup-*.json`, alongside
Core optimizer/publication/replay contracts and retained rollout metadata.
`driver-RANK.log` and `judge-NAME-RANK.log` stay on WEKA. Tiny exercise success must
include finite optimizer steps, policy refresh and actual judge replies; process
exit alone is insufficient. Startup controls do not establish judge calibration
or throughput for a production mixture.

Implementation status: local ownership/HTTP/lifecycle tests pass; GPU exercise
results will be recorded separately. This does not yet qualify EP8, 32K responses,
large inference pools, or sustained multi-node performance.
