# Multi-node runs and named GPU judges

Distributed disaggregated runs require a Beaker replica group on distinct physical
nodes. Trainer GPU count, rollout GPU count and judge GPU count are independent.
See the [scheduling contract](launching.md#distributed-scheduling-contract)
before submitting a multi-node run.
The initial two-node exercise and later four-update combined async exercise are
recorded below; short execution gates do not establish mixture learning quality.

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
remain unsupported. Restarting a run that carries a managed judge is not qualified: the [multi-node resume qualification](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/multinode-resume-20260922.md) covers a two-replica trainer and engine with no judge in the allocation.

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

For long policy responses, budget the complete grading request in **judge**
tokens: question, candidate answer, optional reference, rubric, chat framing and
the judge's output. A 32K policy response can exceed 40K judge tokens because the
tokenizers differ. Managed Qwen/Qwen3-32B can explicitly request
`context_extension="qwen3-yarn-128k"` with `max_context_length=131072`. This uses
Qwen's documented factor-4 YaRN configuration (original context 32768), passed as
a serving override without rewriting prepared weights. The native default stays
unchanged; unsupported models, double scaling and limits above 131072 fail.
This changes judge computation, so compare runs using the same judge setting and
qualify long grading requests before using it for a benchmark. See the
[qualification record](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/learning-confidence-20260914/README.md).

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
for the packed replica group. Liveness uses SGLang's non-generating `/health` mode and three consecutive
failures; actual grading requests retain their own hard deadlines. Normal completion uses a peer acknowledgement before
tearing down Ray. Service failures stop training; no automatic reward-zero fallback.

The driver checks model discovery and known-good versus known-bad answers through
each named binding before training. Inspect `placement-*.json`, `ray-layout.json`,
`judge-canaries.json`, `registry.json`, `complete.json` and `cleanup-*.json`, alongside
Core optimizer/publication/replay contracts and retained rollout metadata.
`driver-RANK.log` and `judge-NAME-RANK.log` stay on WEKA. Tiny exercise success must
include finite optimizer steps, policy refresh and actual judge replies; process
exit alone is insufficient. Startup controls do not establish judge calibration
or throughput for a production mixture.

Implementation status: the two-node, two-update EP2 mixed-task exercise passed
with the native MILES router and both named judge bindings; its retained-sample
audit also passed. See the [results](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/multinode-judges-20260912.md).
The later [EP2 + three engines + one judge exercise](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/colleague-20260913/README.md)
also passed async/TIS, replay/packing, saves and actual prefix-cache hits. Neither
establishes the full EP8 judged combination, 32K judged responses or sustained
multi-node performance.

## Policy-engine health

The pinned MILES fork now owns the raw-JSON router fix ported from olmo-miles.
Open Instruct uses its native spawn target without a startup hook or subclass.
Generation and health have independent HTTP connection pools: saturating the
generation pool must not prevent probes from reaching healthy engines. Native
regressions cover real HTTP saturation, retirement/re-registration, stale probe
results, retryable 503 responses, cancellation counters and joined shutdown.
Health failure logs include the exception class.

Quarantine remains sticky until explicit worker registration; HTTP recovery alone
cannot prove an engine has current policy weights. Registration epochs reject
late probe results from retired incarnations. This fix does not independently
qualify engine replacement or enable fault tolerance. The bounded named-judge
exercise and its failed attempts are recorded in the
[measurement note](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/multinode-judges-20260912.md).

## IF gate-yes reward

The opt-in `open-instruct/if-gate-yes` rubric binds an IF verifier name to a
composite reward: binary prompt-level loose constraint pass P multiplied by one
only for a parsed `yes`; `partial` and `no` multiply by zero. P requires every
constraint to pass at least one of the eight IFEval loose response variants,
using the same instruction registry and checker functions as `ManifestIFVerifier`.
Its fractional constraint score C remains a separate diagnostic. Every eligible
IF response is judged, including P=0. Truncated responses and missing/empty final
sections retain the bridge's pre-scoring zero gates and are not judged.

Use this registry entry in the trusted `reward_config` JSON:

```json
{"ifeval": {"factory": "open_instruct.miles.rewards.if_gate_yes.IfGateYesVerifier", "config": {"name": "ifeval"}}}
```

Keep any other registry entries, including maths, unchanged. Bind `ifeval` to
`rubrics.gate_yes` and the managed H066/H086 service:

```toml
[rubrics.gate_yes]
profile = "open-instruct/if-gate-yes"
max_response_tokens = 16
temperature = 0.0
seed = 1
max_concurrent_calls = 32

[judging.bindings.ifeval]
judge = "general"
rubric = "gate_yes"

[launch.env]
OI_MILES_JUDGE_FAILURE_POLICY = "raise"
```

The profile validates Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`,
SGLang TP1, YaRN context 131072, `qwen3-no-thinking`, server concurrency 16,
timeout 600, and H066 production's prepared directory (see `rewards/h080.py`).
The client seed is independent of the training seed. Both context tokenization
and completion have a literal 600-second timeout, unaffected by the infrastructure
timeout multiplier. Context overflow never truncates the evidence and remains fatal.

The vendored H080 prompt is checked against SHA256
`133a5e5448fff28da6b848efefb303fd7bd51dc27d37981e48054db90555cb36`.
The complete role-labelled conversation and extracted final text enter it
verbatim; parsing full-matches `Verdict: (yes|partial|no)\.?` case-insensitively
after stripping the reply. There are three attempts total. The IF adapter defaults
to `OI_MILES_JUDGE_FAILURE_POLICY=raise`: exhausted transport or parsing failures
halt scoring, including when P=0, with failure diagnostics and no quality zero.
Every failed attempt is logged, including grades from groups later filtered out.

Prepared rows use raw `metadata.query` for H070 single-turn inputs. For other
conversations, preserve raw structured turns in `metadata.judge_conversation`.
Preparation re-renders these turns with the policy template and requires exact
equality to the policy input, preventing silent loss of earlier turns. Neither
prompt nor final-answer whitespace is normalized by the judge adapter.

Every new training server must replay the sealed H086 controls before its ready
marker is published. This reads `oe-eval-default` and requires exact 540/540
parsed-label identity, pinned manifest/labels, ids/hashes, and identical prompts.
Submission requires a `launch.weka_mounts` entry with `weka = "oe-eval-default"`
and `mount_path = "/weka/oe-eval-default"` for these controls.
The report lives under `OUTPUT/cluster/ATTEMPT/h087-qualification-*.json`.
For standalone qualification inside the image, run
`python scripts/miles/h087_qualify_adapter.py --out /output/qualification` on one
GPU; it uses the same managed server command and adapter client. This does not
replace qualification of the eventual training server process.

Existing `rollout/general_judge/*` metrics describe consumed samples; skipped
samples are counted separately. H087 also uses one CPU Ray actor to count all
logical judge requests, including filtered groups. Each worker admits at most
32 logical requests on its local event loop with an asyncio semaphore shared
across clients. Cancellation releases admission locally; counter updates are
fire-and-forget and never hold scoring slots while waiting for the metrics actor.
Retries stay within the same logical request; `attempts` counts
grading attempts (including failures during context checks). `completed` includes all terminal requests and `failed`
is a subset. At each optimizer boundary the driver logs `h087_judge_boundary`
JSON with cumulative arrivals/completions/failures/skips, queued/in-flight counts,
backlog, attempts, latency, and step duration. The first record's startup t0 runs
from cluster-bootstrap entry to the first completed optimizer step. This hook
requires one optimizer update per collection, as in H087. Native dynamic-filter
and refill diagnostics remain in the full driver log.
