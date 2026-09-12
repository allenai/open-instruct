# Proposed Core Dolci Think mixed-domain run

Historical proposal, reviewed on 2026-09-12 UTC against Open Instruct
`e103cc7ed`. The user subsequently clarified that **200 updates was arbitrary**;
the immediate goal is a tiny real-data exercise of multi-node training and named
judge placement. Multi-node launching, named judge sections and the strict reward
bridge described as missing below have since been implemented. See the
[current managed-judge guide](miles-managed-judges.md) and its
[two-update configuration](../configs/miles/qualification/multinode-judges.toml).
The 200-update configuration remains an unlaunched scaling proposal; its full
data preparation, EP8 and long-context settings still need qualification.

The remainder records the original production proposal and its then-current gaps. Working assumption: Olmo 3 **Think**, using the recent full-SFT
KDA checkpoint, not SFT1000 or hero. This is an adaptation of the RL recipe;
it does not reproduce the released dense model's SFT/DPO history.

The [candidate run file](../configs/miles/proposals/dolci-think-200.toml) uses the
existing Core run schema and successfully compiles through `plan`. It names
future prepared-data artifacts. Its 15 policy GPUs deliberately exclude the
additional managed judge GPU. The current launcher rejects the two-node
allocation. Judge configuration below is a proposed port, not an already
supported Open Instruct/MILES configuration section.

## What we already have

- [grpo-multitask.toml](../configs/miles/examples/grpo-multitask.toml) is a small
  GSM8K/math example. Separate full-SFT GSM8K/math/IF mixed updates passed the
  [retained mixture audit](measurements/miles-mixture-20260910.json). That EP2/one-engine run used an 8,192-token response cap; the mixture
  guide now links this later evidence.
- Recent judge work is in `/tmp/olmo-miles-managed-judges`, branch
  `docs/managed-judge-guidance`, head `afbdd6f`; `feature/managed-judges` is
  `cc483bf`. The ordinary `/home/robert/proj/olmo-miles` main checkout does not
  contain all this work, so inspecting main alone misses the implementation.
- The [actual Dolci trial](https://beaker.org/ex/01M26E6B1J1QT5XFQF14WA9N8K)
  passed on four B300s: 2 trainer + 1 rollout + 1 Qwen judge, one async replay
  update. Beaker exit zero was independently checked during this review. Its
  batch contained four code, two IF and two reference-judged responses. The
  two real grades were 0.3/0.2, taking 8.85/9.02 seconds without retries. This
  validates real reward delivery, not 200-update learning or judge throughput.
- The baseline already has a proposed 8+7+1 packed two-node config. An earlier
  16-GPU attempt failed before its first update: 32,088 judge-input tokens plus
  2,048 grading tokens exceeded its configured 32,768 context. The corrected
  40,960-context judge passed capacity and actual-data checks on the small layout.
  Sustained two-node managed-judge execution remains unqualified in these records.

## Proposed recipe

Use the complete pinned
[Dolci-Think-RL-7B release](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B/blob/0fb6466d31ef3a9dd16985ef635e6429e05a6491/README.md):
102,014 source prompts, with source proportions preserved rather than equal
sampling from four independent datasets.

| Domain | Source rows | Approximate share | Reward |
| --- | ---: | ---: | --- |
| Math | 30,180 | 29.6% | Existing math verifier with preserved answer targets |
| Instruction following | 29,813 | 29.2% | Per-row constraint checks |
| Code | 21,385 | 21.0% | Function-test and stdio execution, threshold 0.99 |
| General | 20,636 | 20.2% | Qwen3-32B, reference/no-reference rubric selected per row |

Existing baseline preparation already covered the entire release, used the exact
full-SFT tokenizer/template, removed 32 overlong prompts, and held out 16 prompt
identities per domain. Its report contains 101,917 training rows and zero exact
train/eval overlap. Rebuild an immutable Core-compatible artifact with **128
held-out prompt identities per domain** (512 total), retaining the old held-out
identities where possible. Remove all occurrences of held-out identities from
training, preserve intentional training duplicates/source weighting, and report
final row counts. Do not claim the existing 64-entry eval is a 512-entry eval.

Preserve the policy checkpoint's chat template, including its recorded digest
`a3d6a7ad3fde8e26fa9953d80ac30682f7d56ac1d03f3428910a4239a4e23bef` after verifying
it against the actual source. Re-tokenize to prove identity; do not substitute the
released dense Olmo tokenizer or accept the dataset's historical token IDs.
Preserve full judge queries, references and per-row verifier identifiers.

| Control | Proposal |
| --- | --- |
| Placement | Two Holmes B300 nodes; 8 Core EP8 trainer GPUs, 7 TP1 policy engines, 1 fixed judge |
| Batch | 64 prompts × 8 responses = 512 accepted responses/update; one optimizer step/collection |
| Duration | 200 optimizer updates = 102,400 accepted responses; filtering may generate more |
| Scheduling | Bounded async; one-update staleness, buffer factor 2, retry stale groups |
| Objective | LR 1e-6 constant, no KL, GRPO without std normalization, token-averaged policy loss, clip 0.2/0.272 |
| Correction | Explicit TIS, lower/upper 0.5/2.0; record clipping and behavior-policy lag |
| Filtering | Existing native nonzero-reward-variance group filter; qualify mixed-domain refill/exhaustion behavior |
| Replay | Core rollout-router replay; no Megatron-specific `use_routing_replay` flag |
| Length | 2,048 prompt + 32,768 response = 34,816 Core/SGLang context |
| Admission | Candidate 16 requests/engine, 1,048,576 token slots, 128 mamba slots, memory fraction 0.6 |
| Graphs | Decode/prefill graphs initially disabled to match the long-context baseline; enable after a screen |
| Evaluation | Fixed 128/domain, greedy, full response budget, update 0 and every 20 updates |
| Checkpoints | Native Core saves every 50 updates; final HF export; saves are synchronous |
| Checks | Initial/every-20 diagnostic publication checks; scoring check on first/every-50 update; drift failure at 0.05 |
| Tracking | Online W&B, per-domain metrics, retained generations and judge diagnostics |
| Placement policy | urgent, ai2/open-instruct-dev, ai2/oe-other; min runtime 8h |

The 64×8 batch matches the Think-scale recipe; it intentionally exceeds our
8×8 development starter. Sixteen long-context requests per engine is a capacity
candidate, not a measured optimum or a reduction of the short-context default.
Do not automatically carry admission 64 into 32K generation. Increase it after
measuring KV/state memory and tokens/second. EP8, this length, and seven-way
publication all need qualification on the Core path.

Zero-variance filtering changes accepted source proportions, especially when
binary verifiers and noisy continuous judges are mixed. Report proposed,
filtered and accepted counts per source; use no extra source balancing initially.
The filter hook exists in the pinned MILES async buffer and passes through our
config. This is a qualification gap, not an absent native feature. TIS/replay,
greedy held-out evaluation and our auxiliary-loss defaults are deliberate
adaptation choices; exact historical Open Instruct numerical parity is not claimed.

## Judge and execution services

Port the baseline's packed service ownership instead of launching another
training actor. Node A exposes all eight GPUs to Ray; node B exposes seven and
reserves its final GPU for the judge. Verify Ray's actual node ordering and
placement, because Beaker replica rank alone is not proof of trainer ownership.
An 8+8+1 alternative needs 17 GPUs and independently coordinated allocations.

Proposed judge contract (not supported run-file sections yet):

```toml
[judges.general]
mode = "managed"
backend = "sglang"
model = "Qwen/Qwen3-32B"
revision = "9216db5781bf21249d130ec9da846c4624c16137"
gpus = 1
tensor_parallel_size = 1
max_context_length = 40960
max_concurrent_calls = 16
chat_template = "qwen3-no-thinking"
prepared_dir = "/weka/oe-adapt-default/robertb/olmo-miles/trial-data/dolci-think-20260908/judge"

[rubrics.quality]
profile = "open-instruct/general-quality"
max_response_tokens = 2048
temperature = 1.0

[rubrics.reference_quality]
profile = "open-instruct/general-quality_ref"
max_response_tokens = 2048
temperature = 1.0

[judging.bindings."general-quality"]
judge = "general"
rubric = "quality"

[judging.bindings."general-quality_ref"]
judge = "general"
rubric = "reference_quality"
```

[Qwen's pinned configuration](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json)
specifies 40,960 positions without RoPE scaling. Nevertheless, actor-token
counts do not establish judge-token fit. Tokenize the complete rubric, query,
reference and response with the judge tokenizer and reserve its output budget
on every request. Retain failed/truncated responses and stop on operational
errors. A valid zero grade is different from an unavailable judge.

This matters when reusing Open Instruct: `LMJudgeVerifier` currently truncates
oversized evidence and returns zero after exhausted transport errors; the generic
MILES bridge retains score/cost but drops reasoning/error details. The baseline's
strict service wrapper is safer for this experiment. Preserve its exact-context
checks, incomplete-grade rejection, service/driver canaries, endpoint propagation,
health monitoring, teardown, model/template provenance and per-request diagnostics.
Client limits can multiply across processes, so enforce admission at the server
and measure queueing. One B300 judge fits in the baseline; its production capacity
for this workload has not been established.

Code needs a live execution service too. The baseline trial used the existing
Open Instruct function/stdio endpoints and passed positive/negative canaries.
Do not assume the historical endpoint is currently healthy or sized for this run;
recheck both protocols, concurrency, timeouts and error propagation before launch.
Judge and code failures must not silently become negative policy examples.

## Remaining work and acceptance

1. **Two-node launcher plus judge lifecycle:** `open_instruct/miles/launch.py`
   explicitly rejects allocations larger than one node. `judges`, `rubrics`,
   `judging`, and launch coordination are not in the Core run schema. Port the
   proven ownership/supervision shape; do not evade the guard by declaring a
   fictitious 16-GPU node. Verify networking and cross-node weight publication.
2. **Full data/reward bridge:** named tasks only include GSM8K, math, old IFEval
   and multiplication. Baseline manifest adoption rejects code/judge names through
   its finite factory list. Prepared JSONL plus a trusted explicit registry is an
   existing escape hatch; port the Dolci canonicalizer, function/stdio registry
   and strict judge adapter there, preserving full queries/references and source IDs.
3. **Production Core qualification:** EP8 forward/backward, long responses with
   replay, mixed-loss normalization, filter/refill, multi-engine publication and
   the new scorer-skip check need a short real-data screen. No new trainer design
   or Megatron conversion is implied. Keep the first/every-50 scoring gate;
   diagnose a failure or force standalone scoring rather than weakening tolerance.
4. **Observation and restart:** add per-domain accepted/dropped counts, reward,
   lengths/caps, service error/latency/queue metrics and fixed held-out exports.
   Existing raw generations and timing records are useful foundations. Verify
   native checkpoint/cursor/version resume and restart of the coupled services;
   automatic Beaker resume stays off until that lifecycle is qualified.

Use one short 2-trainer/1-rollout/1-judge actual-data check to validate newly ported
services, then a few representative updates on the exact 8+7+1 shape before the
200-update allocation. This need not repeat the old synthetic filler campaign.
All four domains, full configured length capability, health/overflow/error controls,
finite updates, disjoint GPU ownership, policy lag, replay and version refresh
must be represented in the acceptance evidence.

No defensible ETA follows from GSM8K timings. At 10,000 average accepted response
tokens, the nominal training target alone is **1.024 billion response tokens**,
plus rejected groups, evaluation and judge generation. The held-out schedule adds
5,632 policy responses (512 × 11). Source row shares suggest about 20% judge demand
before filtering, but accepted shares may differ. Measure generation, scoring,
forward/backward, publication, judge/code queues and evaluation separately, while
accounting for async overlap.

The file's 48h ceiling is a provisional upper bound, not an estimate or launch
approval: 16 GPUs × 48h is at most 768 allocated GPU-hours for that window.
Reprice after the exact-shape screen; 200 updates might not fit. Check free WEKA
space for four complete native checkpoints, staging, final HF export and retained
rollouts/replay IDs. Retention is not automatically bounded by our current Core
writer. No artifact deletion is proposed.

Local review validation: the candidate CPU `plan` resolves EP8, seven TP1 engines,
64×8/512, 200 updates, the native filter, token loss, replay and TIS correctly.
Calling the launch-spec compiler fails with the expected one-node-only error.
This verifies the proposal's current boundary; it does not prove runtime fit,
data preparation, GPU execution, or endpoint availability.

Review checks: `make style && make quality` passed, as did 104 focused run-spec,
launcher and option tests. The proposal has not allocated GPUs or contacted reward
services, and its prepared paths are still to be created.
