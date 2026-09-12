# Internal colleague exercise campaign, 2026-09-12

Candidate starts at primary Open Instruct `9403772d1`; runtime dependency revisions
are pinned by `runtime/miles/runtime.lock.json`. Execution branch:
`robertb/miles-colleague-exercises`. This note distinguishes planned coverage from results.

## Synthesis and limits

At most ten GPU attempts including retries, plus one shared CPU preparation job
on Saturn. GPU jobs use urgent Holmes, open-instruct-dev, 1h minimum runtime.
No recovery, H100, hero or full-model colocation claims transfer from other trainers.
The existing EP8 100-update run covers unpacked asynchronous training/checkpointing;
the existing radix A/B covers a different mixture without managed judges.

| Case | Exercise | Dependency / decision |
| --- | --- | --- |
| 1 | Dense 1 trainer + 3 engines, math/IF, sync, two optimizer steps/collection, KL and warmup; persistence off | First wave, four collections |
| 2 | Full SFT MoE EP2 + 2 engines + 1 judge; six domains, packing/replay, cold cache | First wave after CPU preparation, four collections |
| 3 | Case 2 fresh allocation, same weights/data, warm compiler cache | Only after case 2 publishes compatible artifacts |
| 4 | Case 2 with radix extra_buffer and cache-aware router | Audit case 2 first; existing A/B is supplementary |
| 5 | Async, mixed chunks, long inputs/responses | Actual token lengths required; configured limits do not count |
| 6 | Two nodes: EP8 + 7 engines + judge, mixed async packed/replayed data | Case 2 services/numerics must pass |
| 7 | Full hero EP2/TP2 numerical checks | Existing failed scoring gate is retained; no training unless it passes |
| 8 | Dense colocated FSDP save/restart/export/reload | Fit/lifecycle gate before claiming support |
| 9 | MoE async + packing + replay fresh-process checkpoint continuation | Prioritize ahead of optional long-context case 5 |
| 10 | Bounded failure or isolate first unexplained failure | Never disrupt shared services; count retries against ceiling |

Cases 1–4 have concrete TOMLs under `configs/miles/qualification/colleague-20260912`.
Remaining cases are deliberately not pre-submitted: resolve the first-wave findings
before multiplying them across longer contexts or larger topology.

## Gates and evidence

Separate lifecycle, numerical/data correctness, learning-path coverage, and specific
feature coverage. Require nonzero advantages/gradients and parameter changes for
learning coverage; short-run held-out score lift is not an acceptance requirement.
Retain exact config, source/image, sample artifacts, policy versions, replay contracts,
startup/stage timings, cache keys and hit reports, service outcomes and exit codes.
A failed canary or numerical gate blocks its dependent jobs, not unrelated work.

Shared CPU preparation validates source hashes, checkpoint shard inventories,
concurrent tokenizer imports into a fresh shared module cache, code known-answer and
execution-timeout canaries, both judge descriptors, and normalized train/eval prompt
and source-identity separation. It prepares four balanced collections per fixture
using each checkpoint's own pinned HF chat template. Source token IDs are explicitly
replaced and their hashes retained. This is a coverage fixture, not recipe parity.
A code HTTP rejection must not pass as a legitimate wrong-answer canary.

## Current evidence

- Initial host suite: 255 passed (config, launch, data, workflow, judges, input
  validation, packing configuration, code retries, scoring decisions, compiler cache).
- Runtime replay/packing actor tests cannot run in the laptop environment: it lacks
  the pinned MILES and Core MoE modules. Run these in the candidate image.
- Draft validation caught save-interval/disabled-save and correction-value mistakes;
  fixed before submission. Dense multi-update collections also explicitly declare
  `core.max_policy_lag=1` as required by the policy-age contract.
- GPU attempts used: 0. No new lifecycle or learning claim yet.

### Candidate A / first preparation

Source `af7aa8c1e99b`, image `01M2BX0VPAB4XDWTRVQNJP8BES`.
Host tests 255 passed plus 3 fixture tests. Image runtime tests 94 passed, 2 skipped.
Fresh dependency-free Python 3.12 validates all four initial configs.
Saturn preparation `01M2BX13SSQNY91TWM39GWP562` exited 1: the requested two held-out
rows per domain exceeded the one eligible general-quality row. Dense preparation,
all six code canaries, concurrent imports and inventories succeeded. No GPU launch.

The preparation-only correction explicitly uses one eval row per MoE domain and
hash-validates an already completed sibling fixture before reuse. This requires a
second short CPU preparation job beyond the original one-job estimate; GPU budget
remains untouched. Training source and runtime dependency pins are unchanged.

### Candidate B / first GPU wave

Source `c143bea53fc1`, immutable image `01M2BX9JPT44DGQ9QWY5C0HF52`.
Saturn preparation retry `01M2BX9TAJD31D652BVSDP7DF4` passed, exit 0: 48 dense
training prompts + 4 eval; 64 MoE training prompts + 6 eval; six successful code
canaries (correct/wrong in both formats, syntax error and bounded execution timeout).
The original successful dense fixture was verified without modification.

GPU attempt 1: dense case 1, `01M2BXECF7QDB19Z2F9071W6K9` (4 GPUs).
GPU attempt 2: cold mixed MoE case 2, `01M2BXF4PWKTPKNP7GYKHN1HZP` (5 GPUs).
Both submitted through the documented CLI/wrapper, urgent Holmes, 1h minimum,
no auto-resume. Both started; training results pending. Eight GPU attempt slots
remain. Cases 3–4 are staged but not submitted.

Launch receipts, both preparation reports and test logs are retained under
`docs/miles/measurements/colleague-20260912/`.

Local quality follow-up: repaired two existing type errors in code-service
program-length diagnostics without changing scoring behavior. Eight focused
code/fixture tests, full Open Instruct formatting/lint and type checks passed.
This small follow-up is not baked into the currently running image.

## Expanded EP and inference-pool coverage

The user authorized larger GPU allocations and additional EP/engine coverage.
Retain the original readiness cases while adding a controlled sizing comparison:

| Trainers | EP | TP1 policy engines | Physical nodes / GPUs | Purpose |
| --- | --- | --- | --- | --- |
| 8 | 8 | 8 | 2 / 16 | Packed/replayed baseline |
| 8 | 8 | 16 | 3 / 24 | Does a second inference node reduce trainer starvation? |
| 8 | 8 | 24 | 4 / 32 | Staged follow-up only if 16 still starves the trainer |
| 8 | 4 | 8 | 2 / 16 | Same trainer GPU count, different EP grouping |

Each has 12 collections of 64 prompts x 8 responses, global batch 512, the same
immutable math/IF/function-code/stdio-code mixture, seed, weights, 4096 response
cap, 6144 pack/context budget, replay, async lag 2, TIS, 64 requests per engine,
and cache-off serving. Initial/final eval remains enabled. Save/export are off
for timing. These are shape/throughput exercises, not learning comparisons.
The judge mixture remains a separate test: judge grading can impose an independent
bottleneck, so this sweep cannot by itself prescribe the judged-mixture pool.

Compare warm training/publication time, generation wait and its tail, consumed
response tokens per second, per-engine admission/queue/load, lag and discarded
work, peak memory, and total allocated GPU-hours per consumed token. Keep startup,
compilation and evaluations separate. Phase busy fraction is not SM utilization.
If 12 collections do not contain enough warm observations, mark sizing provisional.

A useful initial decision rule is the smallest pool within 10% of the best measured
warm cadence, with generation wait below 10% of the cycle and acceptable lag/loss
contracts. Also report the fastest option regardless of cost. Fit a rough production
rate versus trainer demand estimate, then check it against actual multi-node runs;
do not extrapolate linear engine scaling or assume 8 trainer GPUs imply 8 engines.
Keep batch/group size fixed while sizing. Extra concurrency, mixed chunks, radix,
and alternate EP are separate comparisons rather than simultaneous confounders.

### First-wave finding

Case 2 failed before training. Its judge was healthy, but Ray GCS could not be
reached at the advertised physical host address. Inspection found that the launcher
used host networking only for multiple replicas, while a single-node judge run
also enters the cluster bootstrap that advertises host IPs. Enable host networking
for all cluster-bootstrap jobs, with a single-node judge regression test. Retry
uses a fresh output root. This is a launcher fix; the existing pinned training
image can be reused because networking is encoded in the submitted Beaker spec.
Case 1 has reached generation. No full learning/lifecycle pass yet.
