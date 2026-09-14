# Colleague-readiness exercise plan

> Point-in-time proposal. Recorded defaults, branch names and then-pending work are historical; consult the [current support matrix](../feature-parity.md) and current examples before launching.

Historical plan. Current outcomes are in the [readiness measurements](../measurements/colleague-20260913/README.md); use the [MILES guide](../index.md) for supported workflows.

Proposed September 12, 2026, against `robertb/miles-olmo-core` at `875d972f1`.
This is a plan, not new qualification evidence. No experiments are launched by
this document. Freeze one candidate image and the resolved run files before
execution; record any later fixes as a new candidate.

## Scope and budget

Use eight core GPU experiments, with two slots reserved for a failure isolation
or an important uncovered interaction. Count failed launches and retries against
the ten-attempt GPU budget. Do not spend the reserve automatically. One shared
CPU-only WEKA preparation job on Saturn is additional and explicitly accounted
for; host/schema checks require no GPU allocation.

The purpose is to catch first-use failures and consequential integration errors,
not establish learning curves, exact trainer parity, or exhaustive compatibility.
Most experiments need 4–6 collections, at least three publications, and a small
initial/final evaluation. Set `skip_eval_before_train=false` explicitly and use
the same held-out items at both endpoints. Make the workload representative before increasing the
number of updates. Stop an ordinary allocation around 90 minutes and a long or
multi-node allocation around two hours; these are planning budgets, not measured
runtime guarantees. Report incomplete coverage instead of silently extending.

Reuse existing qualified results when image, architecture, topology and relevant
controls match. In particular, review the in-flight math/IF/code radix A/B work
before commissioning overlapping experiments. Its existing 20-update EP8 arms
can supply evidence, but do not stand in for a managed-judge or replay/packing
combination they did not actually exercise.

## Lessons carried over from olmo-miles

The issue threads contain historical failures and later corrections. They are
regression scenarios, not a claim that all defects remain in either integration.

| Evidence | Failure class to exercise here |
| --- | --- |
| [Issue 7](https://github.com/allenai/olmo-miles/issues/7), items 1–6, 12, 22 | Fresh checkout/dependencies, launching from a Beaker session without local Docker work, multiple WEKA mounts, TTL cache paths, user-selectable secrets, discoverable supported tasks and native option mapping. |
| Issue 7, items 7, 9, 10, 14, 15, 19 | Checkpoint descriptor and precision compatibility; invalid combinations rejected before GPU work; concurrent model/tokenizer imports; probability checks with enough tokens and per-layer detail. Do not relax a failed gate until the cause is understood. |
| Issue 7, items 11, 13, 16, 21 | Several changing batches rather than one repeated tiny shape; preparation outside the large allocation; checkpoint finalization, storage/read timings, actual fresh export reload. The reported DeepGEMM failure and async-save mechanism were Megatron-specific; Core needs equivalent boundary coverage, not those exact patches. |
| Issue 7, items 17, 20 | Multi-node readiness, replaced-job status, bounded failure and group cleanup. Do not claim automatic coordinated restart where our launcher rejects it. |
| [Issue 8](https://github.com/allenai/olmo-miles/issues/8) | Verify initial evaluation, ID/content train–eval separation, reward signal, actual lengths, and sufficient requests per engine. More GPUs alone did not make their workload efficient; admission mattered. |
| [Issue 9](https://github.com/allenai/olmo-miles/issues/9) | Representative mixed tasks, long responses, async policy lag, and more than one optimizer step per collection. Keep a readiness smoke separate from a learning-quality experiment. |

## Cheap work before the GPU matrix

1. Exercise the documented install, `plan`, `validate`, `run` rendering and
   `status` from a fresh checkout/environment. Also render an immutable-image
   launch from a Beaker-session-like environment without starting Docker. Use
   the ordinary CLI and documented wrapper, not an unpublished helper.
2. Validate path/mount coverage for checkpoint, prepared data, output, compiler
   cache and judge weights across both `oe-training-default` and
   `oe-adapt-default`. Check actual read/write access in preparation. Confirm
   named credentials can be supplied by another user; do not print their values
   or borrow a colleague's credentials for the exercise.
3. Run existing contract/config tests and small table-driven invalid-input cases:
   GPU/TP/EP divisibility, colocation counts, too-small context/pack budgets,
   unsupported trainer offload/debug modes, dense router replay, KDA cache
   capacity/strategy, malformed judge bindings, and unsupported multi-node
   auto-resume. Each must fail with a specific remedy before expensive work.
4. Stage pinned model descriptors/tokenizers and source revisions once. Check
   architecture/precision metadata and complete shard inventories. Time reads
   of checkpoint/export files; an unexpectedly slow read is an I/O finding, not
   proof that pre-reading repairs storage. Stress simultaneous config/tokenizer
   imports in fresh process-local caches; one successful import is not a race test.
5. Prepare immutable, tokenized training/eval manifests. Check source IDs and
   normalized prompt content for overlap, including across mixture components.
   Record dropped/over-budget rows and reasons. Render the chat template once;
   preserve judge references and code tests without leaking answers into prompts.
6. Verify code and judge service canaries: correct, wrong, malformed, timeout and
   service-unavailable cases. Distinguish a program failing a test (legitimate
   zero reward) from an unavailable evaluator (infrastructure error). Use the
   existing configured execution service; managed GPU judges do not themselves
   provide a code execution environment.
7. Verify compiler-cache fingerprint reuse and deliberate invalidation with
   metadata/unit checks. A changed compatibility key must miss. Reuse previous
   live invalidation evidence unless the keying implementation changes; this
   does not need another full RL run.

Known capability constraints come from [run controls](../run-controls.md),
[managed judges/topology](../../miles-managed-judges.md), and
[packing semantics](../../miles-sequence-packing.md). Parser acceptance alone is not a
claim that a model fits or that a backend combination has been exercised.

## Shared fixtures

- **D:** published dense Olmo 3 7B Think-DPO, original pinned RL template from the
  successful smoke. Use it for one-trainer/three-engine and FSDP coverage. Do not
  assume the larger MoE optimizer fits one GPU.
- **M:** the already-used full-SFT 18.5B KDA/latent-MoE checkpoint with the
  BF16-storage/FP32-router-compute contract. Start from its qualified descriptor,
  not a silently edited source checkpoint.
- **H:** actual non-EMO small hero: 16 blocks, 14 KDA/two full attention, latent
  width 512, 512 experts/top-16, plus its attention gains/scaling. Use the pinned
  native/HF pair already audited. A tiny hero-shaped model is supplementary
  evidence, not a substitute for the full checkpoint.
- **S:** a compact math/IF slice with varied lengths and a separate small held-out
  set. Include harder math: the recent two-update dense GSM8K run solved all 16
  training responses and therefore supplied no gradient signal.
- **J:** a balanced prepared six-task slice: math, IF, function-code, stdio-code,
  general-quality, reference-quality. Reuse the existing prepared-data/verifier
  and named-judge machinery. Ensure every type actually reaches a training step;
  being present in the source manifest is insufficient.
- **L:** separate long-input and long-output buckets, including repeated 4–8K
  prefixes with different suffixes and hard math/code expected to generate long
  responses. Include cases near configured context/admission boundaries.

Pin the same initial weights, seed, examples, order, batch and eval settings for
matched comparisons. Frozen inputs and forced continuations provide numerical
comparisons; free generation is allowed to diverge with serving batch schedules.

## The eight core experiments

`T` means trainer GPUs, `I` policy inference GPUs, `J` judge GPUs. These are GPU
ownership counts, not node counts. Capacity must pass the rendered plan and a
peak-memory preflight before full work starts.

| # | Model and topology | Workload and controls | Main evidence sought |
| --- | --- | --- | --- |
| 1 | D; one node, **1T + 3I**, three TP1 engines | S; sync, compiler persistence **off**, radix off, unpacked; 12 prompts × 4 samples, global batch 24 gives two optimizer steps/collection. Small nonzero KL coefficient and short LR warmup exercise reference scoring and scheduling. | Ordinary colleague launch; actual requests on all three engines; live gradients/updates; reference model stays frozen; exact step/publication counts and forced standalone scoring. |
| 2 | M; one node, **2T/EP2 + 2I + 1J** | J; sync; **empty compiler-cache namespace**, radix off, packing and router replay on; 16 × 4 samples, global batch 64; 4K response cap. Both judge rubrics bind to one pinned service. | Cold startup, packed/replayed training, both code payload forms, both judge bindings, immutable inputs and service ownership. |
| 3 | Exactly #2, fresh allocation and process state | Identical frozen image/model/config and inputs; new output identity; restore the compiler artifacts published by #2. Start from original weights, not its RL checkpoint. | Cold-versus-warm reuse across allocations. Check the actual compatibility keys/hit status per worker; distinguish serving/trainer/import/compile timings. |
| 4 | Exactly #3 topology/model/data | Enable KDA radix `extra_buffer` and cache-aware routing; preserve packing, replay and judge controls. Use explicit capacity satisfying `slots > 5 × running_requests`, with memory headroom. | Radix × replay × packing × judges/code. This compares the production radix/routing bundle; it is not an isolated estimate of router-policy speed. Verify real prefix hits and post-publication invalidation. |
| 5 | M; **2T/EP2 + 2I + 1J** | J + L; radix/replay/packing retained; bounded async lag 1 or 2 with TIS; mixed prefill/decode chunks enabled. Exercise up to 8K prompts and 16K responses with matching context and judge budgets; reduce concurrent requests to fit. | Interaction stress: long inputs/outputs, recurrent cache, packing boundaries, overlapping generation/training/publication, and backpressure from slow graders. This is not a causal speed A/B against #4. |
| 6 | M; **two nodes: 8T/EP8, then 7I + 1J**, 16 GPUs total | J, async, packing, replay, radix; 32 × 4 samples then one bounded 64 × 8 collection if memory/time permit. Moderate 4K cap; enough admitted work for seven TP1 engines. | Real multi-node startup, ownership/isolation and collective schedules; workload-size variation; no idle engines caused by a hidden admission cap. Keep `auto_resume=false`, as required for managed/multi-node runs. |
| 7 | H; one node, **2T/EP2 + 2I/TP2** (one serving engine) | First frozen Core/HF/serving scores, prompt branching/prefix reuse and publication checks; only then 3–4 short real collections if numerical gates pass. Short math/IF plus fixed long-boundary probes; compiler cache on, radix on, replay on, packing off. | Full hero geometry, router precision, gains/scales, TP2 and cache compatibility. Existing full-hero probability qualification failed; preserve a failed/blocked outcome rather than weakening the gate or substituting tiny-model results. |
| 8 | D; **two colocated GPUs**, Core FSDP2 and two TP1 engines sharing them | S; short contexts and explicitly bounded KV allocation because Core remains resident. Radix on, compiler persistence on, tracking off/offline; save after a live update, continue a fixed next-step probe, then restart trainer processes from that save and replay the probe. Export and load in a fresh serving process. | Dense radix does not inherit KDA restrictions; FSDP plus live IPC publication; native optimizer/scheduler/RNG/cursor restore; actual exported-weight reload. Up to three process segments within this one allocation, explicitly timed. |

For #2–4, include an identical small frozen long-prefix request set before training
to obtain real cache-hit opportunity. Use the same query order and enough requests
to cover branching, a partial prefix hit, eviction, and fresh policy versions.
Score identical continuations with cache off/on using a predeclared numerical
bound; do not expect free-running answers to be byte-identical.

For #5, perform a short frozen serving check with mixed chunks off/on before the
integrated stress phase (fresh engine processes if that startup flag requires it).
This isolates a numerical failure without allocating two
additional training jobs. Record those two phases explicitly; do not count the
whole stress experiment as proof that each individual switch improved speed.

For #8, native continuation comparison uses the same saved RNG state, exact token
batch and objective, same topology and qualified deterministic arithmetic. Check
optimizer moments, schedule/cursor and the next-step log-probabilities/weights.
If bitwise reproducibility is not available, record the measured discrepancy and
its cause; do not call it exact resume. Fresh serving export reload is separately
compared against the pre-export policy on fixed inputs. A completion marker or
presence of safetensors files is not sufficient.

Keep other controls fixed unless the table names them. Use activation
recomputation, dynamic-row kernels, token loss reduction and flattened publication;
M/H use fused expert publication. Keep dense decode graphs off initially, and
use the already-qualified decode-graph setting for M in #2–6. Prefill graphs stay
off. The H numerical gate starts with graphs off. Record actual flags in each
case; compiler-cache persistence and CUDA graphs are different controls.

## Two contingency slots

- **9, preferred if the core matrix passes:** M on one node, EP2 plus engines,
  checkpoint/restart through the async boundary. Use a dedicated test judge/code
  endpoint or test proxy to delay/fail one request; never interrupt shared
  production services. Verify bounded retries or explicit job failure, no silent
  zero reward, no duplicate consumed groups, and clean teardown. Exercise a
  trainer-resume boundary separately from continuing an in-flight rollout queue;
  only claim the restart semantics the implementation supports.
- **10:** isolate the first unexplained failure with one changed control, repeat
  an intermittent concurrent-start case, or cover 32K output/greater-than-16K
  actual training sequences if #5 never reaches them. Choose based on the first
  eight reports. No failure is obliged to consume this slot if documenting a
  clear unsupported combination is sufficient for colleagues.

If an earlier attempt fails, spend these slots there first. A failed hero gate
counts as #7 coverage with a known limitation; it does not authorize an open-ended
hero-debugging campaign or block sharing the qualified older-MoE/dense profiles.

## Acceptance: three distinct outcomes

Every row records **passed**, **failed with a reproducible cause**, or
**not exercised/unsupported**, independently for these layers:

1. **Lifecycle:** parse/preparation/startup, several collections, publication,
   evaluation and shutdown complete; allocated GPUs and endpoints match the plan.
2. **Numerics and data:** real groups/masks/versions are valid; required expert
   replay agrees on covered tokens; standalone/training scores satisfy the
   established gate; no nonfinite losses/gradients; references/judges never receive
   actor weight updates; reward recomputation and source/held-out identity pass.
3. **Learning-path coverage:** at least two optimizer updates have nonzero
   gradients and measurable parameter changes on a full model. All-right/all-wrong
   groups are legitimate, but an all-zero-advantage run only passes lifecycle.
   Choose a bounded alternate harder slice when needed, with its own manifest;
   never alter rewards merely to manufacture apparent learning.

For #1, multiple optimizer steps exercise the old-policy ratio path, but clipping
need not occur at a small LR. Use an independent fixed-ratio loss fixture to test
both clipping bounds rather than demand a nonzero empirical clip fraction or
raise the LR until clipping occurs. Record the same distinction for KL and TIS.
Do not require positive held-out lift in six updates or declare regressions from
one changed answer in a 16-question evaluation.

Long-sequence coverage is measured from retained tokens, not configured caps:
record prompt, response and total-token distributions separately; require actual
training samples beyond 8K input and 8K response for the corresponding #5 claims.
Record greater-than-16K total samples if obtained. Explicit overflow cases should
be rejected or counted by a documented admission policy, without silent truncation
of prompts, code tests or judge evidence. If a quota is unmet, report that gap.

## Measurements and notes to keep

Write one `case-NN.md` and one compact JSON report, indexed from a single coverage
table. Each note should contain:

- Exact command/config, image and dependency pins; checkpoint/tokenizer/template
  hashes; source manifest; endpoints/rubric identities; seed and attempt lineage.
- Actual trainer/engine/TP/EP/node/judge ownership, usable request capacity, maximum
  GPU memory, and whether each engine and each task contributed trained samples.
- Startup split into preparation, reads/imports, serving warmup/graphs, trainer
  construction, compiler restore/publish, initial weight publication and eval.
  Mark compile misses/hits per role and distinguish cold from warm timings.
- Per-collection real tokens, live-advantage groups, cap hits, score-gap
  distributions, gradient/update norms, lag/TIS metrics, packing fill, cache hits,
  and exact completed optimizer/publication counters. Include MoE per-layer load
  extremes: a pathological expert group need not have an unusual total token count.
- Generation, scoring, training, weight-publication, checkpoint write/finalize,
  evaluation, grading/execution latency, retries, and driver idle time. Include
  GPU-hours (judge GPUs too). Short runs produce descriptive ranges, not stable
  throughput estimates; do not infer compile cost solely from total startup time.
- Save size/completion time, export/read/reload time, and actual restore outcome.
  Keep at most a small declared number of checkpoints per case and report bytes.
- A few successes/failures/capped generations per task, raw judge/code outcomes,
  and a concise symptom → workaround → supported boundary note for each problem.

Use the existing retained-data auditors, `training_contract_rank*.jsonl`,
`driver_timing.jsonl`, publication/cache reports and W&B rather than introducing
another dashboard. Audit all retained samples in these small runs, including
that both judge bindings and both code formats actually supplied trained rewards.

Add a colleague-facing matrix with rows such as “dense 1T+3I”, “KDA radix + replay
+ packing”, “long-input grading”, “hero TP2”, and “native resume”. Link each claim
to its case and make unknowns visible. Put tested starter commands and the supported
task/source table near the workflow entrypoint, not only in measurement notes.

## Ordering and stopping

Preparation precedes all GPU allocations. Run #1 and #2 first; #3 follows #2's
cache publication; #4 follows their audits. #5 and #6 build on the services and
cache path, while #7 and #8 can proceed independently once their inputs are staged.
Do not increase concurrency, length and node count to debug the same failure.

Use urgent Holmes in `ai2/open-instruct-dev`, normally with a one-hour minimum
runtime for full-model allocations. CPU-only WEKA preparation/audits use Saturn.
The existing committed-image build wrapper remains the launch path. Use bounded
health/readiness timeouts and job-event evidence; test replaced-leader/status
logic with fixtures instead of deliberately breaking a cluster node. Automatic
coordinated multi-node recovery, trainer offload, arbitrary serving backends,
and every EP×TP pairing are explicitly outside this pass.
