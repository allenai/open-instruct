# H045 SPORK-64 cancellation qualification: overnight continuation

Authorized by Robert on 2026-10-12 after the initial SFT qualification. The goal
is a stronger test of solving improvement with less opportunity to gain solely
by stopping earlier. This remains one qualification treatment, not a benchmark.

## Starting point and objective

Student and frozen teacher both use the **same** final HF export from
`h045-answer-exit-treatment-20261010`, update 64. Startup checks the completed
Core checkpoint manifest and exporting workflow. A fresh optimizer is created;
the SPORK auxiliary objective and GRPO task advantages are off. The existing
MILES/Core trainer and pinned runtime remain in use.

Keep the previous cancellation objective, LR 1e-6, beta 1.0, no token-pressure
clipping, gradient norm limit 1.0, BF16, 16K cap, 8 prompts × 2 fresh responses
per update. Both teachers are scoring contexts of the same frozen model: full
prepared R1 trace and answer-only. Teaching pressure uses **current-student prefill**
log probabilities from the student inference fleet, compared with frozen teacher
prefill scores. Generation-time behavior probabilities are preserved separately
for importance correction; this synchronous qualification recipe still has TIS
disabled. Preserving the behavior scores does not itself enable that correction.
Scoring requires a single recorded policy version and
rejects a student response with a missing or different version. Synchronous
publication holds weights fixed until rollout scoring completes. The fixed-text
panel follows this matched-prefill objective; decode-based pressure is retained
only as a counterfactual diagnostic.

## Pool and screening

There are 57 locally extracted initial-SFT 0/8 questions; this is not a large
verified SPORK-64 failure pool. On the worker, inspect the original 16,384-question
prepared OpenR1 training pool and prioritize identities with historical BREAD
all-zero gates, including later policy versions. Old gates prioritize candidates;
they do not establish current eligibility. Exclude original heldout identities,
deduplicate questions, and retain only complete references fitting teacher context.

Screen at most 512 candidates, with an approximately 90-minute admission budget
(an admitted wave can finish beyond that). All five allocated GPUs serve the same
SPORK checkpoint for this stage. Generate two attempts first; any success rejects
the question immediately. Otherwise generate six more. Eligibility requires
**0/8 reward and at least 6/8 normal, finished responses**. Every candidate's
attempts, finish status, seeds, rewards, and verifier diagnostics are retained.
Verifier timeouts cannot establish eligibility and stop screening for inspection.

Target 96 training and 32 heldout questions; deterministic split after screening.
Accept a smaller cohort down to 32 training and 16 heldout; below 48 total eligible
questions, preserve data and stop without training. Do not relax the finish rule.
All-zero truncation-heavy questions are retained separately for later evaluation,
not included in this run's headline cohort. The population is "usually finishes,
but failed this screen," not established "confident-wrong" or truly unsolvable.

The heldout cohort is disjoint from this distillation training, not from the
SPORK model's earlier training pool. This is not a clean generalization benchmark.
The new independent baseline is essential: 0/8 screening is not zero true success
probability. No selection uses outcomes from baseline or post-training evaluation.

## Learning and evaluation

Up to **64 updates**, checkpoints every 8. Evaluate at policy versions **0, 16,
and 64**, with **16 fresh responses per question**. Evaluation contains all up to
32 heldout questions and a fixed 16-question training diagnostic, selected before
baseline. At the maximum cohort this is 768 responses per evaluation, 2,304 across
three evaluations. Routine native calls at updates 32/48 are explicitly skipped.

Primary outcome: unconditional correctness on the fixed eligible question cohort.
Also report pass@1/4/8/16 estimates, accuracy among finished responses, truncation,
length, paired question changes, and gains/losses. Finished-only accuracy is a
diagnostic because the population finishing can change. Rising truncation can be
an adverse effect; a null on this selected pool is a narrower claim than failure
on all zero-reward questions. Training and heldout results stay separate.

An eight-question scoring panel (four training, four heldout), two traces per
question, supplies two measurements at the same checkpoints:

1. **Fresh text on fixed questions:** full-reference, answer-only, frozen-bare,
   current-student prefill, and current-student decode scores. It shows how the
   signal changes along the model's actual generated paths. No outcome-based
   choice of traces.
2. **Fixed baseline text:** score the exact saved prefixes/tokens under the updated
   student. Measure log-probability changes separately on originally encouraged
   and discouraged tokens and their pressure-weighted sum. Report matched-prefill
   pressure as the training objective and behavior-score-based pressure as a
   counterfactual diagnostic.

An unchanged aggregate teacher gap is not proof of no learning: new generations
visit different states, and the objective deliberately removes some full-teacher
encouragement. The fixed-text panel is the more direct test of whether an update
followed the signal. Neither metric alone establishes mathematical improvement.

## Numerical checks and observation

For precision-sensitive scoring, `teacher.isolate_prefill=true` configures both
the frozen teacher and student engines with `--prefill-max-requests 1`. This
limits prefill batching, not decode concurrency. The same setting is applied to
the screening workers. It is an explicit experiment option, not a GRPO default;
qualify its performance together with the shared FLA kernel profile. Without
isolation, a shared profile fixes autotuner choices but does not fix the way
different requests share the prefill token budget.

Before screening, generate one 1,024-token probe, score it on the same server,
then on an identically configured second server. Both servers load the identical
export. Save all scores and signed/absolute differences. Engineering guardrails
stop before training if mean absolute decode/prefill gap exceeds 0.1 nats or
cross-server prefill gap exceeds 0.005 nats. These are mismatch guards, not
calibrated thresholds for useful privileged information.

The fixed question panel subsequently measures bare-context scoring differences
on real evaluation traces. Only baseline compares identical weights; after
updates the frozen-versus-current bare difference includes intended learning.
All original token pressure, gradients, policy versions, generations, finish
status, and request/collection/training/publication timing remain recorded.

Raw teacher score and screening JSONL are compressed into Beaker result artifacts;
panel arrays are saved as compressed JSONL. This avoids the previous collector's
32-MiB per-file omission. Binary native rollout archives and checkpoints remain
on WEKA; evaluation texts and token-score logs are exportable without another
GPU generation pass. Preemption can prevent final compression, so durable WEKA
files remain authoritative.

## Resources and bounds

**One Holmes node, five B300 GPUs, high priority, unallocated, eight-hour execution
timeout.** Screening uses five inference servers, then releases them before native
training allocates two trainer / two student-generation / one teacher GPUs.
Training receives the remaining wall-time budget with a ten-minute cleanup reserve.
No automatic resume: the inherited frozen-teacher path requires explicit restart.
Do not duplicate screening or silently restart from SFT after a preemption.

Response-token ceilings: screen 512 × 8 × 16,384 = 67.1M (short-circuiting and time
budget usually reduce this); training 64 × 16 × 16,384 = 16.8M; three maximum-size
evaluations 3 × 48 × 16 × 16,384 = 37.7M. These are ceilings, not expected usage or
a promise of completing every stage in eight hours. Teacher prefill and diagnostics
add scoring cost. If the wall limit ends training early, report actual steps and
available evaluations rather than interpreting it as a completed null experiment.

Reuse the already exercised trainer optimizations: BF16, packing, activation
recomputation, FlashAttention 4, serving graphs, and compiler cache. Screen batches
are bounded; successful questions terminate after the initial pair. No new hint
search, teacher generation, or training objective arm is added.

The bootstrap is selected explicitly with `launch.env.OI_MILES_OPD_OVERNIGHT=1`
and runs through the standard committed-image MILES launcher. It writes the frozen
screened inputs and resolved training recipe before starting the native cluster.
