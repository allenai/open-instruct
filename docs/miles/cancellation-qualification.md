# H045 positive-pressure cancellation: qualification run

Prepared 2026-10-12 for Robert and another agent to review. **Not launched.**

The question is whether this teacher signal can produce any useful learning on
questions H045 failed in a saved 0/8 screen. This is one treatment compared with
unchanged H045 SFT, not a benchmark or an optimal-configuration search. It does
not run GRPO first. A null result can disqualify this configuration if updates
were active and stable; it cannot disprove every possible teacher objective.

## Concrete mechanics

1. Freeze a deterministic split of the existing 57 H045 0/8 questions: 40 for
   training and 17 held out from these updates. No new screening or hint search.
2. Evaluate unchanged H045 independently on all 57 questions, four new responses
   each. The original selection screen is **not** the baseline: another draw from
   an unchanged model can succeed after an observed 0/8.
3. Per update, draw eight training questions and generate two fresh, unaided
   student responses each. Sampling is temperature 1, top-p 1, no top-k cutoff.
4. A frozen H045 teacher scores the exact generated token IDs twice, once with
   the full prepared R1 solution and once with just the final answer. Both use
   the same instruction wrapper. There is no teacher generation or text splicing.
5. Compute the token pressure below, detached from autograd. Feed it through the
   existing MILES/Core OPD advantage path and optimizer. Outcome rewards are
   logged but have zero training advantage. Keep later successful samples too;
   the cohort was gated once, not repeatedly.
6. Publish updated student weights synchronously and generate the next batch
   afresh. No stored trajectory replay, early stopping, prefix mask, or queue.
7. Save every four updates; evaluate again after update 16, with the same
   questions, sample count, cap, sampling settings, and verifier.

The two contexts share one teacher GPU and frozen weights. They are scoring
conditions, not separate models. On the initial policy version only, an extra
bare-context score measures numerical disagreement with generation-time student
scores. It does not affect the learning objective.

## Exactly what we teach

For the sampled token at position t, let:

```
F = log p_full(token | student prefix) - log p_behavior(token | student prefix)
A = log p_answer(token | student prefix) - log p_behavior(token | student prefix)
s = F - min(max(F, 0), max(A, 0))
advantage_t = beta * s
```

The full and answer teachers see their respective reference contexts **and the
same student prefix**. These are sampled-token log probabilities, not a
full-vocabulary KL calculation. Positive s encourages the sampled token;
negative s discourages it. No group standardization or advantage whitening.

| Full boost F | Answer boost A | Retained s | Effect |
|---:|---:|---:|---|
| +7.2 | +8.0 | 0 | Cancel overlapping encouragement |
| +5.4 | +4.1 | +1.3 | Keep encouragement beyond answer-only |
| −0.9 | −1.0 | −0.9 | Keep the full teacher's correction |
| +2.0 | −1.0 | +2.0 | Do not amplify encouragement by subtracting a negative |

The native `opd_reverse_kl` field transports `-s`; native MILES subtracts
`beta * opd_reverse_kl` **once**. In this mode that field is a transport for shaped
pressure, not a KL divergence. Native metrics using that name must not be
interpreted as a genuine KL. Core's pre-update student scores remain the PPO
ratio anchor; they do not replace the recorded generation-time behavior scores
used above. Clipped PPO-style likelihood ratios still constrain the update.

This is an OPD-inspired shaped objective, not an unbiased estimator of ordinary
reverse-KL distillation. Positive cancellation does not prove the remaining
pressure represents mathematical reasoning. Negative style penalties remain,
and the two contexts can interact nonlinearly. We are testing the usefulness of
this particular signal.

## Data and teacher provenance

Student: `h045-anchor-emo-olmo35-s34521/hf_step11768`, under
`/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-4t/runs/`.

Teacher: the frozen H045 serving export at
`/weka/olmo-3p5-checkpoints/scratch/robertb/open-instruct/runs/bread-h045-validation-20261009-v2/prepared/hf`.
Before launch, check its export provenance against the student checkpoint; local
recipe validation cannot read these remote weights. The initial bare-context
diagnostic is an additional behavioral check, not a weight-identity proof.

The full reference is all saved `metadata.bread.sections`, joined in order: the
prepared OpenR1 DeepSeek-R1 reasoning trace. It is **not** the BREAD hybrid of an
R1 opening plus H045 continuation, and it is not the source `solution` field.
Reference index, source identity, hashes, and token counts are preserved.

Both reference conditions use this wrapper inside the original user prompt:

```
{question}

Reference material:
<reference_solution>
{full R1 trace OR "The final answer is {answer}."}
</reference_solution>
Use it to help solve the problem. Continue the current reasoning coherently;
another valid approach is allowed.
```

Student and evaluation inputs retain the original bare question/template. The
verifier answer lives in metadata/label, not their input. Evaluation rows contain
neither privileged prompt. The training-question diagnostic is explicitly marked
`data.in_sample_eval`; genuine holdout overlap remains forbidden. The 17 heldout
questions are held out from **this training**, not a pristine research benchmark:
the pool has already supported exploratory work.

## Decisions for review

| Decision | Starting choice | Reason / alternative if observation warrants it |
|---|---|---|
| Optimizer | Existing AdamW, LR **1e-6**, constant, no warmup, weight decay 0 | Reuses the historical H045 OPD starting scale. A short run should receive its intended LR immediately. Consider 2e-6 in a follow-up if mechanics are healthy but policy movement is negligible. |
| Teacher coefficient | **beta = 1.0** | Give the retained signal a real opportunity. In this pure objective, uniform rescaling can largely cancel in Adam's adaptive normalization; beta is not interchangeable with LR. Clipping, epsilon, and optimizer state complicate that cancellation. |
| Pressure clipping | **Off** | Preserve the agreed negative corrections and inspect tails. Optional symmetric clipping is implemented; ±2 nats is a possible follow-up if spikes dominate, not a claim of an optimal threshold. |
| Gradient clipping | Global norm **1.0** | Retain the existing protection. Inspect pre-clip norm and clipping frequency; continuous saturation undermines an LR/pressure comparison. |
| Reduction | Native per-token loss, no whitening, no extra KL or entropy term | Keep the signal's sign and relative magnitude visible. Long responses contribute more token terms; this is an explicit starting choice. |
| Batch / duration | **16 updates**, 8 prompts × 2 responses | 256 fresh training responses, 128 question visits: about 3.2 visits per training question on average. A small learning probe, not a single gradient nudge. |
| Generation cap | **16,384 tokens**, including truncated responses in loss | Truncated responses receive observed reward zero but retain teacher pressure. No 32K/64K generation expansion. |
| Evaluation | Independent baseline and update 16; 4 responses/question | Spend on a before/after comparison. Checkpoints 4/8/12 permit targeted follow-up evaluation without routine intermediate evaluation cost. |
| Teacher refresh | Never within this run | Avoid a moving teacher when diagnosing whether the retained signal learns. |
| Collection | Synchronous publication | Isolate learning behavior first; asynchronous gating/queues remain later engineering. |

Inspect the first two updates for finite nonzero pressure, gradients, valid
weight publication, and sensible generation. If the mechanics fail, stop and fix
them. If they work, complete the bounded run rather than choosing a checkpoint
by a lucky early answer. No automatic LR escalation or configuration sweep.

## Instrumentation and interpretation

- `teacher-scores.jsonl`: response token IDs, full/answer/behavior log probabilities,
  exact retained pressure, policy versions, sample IDs, observed rewards and finish
  status. This supports later token/span analysis and length/position splits.
- `teacher-metrics.jsonl`: token counts, positive pressure removed/retained,
  negative pressure retained, zeros/sign counts, squared pressure and largest
  absolute pressure. Initial bare-context gaps are recorded separately.
- Teacher timings distinguish admission wait, scoring-request latency, and
  per-response end-to-end wall time. Sums of overlapping requests are not GPU
  compute time. Native pipeline metrics cover collection/training/publication.
- `qualification-evaluation.jsonl`: per-response text, answer reward, finish
  status, length, dataset and weight version. Native rollout archives preserve
  training generations; sample logging is 100%.
- The report command checks an initial policy version of zero, duplicate records,
  and matched per-question before/after sample counts. It reports sample accuracy,
  fraction of questions with any success, finished-only accuracy, truncation,
  mean length, and paired question changes, separately for training and holdout.

Watch the fraction of full positive pressure cancelled, retained positive versus
negative magnitude, pressure RMS/tails, and how these evolve with policy version.
If nearly everything is removed, or negative pressure dominates, that is an
important result about the proposed objective. Compare initial numerical gaps
with the signal scale before attributing tiny boosts to information.

The primary desired sign is improved **unaided answer correctness**, with
training-question performance as the sensitive qualification diagnostic and the
small holdout as a transfer check. Shorter outputs alone mean learning to stop,
not learning more mathematics. Improvements confined to training questions may
reflect memorization. A correct answer does not establish a sound derivation;
inspect changed examples, especially new successes and premature finishes.

Four samples per question are noisy. One new success is a lead to reproduce,
not a demonstrated effect. An independent baseline addresses selection bias
from the 0/8 screen but not all sampling uncertainty. If there is no improvement,
distinguish an inactive/underpowered update from a meaningful negative result:
check pressure, gradients, publication and policy movement first. The objective
need not win a benchmark to qualify for further study.

## Bounded resource plan and efficiency

Proposed launch after review: **ai2/holmes, one node, five B300 GPUs**, split two
Core trainer / two student inference / one frozen teacher. Priority **high**,
unallocated (`min_runtime=0`), execution timeout **2 hours**, internal training
budget 6,600 seconds. Historical frozen-teacher integration requires
`auto_resume=false`; checkpoints every four updates preserve evidence, but a
preemption needs an explicit, provenance-aware restart. No duplicate auto retry.

New response-token ceilings (not predictions):

- Training: 16 × 16 × 16,384 = **4,194,304**.
- Evaluation: 2 × 57 × 4 × 16,384 = **7,471,104**.
- Total: **11,665,408**, plus prompts and teacher scoring. Teacher scoring reads
  responses twice, plus references; the initial bare diagnostic adds one pass on
  initial-policy samples. These extra passes are not generated tokens.

This uses the existing trainer and pinned runtime. The first efficiency pass
retains BF16, sequence packing, activation recomputation, FlashAttention 4 in the
trainer, the existing optimizer, inference CUDA graphs/compiler cache, and
bounded parallel teacher admission. Eight serving slots per inference GPU match
the 16-response training batch. We avoid fresh 0/8 screening, repeated rescue
search, and routine intermediate evaluations.

No new GPU throughput or memory result is claimed. At startup, verify the actual
attention/optimizer configuration, sustained train and teacher-score throughput,
peak memory, and generation occupancy. Record startup/compilation, evaluation,
checkpoint/export and steady-state costs separately. The two-hour timeout is a
ceiling, not a promise of 16 completed updates; an interrupted run is not a null
learning result. Teacher prefills or evaluation may dominate this small run.

## Review artifacts and checks

Implementation lives on `robertb/h045-opd-cancellation-qualification`, based on
historical H045 revision `a5ce054b9`, in
`/tmp/open-instruct-h045-opd-qualification`. The stable review package is
`/home/robert/proj/open-instruct/runs/h045-opd-cancellation-qualification-20261012`.
It contains frozen JSONL inputs, manifest/hashes, local and container recipes,
compiled plan, this note, and a container build context. No Beaker job is created
by preparation or validation.

Local objective/data/configuration/documentation tests: **183 passed**. CPU integration in the
existing pinned H045 runtime: **12 passed**, exercising exact token alignment,
truncated-token retention, native OPD gradient direction, single coefficient
application, evaluation logging/restoration, and the initial numerical-control request. Ruff checks pass. GPU
forward/backward, end-to-end data loading, remote checkpoint identity, throughput
and memory still require validation on the launch hardware.

From the qualification checkout, use the project environment via `uv run`:

```bash
uv run python -m open_instruct.miles validate /path/to/review/run-local.json
uv run python -m open_instruct.miles.distillation.qualification_report /path/to/downloaded/output --output /path/to/report.json
```

The container recipe uses `/opt/h045-qualification-data`; use the prepared image
overlay so those files actually exist. Build/publish/launch remains a separate
step after review. Do not point this recipe at an older image merely because it
contains a trainer: it must contain the cancellation hooks and data changes.
