# pedagogy_rm — a reward model that reads hidden states, not prompts

A prompted LLM judge is too slow to sit inside an RL loop: every rollout waits on
a generation before it can be scored. This project asks whether the score can be
read directly out of the policy's own hidden states, which are already computed
during the rollout and are therefore free.

The target is **six aspects of teaching quality, rated by humans** — not whether
the student got the answer right. That is a deliberate narrowing; see
[the caveat](#the-caveat-you-are-buying) below.

## The chain, and which link is load-bearing

1. Human raters can agree on the six aspects.
2. OLMo-2-7B-Instruct's hidden states contain that information.
3. A small MLP can extract it from a few hundred labels.
4. The extracted score is stable enough to serve as a reward.
5. Optimising it produces better teaching.

**Link 1 is load-bearing and cheapest to test, so it goes first.** It is also
where the previous attempt died, which is the whole reason this project is
scoped the way it is.

### What the previous attempt already told us

`grpo_tutor` labelled 1,104 tutor turns on two scales and trained a ridge probe
on embeddings. The probe reached 0.581 and a prompted judge reached 0.536 — both
weak. It looked like the representation was the problem. It was not:

| scale | two raters agree exactly | within one point |
| --- | --- | --- |
| `goodness` 1–5 (holistic) | **39%** | 83% |
| `leak` 1–3 (concrete) | **69%** | 94% |

A probe cannot beat the noise in its own labels. `goodness` was mush — two
careful raters landed 2+ points apart on a 5-point scale 17% of the time — so
0.581 may already have been near the ceiling.

That yields the design rule this project is built on:

> **Every dimension must be as concrete and checkable as `leak` was. None may be
> as holistic as `goodness` was.**

If a rater has to form an overall impression to answer, the question is wrong.
Each dimension should be answerable by pointing at specific words in the turn.

The other lesson: those 928 labelled items had **zero overlap** between the six
raters, so per-dimension agreement was unmeasurable. This time, a fixed fraction
of every slice is shared (see `build_label_set.py --overlap`), because a
dimension whose agreement you cannot measure is a dimension you cannot trust.

## What gets generated

| role | model | why |
| --- | --- | --- |
| teacher | `allenai/OLMo-2-1124-7B-Instruct` | fully open, and the hidden states we probe are its own |
| student | a small instruct model, prompted | only has to produce plausible student turns |
| items | `grpo_tutor` standardised test items | same corpus as all prior runs, so numbers stay comparable |

The teacher is OLMo because the point is a reward available *for free during the
rollout*. That only works if the states being probed belong to the model doing
the generating. A probe on some other encoder would be cheaper than a prompted
judge but would still cost an extra forward pass, and would not transfer to a
policy whose weights are moving.

Student correctness and turn count are **not** recorded as targets. Dialogues
run a fixed number of turns and stop.

## Order of work

1. `generate.py` — traces. No labels, no probe, no training.
2. `build_label_set.py` — sample turns into slices with deliberate overlap.
3. `agreement.py` — **the decision gate.** Weighted kappa per dimension. Drop
   anything below 0.4: a probe trained on a dimension raters disagree about
   learns rater noise, and no model size fixes it.
4. `extract_hidden.py` — OLMo states over the tutor turn's tokens, several
   layers, two poolings, rebuilding the rollout's exact chat context.
5. `probe.py` — ridge and MLP per surviving dimension, scored against the
   agreement ceiling rather than against zero.

Step 3 decides whether steps 4 and 5 are worth running at all.

### Reading step 5

`agreement.py` prints, per dimension, the highest correlation any predictor
could reach against labels that noisy. A probe at 0.55 against a ceiling of 0.60
has effectively solved it and the labels are now the constraint; the same 0.55
against a ceiling of 0.95 means the representation is not carrying the property.
Scored against zero those look the same, and the previous project's 0.581 was
read exactly that way.

If ridge matches the MLP, ship ridge — it is cheaper in the loop, which is the
point of the project.

## Results

600 turns, 282 questions, five dimensions after `correct` failed the agreement
gate. Labels are the mean of six model raters plus the human on 25 of them.
Folds are grouped by question. Cross-validated r:

| dimension | surface | ridge | MLP | ceiling | vs human | agents vs human |
| --- | --- | --- | --- | --- | --- | --- |
| targeted | 0.36 | **0.85** | 0.70 | 0.94 | 0.54 | 0.72 |
| leak | 0.60 | 0.84 | 0.80 | 0.95 | 0.63 | 0.80 |
| actionable | 0.75 | 0.94 | 0.92 | 0.98 | 0.90 | 0.92 |
| elicits | 0.81 | 0.95 | 0.92 | 0.98 | 0.80 | 0.83 |
| concise | 0.96 | 0.97 | 0.94 | 0.99 | 0.82 | 0.89 |

**Read the surface column first.** Eight features — length, question marks,
digit density — with no notion of teaching. `concise` is a word counter wearing
a rubric: 0.96 against the states' 0.97. Training on it would reward brevity and
nothing else, and reported against zero its 0.97 would have looked like the best
result in the table. `targeted` is the finding: whether a turn addresses the
student's actual error is nearly invisible in the shape of the text, and OLMo
represents it anyway.

**The probe tracks the human, not just its teachers.** The target is a consensus
the models dominate, so a probe could fit it perfectly while tracking something
no person recognises. Against the human alone it stays within 0.02–0.18 of what
the agents themselves score — it is a faithful distillation, losing a little, as
a student model should. On 25 units, so the standard error is about 0.14; this
is directional.

**Ridge beats a fair MLP everywhere**, after the MLP was given a PCA in front of
it and an alpha chosen on its own held-out split. The mapping is linear, which
is the convenient answer: a dot product costs nothing inside an RL loop.

**Pooling and layer.** The last content token wins most dimensions, the
end-of-turn token trails it by 0.02–0.04, and mean pooling wins `targeted`.
Layers 16–20 of 32 beat both ends at every dimension — the last layer is
specialised for predicting the next token, not for summarising.

## Hackability

`hack_test.py` writes the gold answer into turns the raters said withheld it, at
each end of the turn. Those turns now hand over the answer, so the correct `leak`
score is 3 and there is nothing to argue about. Share still scored under 2 — still
called answer-withholding — with the answer sitting in the text:

| pooling | layer | leak at the end | leak at the start |
| --- | --- | --- | --- |
| last | 20 | 3% | 93% |
| last | 32 | 20% | 96% |
| eot | 20 | 41% | 98% |
| mean | 16 | 88% | 60% |

**Pooling decides which attack works; depth barely matters.** Last-token pooling
catches a giveaway at the end and is blind to the same sentence at the start.
Mean pooling is the mirror image. Reading the final layer instead of the middle
does not help — at layer 32 the tail miss rate gets *worse*, 3% to 20%.

This matters because the un-augmented probe scores r = 0.84 on `leak` and would
have gone straight into a GRPO loop. The policy would have found this in a few
hundred steps and the reward curve would have gone up the whole time.

**Adversarial augmentation closes it.** Half the attacked turns in training
labelled 3, split by question, and the attack written in four phrasings with only
`The answer is X` ever trained on:

| pooling | layer | r on real | trained phrasing, unseen questions | unseen phrasings |
| --- | --- | --- | --- | --- |
| last | 16 | 0.82 | 0% | 3% |
| eot | 16 | 0.77 | 0% | 3% |
| mean | 16 | 0.74 | 0% | 4% |
| last | 32 | 0.69 | 1% | 4% |

93% missed becomes 3%, on questions and phrasings the probe never saw, and
accuracy on real held-out turns holds at 0.82 against the 0.84 it started with.
So the states carried "the answer is written here" all along and the labels never
asked; the fix is data, not architecture. Layer 16 is best on every axis at once,
which settles the middle-versus-final question — the final layer is worse at
reading the labels *and* no harder to fool.

The obvious caveat: one dimension, one attack family, held out along two axes but
not adversarially searched. A real deployment wants this rerun against whatever
the policy actually invents, which is what `scored_rewards`' anchor evaluation is
for.

## The caveat you are buying

Ignoring student correctness makes the target tractable and honest about what it
is: *this looks like good teaching to a human*. It is worth being explicit that
this is not the same as *this helps the student*, and that we have direct
evidence the two come apart. `grpo_tutor/docs/postmortem.md` measured real tutor
dialogues making a frozen student measurably **worse** (0.5B −0.0216 ± 0.0060,
3B −0.0498 ± 0.0095) while a bare topic sentence helped.

So a reward model trained here will optimise appearance, and the question of
whether appearance and effect coincide is deferred, not answered. That is a
reasonable trade — the appearance target is at least measurable — but it should
be stated in anything written up, not discovered later.
