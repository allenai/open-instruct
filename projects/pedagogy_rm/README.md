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
3. **Label, then measure per-dimension agreement.** Drop any dimension whose
   raters cannot agree; keep the rest. This is a decision point, not a formality.
4. `extract_hidden.py` — OLMo hidden states for each labelled turn.
5. `probe.py` — MLP per surviving dimension, reported against the agreement
   ceiling rather than against zero.

Nothing after step 3 is worth writing until step 3 has a number.

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
