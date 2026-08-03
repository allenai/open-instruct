# tutor — training a teacher against a frozen student

**This directory is one project's specifics.** Everything reusable lives in
[`open_instruct/scored_rewards/`](../../open_instruct/scored_rewards/README.md);
nothing here is imported by anything there. It is here as a worked example of a
reward that no verifier can express, and because its history is a useful
warning.

The task: a teacher LLM sees a question a frozen student cannot solve and writes
hints. The student replies. Nobody trains the student — it is the environment.
The reward is a judged, multi-dimensional score of the teacher's turns, with a
rule-based veto on giving the answer away.

```
a problem the student fails alone, with the student's opening line in the prompt
  -> the tutor writes a turn
  -> the frozen student replies; its next intent is picked from a closed set
  -> three turns
  -> a judge scores each tutor turn on six dimensions, in context
  -> the leak rule vetoes any dialogue that gave the answer away
  -> each dimension is z-scored inside the group of G dialogues and averaged
  -> GRPO
```

And separately, never in the reward: **the anchor**, which asks a frozen student
to actually answer held-out questions.

---

## Read this before you run anything

Four previous runs of this idea reduced leaking every time and **never improved
teaching.** `teacher_acc` — the held-out anchor — has not moved across two
reward configurations, two corpora, and a corrected leak rule.

An earlier draft of this file blamed the student's capacity — expert tutoring
shifts the 0.5B's answers by only about +0.07, so no reward computed from its
accuracy can carry more signal than that. That ceiling is real, but it was not
the whole story, and `why_flat.md` later measured the rest of it:

| correlation with the student solving | r |
| --- | --- |
| rated tutoring **quality** | **−0.012** |
| rated **leakage** | **+0.291** |

The best-rated turns had the *lowest* solve rates. So the outcome does not
contain the target: it has one channel, information transfer, and the only way a
tutor can move it is to give the answer away. Penalising leakage does not add a
second channel — it removes the one that existed, which is exactly the observed
history of leak rate falling while `teacher_acc` sits still.

That reframes the next experiment. It is not "the student, not the reward" — it
is that **the outcome measure has to admit a pedagogy channel before any reward
built on it can find one.** A bigger student alone does not do that, and it
costs headroom: the corpus is screened to items a 0.5B fails, and a large model
solves those unaided, leaving no gradient at all.

**Run `transfer_probe.py` before another training job.** It asks whether tutoring
about problem A moves the student on a *different* problem B that needs the same
knowledge unit — a channel leakage cannot reach, since A's answer is worthless on
B. It re-scores dialogues that already exist, so it needs no tutor generation and
no training, and it answers usefully either way: flat means the environment has
one channel and the honest move is to change the outcome or write up the negative
result, while positive means there is a pedagogy signal `teacher_acc` cannot see.

---

## Files

| | |
|---|---|
| `leak.py` | the leak rule. Four signals computed, two wired in, calibrated against 1,104 rated turns |
| `rubric.py` | six judged dimensions plus a completion gate, and the views the judge is shown |
| `student.py` | the student — a `Director` that drives the partner model, and a log-prob answering channel for the anchor |
| `plugin.py` | registers the `tutor` group scorer and the `tutor_student` environment |
| `build_dataset.py` | screened problems → RLVR rows |
| `run_anchor.py` | the held-out measurement, outside the reward |
| `units.py` | knowledge-unit decomposition, annotated once offline — the only thing that says two items test the same skill |
| `transfer_probe.py` | does tutoring on A move the student on a different item B sharing A's unit? Inference only |
| `scripts/` | serve the two frozen models, smoke, train |

---

## Running it

```bash
# 1. rows. --held-out PA holds out a whole state: different test, different
#    authors, different year. A random slice of one corpus measures nothing.
python -m projects.tutor.build_dataset \
    --items data/state_tests/train_items.jsonl \
    --out data/tutor_train.jsonl \
    --split-key state --held-out PA --eval-out data/tutor_eval.jsonl

# 2. the cheapest integration test: tiny policy, stubbed judge, no endpoints
./projects/tutor/scripts/smoke.sh

# 3. the two frozen models
./projects/tutor/scripts/serve_env.sh

# 4. the anchor BEFORE training. If you skip this you have no comparison.
python -m projects.tutor.run_anchor --items data/tutor_eval.jsonl \
    --tutor-model <base> --tutor-url http://localhost:8003/v1 \
    --student-url http://localhost:8001/v1 --out anchor_before.json

# 5. train
PARTNER_URL=http://localhost:8001/v1 JUDGE_URL=http://localhost:8002/v1 \
    ./projects/tutor/scripts/train.sh

# 6. the anchor after, stated in standard errors
python -m projects.tutor.run_anchor --items data/tutor_eval.jsonl \
    --tutor-model output/tutor_v5 --tutor-url http://localhost:8003/v1 \
    --student-url http://localhost:8001/v1 \
    --out anchor_after.json --compare-to anchor_before.json
```

**Run the ablation first.** `--group_scorer tutor_leak_only` is the leak rule
alone: no judge, no endpoint, no GPU beyond the policy. It is the term with
independent calibration behind it, and it is the only part of this reward that
has ever moved a policy.

**Screen your items.** The reward has a gradient only on problems the student
fails alone. An unscreened attempt opened at 97% solved with three quarters of
its groups contributing no gradient at all; screened, the same setup ran at 27%.
`build_dataset.py` does not screen — it assumes the input is the kept set.

**Sizes.** Training wants 300–500 items; below ~200 the tutor sees each problem
6–13 times and learns per-problem hints instead of how to tutor. The held-out
set wants ~200, where the minimum detectable difference drops below ~0.07. And
250 steps is plenty: every run here plateaued by ~150 and one degraded after
that.

---

## What changed in the move from the standalone stack

The previous implementation was a hand-rolled GRPO trainer, then a TRL
`rollout_func`. This is the same experiment on open-instruct.

| before | now |
|---|---|
| `interfaces.py` `RewardModel` protocol | `scored_rewards.Scorer` / `GroupScorer` |
| `rewards.LeakGuard` | `guards.Veto` + `leak.py` |
| `rewards.SpecificityGuard` | `guards.Contrast` + `guards.ItemPool` |
| `rewards_multi.group_advantages` | `aggregate.normalize_then_sum` |
| `judge_rm.TeachingJudge` | `judge.Judge` + `rubric.py` |
| `reward_model.TeachingScorer` | `head.LinearHead` |
| `student_state.py` | `student.StudentDirector`, a `partner_env` `Director` |
| `trl_env.TutoringRollout` | `partner_env.PartnerModelEnv` |
| `trl_train.AnchorEval` | `anchor.Anchor` + `run_anchor.py` |
| in-process vLLM + HF student + HF judge on one H100 | policy in the trainer; student and judge behind endpoints |

Four things genuinely changed, not just moved:

**The student's opening line is generated offline, at dataset-build time.**
open-instruct discards the observation an environment returns from `reset`, so
the partner cannot speak first at runtime. Baking the opener into the prompt is
strictly better: it is now identical across the G completions of a group, which
is the condition PEARL states for the group to be a comparison between tutors
rather than between students, and it costs no GPU during training.

**The specificity contrast draws its foreign problem from a pool, not from the
group.** In the old trainer the group spanned one problem and the swap was
arranged by hand. Under open-instruct a group is G samples of one prompt, so
every member carries the same item — swapping inside the group would have
returned each member its own problem and the whole term would have silently
been zero. `guards.ItemPool` hands the group one foreign item, fixed across the
group so it survives mean-centring.

**Normalise-then-sum happens inside the reward.** Its output has zero mean
within the group, so open-instruct's default `centered` advantage step is a
no-op and the arithmetic is exactly MO-GRPO's estimator. Keep
`--advantage_normalization_type centered`.

**Everything is built lazily from a spec string.** The reward config is pickled
into the vLLM Ray actors and a judge holding an HTTP client does not survive
that trip.

### The thing that got worse

open-instruct's GRPO is **full-parameter**. `use_peft` is declared in
`model_utils.py` and referenced nowhere in `grpo_fast.py`, so there is no LoRA
escape hatch. A 3B policy needs weights, gradients, optimiser state and
activations resident beside vLLM, which does not fit on one 80GB card next to an
environment and a judge — and it is why the previous version was on TRL. The
options are a smaller policy (0.6–1.5B on one card), or more than one card with
the student and judge served elsewhere. `scripts/train.sh` assumes the latter.

---

## What the reward deliberately does not contain

Whether the student got the answer right.

PEARL's reward is entirely judge-based and their headline evaluation is that
same trained judge, which is how one of their dimensions moves 50.0 → 95.0 on a
scale the judge was trained to score. If the judged reward here climbs while
`anchor/treated` stays flat, the judge is being gamed, and that is visible
within one eval cycle only because the anchor is outside the reward.

Two more separations worth keeping:

`clean_leaked` **next to** `treated`. A rising outcome with a flat clean subset
is not teaching improving — it is leaking falling, and the leaked solves
disappearing from the numerator. That decomposition is what every run here
turned on.

`specificity` **next to** `gain`. In the most careful measurement: no hint
0.489, a hint for a *different* problem 0.495, the tutor's own hint 0.552. So
91% of a +0.063 gain was question-specific. The help is real. It is just small,
and training did not grow it.

---

## Known open issues, inherited

- **Teaching has not improved in four runs**, and the objective is no longer the
  prime suspect. Both objective-side fixes have been tried: a dense learned
  score (it read *provenance*, not quality, and drove the question rate from 79%
  to 32%) and removing the leak rule's false positives (precision on maths more
  than doubled; the anchor moved +0.008). What remains is the environment.
- **Leak rates before and after the rule correction are not comparable.** A
  narrower rule flags fewer turns whatever the policy does. The anchor's
  `treated` is comparable throughout, and it has never moved.
- **The leak rule still misses semantics.** `_content` drops tokens of two
  characters or fewer, so numeric golds collapse to their units; and 46 of 77
  known misses share no content word with gold at all. No string rule reaches
  those.
- **The rule and the information-theoretic probe still disagree on level.** They
  agree on direction in every run and part company on magnitude. Unresolved.
- **A different framing for the head is a different head.** If you use
  `head.LinearHead`, the `view` function must render exactly the string the head
  was fitted on. Nothing about the output looks wrong when it does not.

---

## Tests

```bash
python -m unittest projects.tutor.test_tutor -v
```

32 tests: the leak rule's calibrated cases, the student's mastery update and
closed intent set, the judge views, the episode split, and the scorer end to end
with a stubbed judge. No GPU, no endpoints, no torch.
