# scored_rewards

Rewards that are a **score** rather than a verdict, for open-instruct's GRPO.

open-instruct's reward path is built for verifiable rewards: a rule reads one
completion and says whether it is right. That covers maths, code and instruction
following. It does not cover the tasks where no rule can decide — was this a
good explanation, a fair negotiation, a useful interview question — and it does
not cover environments whose other participant is a person.

This package adds five things. Everything is optional: with none of the new
flags set, `grpo_fast.py` behaves exactly as it did.

| | what it gives you |
|---|---|
| `registry` | your reward lives in your repo, loaded with `--reward_plugins` |
| `types` | `Scorer` (one completion) and `GroupScorer` (all G at once) |
| `guards` | veto, gate, contrast, and per-dimension normalisation |
| `judge` / `head` | a generative rubric judge, or a learned head on a frozen backbone |
| `partner_env` | the environment is another model |
| `anchor` | a metric computed **outside** the reward |

---

## Quickstart

Write a scorer anywhere in your repo:

```python
# my_rewards.py
from open_instruct.scored_rewards import register_fn

@register_fn("brevity")
def brevity(sample):
    """Shorter is better, up to a point."""
    words = len(sample.completion.split())
    return max(0.0, 1.0 - abs(words - 60) / 60)
```

Point the trainer at it:

```bash
python open_instruct/grpo_fast.py \
    --reward_plugins my_rewards.py \
    --group_scorer brevity \
    ...
```

`--reward_plugins` takes dotted module paths or `.py` file paths, comma
separated. Importing the module is the whole registration protocol — scorers,
environments and `VerifierFunction` subclasses all register as an import side
effect.

---

## The two levels, and why the group one exists

A `Scorer` sees one completion. A `GroupScorer` sees all G completions sampled
from one prompt.

The group is free to ask for: `vllm_utils.compute_rewards` is already called
once per prompt with every sample of that prompt in `result.responses` —
open-instruct just hands them to the verifiers one at a time. Exposing the group
is what makes **relative** rewards possible, and most interesting rewards are
relative:

- **per-dimension normalisation** — you cannot z-score within a group from
  inside a single sample
- **a counterfactual baseline** — "how would this same completion have scored
  against a different item"
- **anything rank-based**

```python
from open_instruct.scored_rewards import GroupScorer, ScoreResult

class MyGroupScorer(GroupScorer):
    name = "my_scorer"

    async def score_group(self, group):
        return [ScoreResult(score=..., dimensions={...}, info={...}) for s in group]
```

`Sample` carries what you need: `completion`, `prompt`, `item` (the row's
`ground_truth`, JSON-decoded), `policy_text` (the policy's turns alone in a
multi-turn rollout), `rollout` (the environment's state), and `index` /
`group_size`.

---

## Multi-dimensional rewards

Summing dimensions and then normalising — what any single scalar reward does —
lets the highest-variance dimension own the gradient no matter what weights you
intended. Normalising each dimension inside the group *first* makes every
dimension contribute equally and makes the result invariant to each one's scale,
so there are no reward weights left to hand-tune:

```
A_k(t_i) = (r_k(t_i) - mean_k) / (std_k + eps)
A(t_i)   = (1/|K|) * sum_k A_k(t_i)
```

This is MO-GRPO (arXiv:2509.22047), which proves the scale-invariance property;
PEARL's Eq. 11-12 is the same estimator, and TRL ships it as
`multi_objective_aggregation="normalize_then_sum"`.

```python
from open_instruct.scored_rewards import guards
scorer = guards.MultiDimensional(my_scorer, ("accuracy", "clarity", "brevity"))
```

**It composes exactly right with open-instruct's advantage step.** `data_loader`
computes `advantages = scores - group_mean` (`advantage_normalization_type=centered`,
the default). The output of `normalize_then_sum` already has zero mean inside
the group, so that subtraction is a no-op and you get the equation above
literally. Leave it on `centered`.

A side effect worth knowing: a group whose members all score identically
produces all-zero advantages, so `--filter_zero_std_samples` (on by default)
drops it. That is the right behaviour — it is exactly the "this group carries no
gradient" case — and `scored/<name>/zero_advantage` reports how often it happens.
Watch it from step 0. When it approaches 1.0 the reward has stopped resolving
differences and no amount of further training will help.

---

## Guards: constraining a score you do not trust

A learned or judged reward is a model, and the policy will optimise the model
rather than the thing you meant.

**`Veto`** — a rule that overrides the score with a floor when it fires. For
behaviour you can detect with certainty and want to make strictly worse than
doing nothing. It should be a *rule*, not a second model; the point is that the
policy cannot negotiate with it.

Two things about the floor. Keep it modest: GRPO normalises within the group, so
a huge penalty mostly inflates the group's standard deviation and crushes the
signal among the samples that did not trip it. And note that the override
*replaces* the score rather than adjusting it, so a false positive does not add
noise — it discards everything else you measured about that sample. Calibrate
the rule's precision before turning it on.

**`Gate`** — zeroes the score unless a precondition holds. Distinct from a veto:
a gate says "this earned nothing", a veto says "this was worse than nothing".

**`Contrast`** — pay only for what a counterfactual does not also earn.

```python
guards.Contrast(my_scorer, guards.ItemPool(other_items))
```

The problem it solves: a completion can raise a score by doing the task, or by
being the sort of text that raises this score on anything. Re-scoring the same
completion against a deliberately mismatched item isolates the second and
subtracts it.

Two constraints, both easy to get wrong:

1. **The foreign item must come from outside the group.** A GRPO group is G
   samples of *one* prompt, so every member carries the same item — swapping
   items around inside the group returns each member its own and the term
   collapses to zero. Use `ItemPool`.
2. **It must vary within the group.** Advantages are mean-centred over the G
   completions, so any term identical across them cancels exactly. `ItemPool`
   hands the whole group *one* foreign item, so the only thing varying in the
   subtracted term is the member's own completion — which is what you want it to
   be measuring.

---

## Where scores come from

**`judge.Judge`** — a generative rubric judge. Dimensions, questions and scale
anchors are data you pass in, so a rubric can be versioned next to the run and
validated against human ratings before it is optimised against. Each dimension
is answered as `{"why": ..., "score": N}` with the reason *first*: a scalar head
can satisfy "high quality" with a style vector, and a model that must justify a
score by pointing at what it saw has a narrower way to be lazily right.

A dimension the judge did not produce comes back as `None`, not as a middling
default. A default is a silent vote for "average" on every parse failure, and
under group normalisation a systematic pull toward the mean is a bias, not
noise.

Serve the judge behind an endpoint (`judge.openai_generator`) rather than in the
trainer: it keeps it off the training GPU, lets you swap judge models without
touching the job, and makes it independently benchmarkable — which you want,
because a judge you cannot evaluate on its own is a reward you cannot debug.
`judge.stub_generator` gives deterministic mid-range scores for smoke tests.

**`head.LinearHead`** — a linear head on a frozen backbone's last hidden state.
Frozen because RL moves the policy every step; last position because attention
is causal; linear rather than MLP because the MLP memorises and does not
extrapolate into the regions RL pushes the policy toward.

Before letting a head drive a run, check that the label *tier* is not
recoverable from the embedding, and that **within-group** ranking holds up on
the tier your policy occupies. A global AUC of 0.92 can coexist with a
within-group ranking barely above a coin flip, because the global number pools
comparisons the algorithm never makes. `head.py`'s docstring has the numbers
from the run where this went wrong.

---

## The environment is another model

`partner_env.PartnerModelEnv` is a `TextRLEnvironment` whose observations come
from a frozen model over an OpenAI-compatible endpoint. Register it from your
plugin and it appears in `--tools` like any built-in environment.

Four things that are easy to get wrong, all documented in the module:

- **The partner cannot open the conversation.** open-instruct discards the
  observation returned by `reset`. Bake the opening line into the dataset prompt
  with `data.build_rows(opening=...)` — which is better anyway, since it is then
  identical across the G completions of a group.
- **The partner's tokens must not be trained on.** Leave `--mask_tool_use true`.
- **The partner's words must not be scored as the policy's.** Read
  `Sample.policy_text`, which the env records separately.
- **Control beats prompting.** A partner told "act confused" is a style, not an
  environment. Give it a `Director` that picks the next behaviour from a closed
  set before any text exists.

---

## The anchor

`anchor.Anchor` is not a reward. It is a number computed by something that is
not in the reward, on items the policy does not train on, and it is the only
thing that can tell you the reward is being gamed.

It measures three conditions on the same held-out items — no policy output, this
item's output, another item's output — and reports `gain` and `specificity`.
`gain` alone cannot tell teaching from filler; plenty of text raises an outcome
on *any* item. Read them together, and read both next to the standard error it
prints. At n=40 the smallest trustworthy difference is about 0.16.

Run it before training and after, and use `anchor.moved()` to state the change
in standard errors rather than in raw points.

---

## CLI reference

| flag | default | |
|---|---|---|
| `--reward_plugins` | `None` | modules or `.py` paths to import, comma separated |
| `--group_scorer` | `None` | `name` or `name:key=value,key=value` |
| `--group_reward_mode` | `replace` | `replace` ignores the verifier score, `add` sums them |
| `--group_reward_scale` | `1.0` | multiplier on the group score |
| `--group_scorer_strict` | `false` | raise on scorer errors instead of falling back |
| `--score_verifiers` | `None` | per-sample scorers exposed as ordinary verifiers |

Everything is built from a spec string **lazily, inside the actor**. The reward
config is pickled and shipped to the vLLM Ray actors, and a scorer holding an
open HTTP client or a CUDA tensor does not survive that trip.

---

## Testing

```bash
python -m unittest open_instruct.scored_rewards.test_scored_rewards -v
```

45 tests, no torch, no vLLM, no ray, no openenv. That is deliberate: a reward
should be checkable on a laptop before it costs a GPU hour.
