# projects/

One directory per experiment. Nothing in `open_instruct/` imports anything here.

That separation is the point: `open_instruct/scored_rewards/` should contain
only things a stranger with a different task would want, and this is where the
things they would not want go. If you find yourself adding a task-specific
special case to `scored_rewards`, it belongs here instead.

This directory is currently empty. The project that shaped `scored_rewards` — a
teacher LLM trained against a frozen student, using a judged multi-dimensional
score, a rule-based veto, a model as the environment, and a held-out anchor — was
removed once its own experiments showed the environment could not distinguish
teaching from telling. It is recoverable from the history of this branch, and is
worth reading before building anything model-as-environment shaped.

---

## Starting your own

A project is a directory with a `plugin.py` that registers things by being
imported. The trainer finds it with `--reward_plugins projects.yours.plugin`.

**1. Decide what the score is, and whether it needs the group.**

Use `Scorer` when one completion is enough. Use `GroupScorer` when the reward
is relative — normalising per dimension, a counterfactual baseline, anything
rank-based. Only `GroupScorer` sees all G samples of a prompt.

**2. Register it.**

```python
# projects/yours/plugin.py
from open_instruct.scored_rewards import registry, ScoreResult, GroupScorer

@registry.register("yours")
def build(judge_model="Qwen/Qwen2.5-7B-Instruct", judge_base_url=None, **kw):
    ...  # returns a GroupScorer
```

The factory takes keyword arguments so the CLI can configure it:
`--group_scorer "yours:judge_model=...,judge_base_url=..."`. Build models and
HTTP clients **inside** the factory, and let the factory be called lazily in the
actor — the reward config is pickled to Ray workers and an open client does not
survive that.

**3. Build rows.**

`scored_rewards.data.build_rows` writes RLVR-format JSONL: `messages`,
`ground_truth` (your item, JSON-encoded — this is what `Sample.item` decodes),
`dataset` (the routing key for per-sample verifiers), and the env config if you
use one.

Hold out along whatever axis you actually want to generalise across, not a
random slice. `build_rows`'s companion `data.split_by` does this.

**4. Write the anchor before you write the training script.**

`anchor.Anchor` computes a number using something that is not in the reward, on
items you do not train on. It is the only thing that can distinguish learning
from the policy finding a shortcut through your judge, and it is much harder to
add convincingly after a run has already produced a nice reward curve.

**5. Test on CPU.**

Both existing test files run with no torch, no ray, no vLLM and no network.
Stub the judge (`judge.stub_generator`), assert on the reward your scorer
returns for hand-written completions, and check that the guard you added fires
where you think it does. A reward bug found on a laptop is free; the same bug
found at step 300 is not.
