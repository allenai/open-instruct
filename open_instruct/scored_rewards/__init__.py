"""Score-based rewards for open-instruct GRPO.

open-instruct's reward path is built for verifiable rewards: a rule looks at one
completion and says whether it is right. This package adds the other kind - a
reward that is a SCORE, produced by a model or by your own code, when there is
no rule that can decide.

Five things it adds, in the order you will meet them:

1. ``registry``       your reward lives in your repo. ``--reward_plugins`` imports
                      it; nothing here needs editing.
2. ``types``          ``Scorer`` scores one completion, ``GroupScorer`` scores all G
                      completions of a prompt together. The group is already
                      computed together upstream - this just exposes it.
3. ``guards``         wrappers for a score you do not fully trust: a rule-based
                      veto, a completion gate, a counterfactual contrast, and
                      per-dimension normalisation.
4. ``judge`` / ``head``   the two ways to produce a score: a generative rubric judge,
                      or a learned linear head on a frozen backbone.
5. ``partner_env``    the environment is another model, for tasks whose
                      environment is really a person.

And ``anchor``, which is none of the above: a metric computed OUTSIDE the reward
on held-out items, which is the only thing that can tell you the reward is being
gamed.

Nothing at this level imports torch, vLLM, ray or openenv. The submodules that
need them import them lazily or are themselves imported lazily, so a scorer can
be written and unit-tested with nothing installed.
"""

from open_instruct.scored_rewards import aggregate, anchor, data, guards, judge, registry
from open_instruct.scored_rewards.registry import build, register, register_fn
from open_instruct.scored_rewards.types import (
    POLICY_TEXT_KEY,
    TRANSCRIPT_KEY,
    FunctionScorer,
    GroupScorer,
    PerSample,
    Sample,
    Scorer,
    ScoreResult,
)

__all__ = [
    "FunctionScorer",
    "GroupScorer",
    "PerSample",
    "POLICY_TEXT_KEY",
    "Sample",
    "ScoreResult",
    "Scorer",
    "TRANSCRIPT_KEY",
    "aggregate",
    "anchor",
    "build",
    "data",
    "guards",
    "judge",
    "register",
    "register_fn",
    "registry",
]
