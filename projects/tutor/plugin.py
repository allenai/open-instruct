"""The plugin. ``--reward_plugins projects.tutor.plugin`` is the whole wiring.

Importing this module registers two things:

  the ``tutor`` group scorer  - judged, multi-dimensional, leak-vetoed
  the ``tutor_student`` env   - a frozen student model as the environment

One training step, end to end:

    a ZPD problem the student fails alone, with the student's opening line
    already in the prompt
      -> the tutor writes a turn
      -> the frozen student replies, its next intent picked from a closed set
      -> three turns
      -> a judge scores each tutor turn on six dimensions in context
      -> the leak rule vetoes any dialogue that gave the answer away
      -> each dimension is z-scored inside the group of G dialogues and averaged
      -> GRPO

And separately, never in the reward: the anchor, which asks a frozen student to
actually answer the held-out questions.

WHAT IS NOT IN THE REWARD, AND WHY. Whether the student solved it. Four runs of
history are denominated in that number, and PEARL's failure mode - evaluating a
judge-shaped policy with the judge that shaped it - is only visible if something
outside the judge is watching. If the judged reward climbs while
``anchor/treated`` stays flat, the judge is being gamed and it shows within one
eval cycle. See ``run_anchor.py``.
"""

from __future__ import annotations

import logging

from open_instruct.scored_rewards import guards, registry
from open_instruct.scored_rewards.judge import Judge, mean_over_turns, openai_generator, stub_generator
from open_instruct.scored_rewards.types import TRANSCRIPT_KEY, GroupScorer, Sample, ScoreResult, parse_transcript
from projects.tutor import leak, rubric

logger = logging.getLogger(__name__)

#: Strictly below any non-leaking dimension score, which live in [0, 1]. An
#: earlier attempt had no floor and 8.2% of non-leaking turns scored below the
#: leak penalty - which told the policy that giving the answer away would have
#: scored better, inverting the one invariant the rule exists to enforce.
LEAK_FLOOR = -1.0


class JudgedDialogue(GroupScorer):
    """Per-dimension scores for a group of tutoring dialogues.

    Emits ``ScoreResult.dimensions`` and leaves the scalar to
    ``guards.MultiDimensional``, which normalises each dimension inside the
    group. That ordering is the point - see ``scored_rewards/aggregate.py``.

    Order of operations inside one dialogue:

    1. Average the judge's per-turn scores. The reward is terminal and every
       turn earned it, so a dialogue is its mean turn.
    2. Apply the completion gate. An abandoned dialogue earns nothing on ANY
       dimension, so a tutor cannot bank points for elegant scaffolding it
       walked away from.
    3. If the leak rule fired, every dimension goes to the floor. Paying for
       quality on top of a leak rewards a well-phrased give-away, and the rule
       is the one leak signal with independent calibration behind it.
    """

    name = "tutor"

    def __init__(
        self,
        judge_model: str = "",
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        judge_trajectory: bool = True,
        stub: bool = False,
        use_overlap: bool = False,
        use_elimination: bool = False,
    ):
        generate = (
            stub_generator(rubric.TURN_RUBRIC)
            if stub
            else openai_generator(judge_model, judge_base_url, judge_api_key, system=rubric.SYSTEM)
        )
        self.turn_judge = Judge(generate, rubric.TURN_RUBRIC)
        self.trajectory_judge = Judge(
            stub_generator(rubric.TRAJECTORY_RUBRIC)
            if stub
            else openai_generator(judge_model, judge_base_url, judge_api_key, system=rubric.SYSTEM),
            rubric.TRAJECTORY_RUBRIC,
        )
        self.judge_trajectory = judge_trajectory
        self.leak_kwargs = {"use_overlap": use_overlap, "use_elimination": use_elimination}

    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        episodes = [episode_of(s) for s in group]

        # one judge call per tutor turn across the WHOLE group, not per dialogue
        bodies, owner = [], []
        for i, (item, turns) in enumerate(episodes):
            for context, text in turns:
                bodies.append(rubric.turn_body(item, context, text))
                owner.append(i)
        turn_scores, _ = await self.turn_judge.score(bodies)

        per_dialogue: list[list[dict]] = [[] for _ in group]
        for i, score in zip(owner, turn_scores):
            per_dialogue[i].append(score)

        completes: list[float | None] = [None] * len(group)
        if self.judge_trajectory:
            traj_bodies = [rubric.trajectory_body(item, full_transcript(turns)) for item, turns in episodes]
            traj_scores, _ = await self.trajectory_judge.score(traj_bodies)
            completes = [s.get("complete") for s in traj_scores]

        results = []
        for sample, (item, turns), scores, complete in zip(group, episodes, per_dialogue, completes):
            tutor_text = "\n".join(text for _, text in turns) or sample.policy_text
            gave_away = leak.leaked_item(tutor_text, item, **self.leak_kwargs) if item.get("choices") else False
            gate = rubric.completion_gate(complete)

            if gave_away:
                dimensions = {n: LEAK_FLOOR for n in rubric.PAID_DIMENSIONS}
            else:
                averaged = mean_over_turns(scores, rubric.PAID_DIMENSIONS)
                # a dimension the judge could not score becomes neutral ONLY
                # after gating, and only because a group needs a number in every
                # slot; `dimensions_missing` is logged so you can drop these
                dimensions = {n: (0.5 if averaged[n] is None else averaged[n]) * gate for n in rubric.PAID_DIMENSIONS}

            results.append(
                ScoreResult(
                    dimensions=dimensions,
                    info={
                        "leaked": float(gave_away),
                        "complete_gate": gate,
                        "turns": float(len(turns)),
                        "judge_parse_failures": float(self.turn_judge.parse_failures),
                    },
                )
            )
        return results


def episode_of(sample: Sample) -> tuple[dict, list[tuple[list[dict], str]]]:
    """``(item, [(context_before_the_turn, tutor_turn), ...])``.

    The judge was validated on exactly this pair - the conversation as it stood
    before a turn, and that turn alone. Handing it a finished transcript instead
    produces noise wearing a reward's clothes.
    """
    item = sample.item
    transcript = parse_transcript(sample.env_info.get(TRANSCRIPT_KEY, "[]"))
    if not transcript:
        # single-turn, or no environment: the whole completion is one tutor turn
        text = sample.completion.strip()
        return item, ([([], text)] if text else [])

    turns, context = [], []
    for entry in transcript:
        if entry.get("who") == "policy":
            text = (entry.get("text") or "").strip()
            if text:
                turns.append((list(context), text))
        context.append(entry)
    return item, turns


def full_transcript(turns: list[tuple[list[dict], str]]) -> list[dict]:
    if not turns:
        return []
    context, last = turns[-1]
    return list(context) + [{"who": "policy", "text": last}]


def swap_item(sample: Sample, group: list[Sample]) -> Sample:
    """Rotate the ITEM and keep the completion - see ``guards.Contrast``."""
    return guards.rotate(sample, group)


@registry.register("tutor")
def build_tutor_scorer(
    judge_model: str = "",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_trajectory: bool = True,
    stub: bool = False,
    use_overlap: bool = False,
    use_elimination: bool = False,
) -> GroupScorer:
    """``--group_scorer tutor:judge_model=...,judge_base_url=http://host:8001/v1``"""
    inner = JudgedDialogue(
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_trajectory=judge_trajectory,
        stub=stub,
        use_overlap=use_overlap,
        use_elimination=use_elimination,
    )
    return guards.MultiDimensional(inner, rubric.PAID_DIMENSIONS, name="tutor")


@registry.register("tutor_leak_only")
def build_leak_only(**_) -> GroupScorer:
    """The ablation: the leak rule alone, no judge, no endpoint, no GPU.

    Worth running first. It is the term with independent calibration behind it,
    it needs nothing served, and four runs of history say it is the only part of
    this reward that has ever moved a policy.
    """

    class LeakOnly(GroupScorer):
        name = "tutor_leak_only"

        async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
            out = []
            for sample in group:
                item, turns = episode_of(sample)
                text = "\n".join(t for _, t in turns) or sample.policy_text
                gave_away = leak.leaked_item(text, item) if item.get("choices") else False
                out.append(ScoreResult(score=LEAK_FLOOR if gave_away else 0.0, info={"leaked": float(gave_away)}))
            return out

    return LeakOnly()


# --- the environment ------------------------------------------------------
#
# Guarded because the environment stack needs `openenv-core` and open-instruct's
# ray/vllm dependencies, and the scorer above needs none of them. That is what
# lets the leak rule and the judge parsing be unit-tested on a laptop.

try:
    from open_instruct.scored_rewards.partner_env import PartnerModelEnvConfig, register_env

    register_env("tutor_student", PartnerModelEnvConfig)
except ImportError as exc:  # pragma: no cover - depends on the install
    logger.info("projects.tutor: environment not registered (%s). The scorer still works.", exc)


__all__ = ["JudgedDialogue", "LEAK_FLOOR", "build_tutor_scorer", "episode_of"]
