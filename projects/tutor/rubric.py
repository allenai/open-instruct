"""The tutoring rubric, and the views the judge is shown.

PEARL's eight dimensions (arXiv:2605.29582 Appendix C), minus ``emotion``.
Dropping it is supported by their own numbers: Table 4A puts its validation
accuracy at 97.20 - near-saturated and therefore barely discriminative - and
Table 1 shows it moving 92.6 to 95.1, the smallest gain of any dimension. It
costs judge tokens to measure something that does not vary.

``adaptive`` is kept even though it is the most expensive to judge, because it
is the only dimension that reads "did the tutor respond to THIS student's
state", which is the entire point of giving the student a controllable
simulator. Without it the simulator is decoration.

Scales are 1-5. PEARL does not specify its criterion scales, so these anchors
are ours, written to match the 1-5 goodness and 1-3 leak scales already used in
the 1,104 human and agent ratings, so those ratings can validate them.

SCORED ONE TUTOR TURN AT A TIME, against the conversation as it stood BEFORE
that turn. Not a detail: the ratings that validate this rubric are per-turn
ratings in context, and an earlier attempt fed the finished transcript with all
three tutor turns concatenated as the "message to grade" - 1,910 characters
against a trained-on 616, with the graded text also duplicated inside the
transcript it was supposedly graded against. What we validate is what we use.

``complete`` is the exception and is scored once per dialogue, because
completion is not a property a single turn can have. It is a GATE, not a paid
dimension.

DELIBERATELY ABSENT: whether the student got the answer right. PEARL's reward is
100% judge and their headline evaluation is that same trained judge, which is
how Metacognition moves 50.0 to 95.0 on a dimension the judge was trained to
score. Solve rate stays out of this file entirely and lives in the anchor.
"""

from __future__ import annotations

from open_instruct.scored_rewards.judge import Dimension, Rubric

SYSTEM = (
    "You are a demanding grader of tutoring messages, of the kind who marks a "
    "competent lesson 3 out of 5.\n"
    "The messages you grade come from a small model that is mediocre at tutoring. "
    "Most deserve a 2 or a 3. A 5 means you could not improve the turn if you "
    "tried; a 4 means it is genuinely good. If you find yourself giving 5s to most "
    "turns you are grading the effort, not the teaching.\n"
    "Write the reason BEFORE the score, and make the reason quote or name the "
    "specific thing in the message you are reacting to. A reason that would fit any "
    "tutoring message means you have not read this one.\n"
    "Reply with JSON only - no preamble, no code fence."
)

TURN_DIMENSIONS = (
    Dimension(
        "acc",
        "Is everything the tutor says correct, and does it point toward the reference answer rather than away from it?",
        {
            1: "states something false, or steers toward a distractor",
            3: "correct but vague, or one careless slip that does not mislead",
            5: "everything stated is correct and on the path to the answer",
        },
    ),
    Dimension(
        "leak",
        "Does the tutor do the student's reasoning for them? Full leakage is naming the answer or handing over the "
        "key calculation. Partial leakage is a hint so specific the task collapses to verification, or narrowing the "
        "options down to one. No leakage means open questions, general strategy or conceptual scaffolding, with the "
        "essential step left to the student.",
        {
            1: "names the correct answer, or eliminates all but one option",
            2: "hands over the key calculation or result the student needed to find",
            3: "hints so specifically the student only has to agree",
            4: "points at the right area but the student still has to do the step",
            5: "asks or scaffolds without narrowing the answer at all",
        },
    ),
    Dimension(
        "load",
        "Is this turn concise, clear, appropriately long, and free of irrelevant material, for a student at this grade?",
        {
            1: "a wall of text, or so terse it says nothing",
            3: "understandable but padded or slightly off-topic",
            5: "one clear idea at a length the student can hold",
        },
    ),
    Dimension(
        "guide",
        "Does the turn keep the student as the one doing the thinking? Judge the questioning strategy and whether the "
        "guidance invites work rather than instructing.",
        {
            1: "lectures; the student has nothing to do but read",
            3: "asks something, but it is rhetorical or trivially answerable",
            5: "asks a question the student must actually think to answer",
        },
    ),
    Dimension(
        "meta",
        "Does the turn get the student to reflect - notice their own mistake, explain or check their reasoning, or "
        "name the method being used?",
        {
            1: "no invitation to reflect at all",
            3: "gestures at the student's thinking without engaging it",
            5: "makes the student examine or justify their own reasoning",
        },
    ),
    Dimension(
        "adaptive",
        "Does the turn respond to what THIS student just said and to the confusion they actually have, changing "
        "approach when they are stuck?",
        {
            1: "generic; would be identical for any student on any problem",
            3: "acknowledges the student but continues the planned line",
            5: "visibly built on this student's specific last message",
        },
    ),
)

COMPLETE = Dimension(
    "complete",
    "Taking the dialogue as a whole: did the tutor finish the job - cover the reasoning the problem needs, address "
    "the student's mistakes, and land somewhere conclusive?",
    {
        1: "abandoned the student mid-confusion",
        3: "covered some ground but left a necessary step untouched",
        5: "the student has everything they need to finish it themselves",
    },
)

TURN_RUBRIC = Rubric(
    dimensions=TURN_DIMENSIONS,
    system=SYSTEM,
    instructions=(
        "Score that message 1-5 on each dimension. 5 is always good tutoring.\n"
        "Grade the MESSAGE, not the topic and not how friendly it sounds. A turn can be "
        "perfectly polite and still score 2 on every dimension."
    ),
)

TRAJECTORY_RUBRIC = Rubric(
    dimensions=(COMPLETE,),
    system=SYSTEM,
    instructions="Score the conversation as a whole, 1-5 on each dimension. 5 is always good.",
)

#: What the reward pays for. `complete` gates it rather than being paid.
PAID_DIMENSIONS = tuple(d.name for d in TURN_DIMENSIONS)


def format_choices(choices) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


def turn_body(item: dict, context: list[dict], tutor_turn: str) -> str:
    """What the judge sees for one tutor turn.

    The gold answer is shown on purpose - ``leak`` cannot be scored without
    knowing what there is to leak - and it is the reason this judge can never be
    run as the student.
    """
    grade = f"Student is in grade {item['grade']}.\n" if item.get("grade") else ""
    return (
        f"Problem the student is working on:\n{item['question']}\n"
        f"{format_choices(item['choices'])}\n"
        f"Correct answer: {item['choices'][item['gold_idx']]}\n{grade}\n"
        f"Conversation before this message:\n{render_context(context)}\n\n"
        f"THE TUTOR MESSAGE TO GRADE:\n{tutor_turn.strip()}"
    )


def trajectory_body(item: dict, transcript: list[dict]) -> str:
    return (
        f"Problem the student was working on:\n{item['question']}\n"
        f"{format_choices(item['choices'])}\n"
        f"Correct answer: {item['choices'][item['gold_idx']]}\n\n"
        f"Full conversation:\n{render_context(transcript) or '(empty)'}"
    )


def render_context(turns: list[dict]) -> str:
    if not turns:
        return "(the tutor is opening the conversation)"
    label = {"policy": "Tutor", "partner": "Student"}
    return "\n".join(f"{label.get(t['who'], t['who'].capitalize())}: {t['text'].strip()}" for t in turns)


def completion_gate(complete_score: float | None, threshold: float = 0.25) -> float:
    """PEARL's g(tau): an unfinished dialogue earns no pedagogical reward.

    Their Eq. 7 zeroes every criterion in every dimension, not just ``complete``,
    so a tutor cannot bank points for elegant scaffolding it walked away from.
    The threshold is one step above the floor on the [0,1] scale - a ``complete``
    of 1/5 is "abandoned the student mid-confusion" and nothing else.

    An UNSCOREABLE dialogue is not gated to zero. A parse failure is our bug and
    should not be charged to the policy as an incomplete dialogue.
    """
    if complete_score is None:
        return 1.0
    return 1.0 if complete_score > threshold else 0.0
