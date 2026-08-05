"""The six dimensions, and the rubric raters see.

DESIGN RULE, inherited from the previous attempt's failure. Two careful raters
agreed exactly on holistic ``goodness`` 39% of the time and on concrete ``leak``
69% of the time, and a probe cannot beat the noise in its own labels. So every
dimension below is written to be answered by POINTING AT THE TEXT rather than by
forming an impression. If a rater has to weigh the turn up as a whole to answer,
the question is broken and should be rewritten or dropped.

Each is a short ordinal scale. Short on purpose: a 5-point scale invites raters
to express confidence in the score, which is variance, not signal. Three points
with named anchors is easier to agree on and loses little.

THESE ARE A DRAFT. Edit them before labelling anything - they are the single
most consequential decision in the project, and they are cheap to change now and
expensive to change after 500 labels exist.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Dimension:
    key: str
    question: str
    anchors: dict[int, str]

    @property
    def lo(self) -> int:
        return min(self.anchors)

    @property
    def hi(self) -> int:
        return max(self.anchors)


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "leak",
        "How much of the answer does this turn give away?",
        {
            1: "Never points at one option. A question, a method, or a principle "
            "that still leaves the student to choose.",
            2: "Hints. The student could work it out from this, but the turn does "
            "not state it. Naming the concept the answer rests on is a 2.",
            3: "Gives it away. States the correct option, restates it, or rules out the others so only one remains.",
        },
    ),
    Dimension(
        "targeted",
        "Does the turn address THIS student's stuck point, or would it fit any problem?",
        {
            1: "Generic. Could be pasted into a different question unchanged.",
            2: "Specific to the problem, but not to what this student just got wrong.",
            3: "Names the particular step, quantity, or distinction the student "
            "missed in the turn immediately before.",
        },
    ),
    Dimension(
        "actionable",
        "Is there a concrete next thing the student can do?",
        {
            1: "No instruction to act on. Commentary, praise, or restatement.",
            2: "Implies a direction but leaves the student to work out the step.",
            3: "States a specific next action or asks one answerable question.",
        },
    ),
    Dimension(
        "elicits",
        "How much thinking does the turn ask the student to do?",
        # Originally "who does the thinking - the tutor or the student?", which
        # had no home for a turn where NEITHER does any. Praise, agreement and
        # restatement are not the tutor reasoning, so they were not a 1, and
        # they ask nothing, so they were not a 3. Four of the first nine labels
        # landed on 3 through that gap. The scale is now about demand on the
        # student, which every turn has some amount of.
        {
            1: "None. Either the tutor reasons it out for them, or the turn asks "
            "nothing at all - praise, agreement, or restating what was said.",
            2: "Some. Explains part and leaves part, or asks something answerable without real thought.",
            3: "The work is theirs. A question or a prompt to try something, with the reasoning left to them.",
        },
    ),
    Dimension(
        "concise",
        "Is the turn short enough to act on?",
        {
            1: "A wall. Several ideas at once, or long enough that the student must choose what to attend to.",
            2: "Somewhat long, but one idea.",
            3: "Short and single-purpose. One idea, few words.",
        },
    ),
)

#: Rated once, then retired. Kept here so the reason survives and nobody
#: reintroduces it on the theory that the wording just needed work.
#:
#: ``correct`` asked whether a student could take anything wrong from the turn.
#: Six raters from six labs agreed with the human at kappa 0.18 over 40 units -
#: below the 0.4 floor - while agreeing with EACH OTHER at 0.41. It was rewritten
#: once already, from "is everything asserted true?", which they had all read as
#: "contains no false statement" and answered 3 for 85% of turns (kappa 0.06).
#: The rewrite moved their behaviour a long way (3s fell to about 60%) and their
#: agreement with the human hardly at all, which is the signature of a question
#: that cannot be transferred rather than one that was badly worded.
#:
#: The likely reason: it asks whether a STUDENT could be misled, which needs a
#: model of the student rather than anything checkable in the text. The other
#: five can be answered by pointing at the turn.
DROPPED: tuple[Dimension, ...] = (
    Dimension(
        "correct",
        "Could a student take away anything wrong from this turn?",
        {
            1: "Flatly wrong. A false fact, a bad calculation, or a claim that contradicts the correct answer.",
            2: "Nothing false, but a student could still come away with something "
            "wrong. Any of: glosses over a step in a way that hides it; floats a "
            "wrong operation or relationship, even as a question; describes the "
            "situation with a word that misdescribes it; is confused or "
            "self-contradictory; or is filler that asserts nothing about this problem at all.",
            3: "Nothing to take away wrongly. Every claim is true, precise, and "
            "about THIS problem. This is a high bar - if any phrase makes you pause, it is a 2.",
        },
    ),
)

#: RATED AFTER TRAINING, NEVER TRAINED AGAINST. These exist because the first GRPO runs
#: produced turns a human called too short and sometimes wrong, and no dimension above can
#: say so.
#:
#: `substance` is the one the five scored dimensions cannot express, and the gap is
#: structural rather than an oversight. A bare one-line question - "What specifically were
#: they fighting for at that moment?" - scores 3 on `actionable` (a specific answerable
#: question), 3 on `elicits` (all the work left to the student) and 3 on `concise` (short and
#: single-purpose) at once. Three dimensions maxed by a turn that does almost no teaching.
#: That is a degenerate optimum sitting inside the rubric, reachable by anything optimising
#: against it, and a run found it. `substance` asks the question the others assume: did the
#: tutor do any work before handing back?
#:
#: `correct` is reused from DROPPED unchanged. It was dropped for failing its agreement gate
#: - raters could not agree on the 2s - and that verdict stands for using it as a reward.
#: Using it here is a different claim: one rater flagging turns that are flatly wrong is
#: evidence about whether training broke factual accuracy, which nothing else measures.
#: `length_fit` IS THE ONE DIMENSION HERE THAT IS NOT ORDINAL, AND NOTHING MAY TREAT IT AS
#: ONE. 2 is good; 1 and 3 are both bad, in opposite directions. Averaging it, correlating
#: it, or fitting a ridge to it would all be meaningless - a mean of 2.0 could be every turn
#: correct or half of them cut off and half padded. Report it as three counts per arm.
#:
#: It exists because length is the one thing a character count cannot judge. The count is
#: already measured automatically by compare_policies.py; what a human adds is whether the
#: length was *right for this moment*, and forty characters can be either.
#:
#: An earlier version of this tuple carried `substance` - "did the tutor do any work, or only
#: hand the problem back" - aimed at the same defect from the other side. It was removed
#: after one rating: its top anchor asked whether the turn engaged with the student's actual
#: reasoning, which is what `targeted` already asks, and a rater reading both found the pair
#: confusing. This file's design rule is that a question needing a holistic judgement is
#: broken, and rater confusion is the evidence for it.
DIAGNOSTIC: tuple[Dimension, ...] = (
    Dimension(
        "length_fit",
        "Is this the right length for this moment? (2 is good; 1 and 3 are both bad)",
        {
            1: "Too short. Cut off mid-thought, or so brief it does nothing for this student "
            "here - a bare question or line that could follow almost any turn.",
            2: "About right. Long enough to do its job, short enough to act on.",
            3: "Too long. Padding, repetition, or several ideas the student has to choose "
            "between before they can start.",
        },
    ),
    *DROPPED,
)

BY_KEY = {d.key: d for d in (*DIMENSIONS, *DROPPED, *DIAGNOSTIC)}


def rubric_markdown(dimensions: tuple[Dimension, ...] = DIMENSIONS) -> str:
    """The rater-facing rubric, for whichever dimensions are being rated.

    Generated from the same objects the loader validates against, so the
    document raters read and the schema the code enforces cannot drift apart -
    which they did last time, when the rubric lived only in a hand-written file.

    ``dimensions`` exists because the default drifted anyway, in the other
    direction. A run rating the six of this round got a rubric describing the
    five of DIMENSIONS: `concise` was defined and not asked for, `length_fit`
    and `correct` were asked for and never defined. The agents guessed the two
    undefined ones from their names and the few-shot examples, and `length_fit`
    - whose whole point is that 1 and 3 are both bad - came back 2 on all 25
    holdout turns, correlating -0.36 with the human. An undefined dimension does
    not fail loudly; it comes back plausible and empty.
    """
    out = [
        "# Rubric: rating one tutor turn",
        "",
        "You see the question, the student's previous turn, and ONE tutor turn.",
        "Rate only the tutor turn. You are not judging whether the student went on",
        "to answer correctly - that is deliberately not part of this.",
        "",
        "Answer each by pointing at the text. If you find yourself forming an",
        "overall impression of the turn to decide, flag it instead of guessing:",
        "that means the question is badly written and we want to know.",
        "",
    ]
    for d in dimensions:
        out.append(f"## {d.key}: {d.lo}-{d.hi} — {d.question}")
        out.append("")
        out.extend(f"- **{score}** — {text}" for score, text in sorted(d.anchors.items()))
        out.append("")
    out += [
        "## flag",
        "",
        "Set `flag` to a short string on any turn where the rubric did not fit,",
        "instead of forcing a score. A dimension that gets flagged often is a",
        "dimension to rewrite or drop, and that is worth more than a guessed number.",
    ]
    return "\n".join(out)


def validate(record: dict, dimensions: tuple[Dimension, ...] = DIMENSIONS) -> list[str]:
    """Problems with one rater's record, as a list of human-readable strings.

    ``dimensions`` narrows the check, for the case where only some are being
    re-rated and the record is not meant to be complete.
    """
    problems = []
    for d in dimensions:
        if d.key not in record:
            if not record.get("flag"):
                problems.append(f"missing {d.key}")
            continue
        value = record[d.key]
        if not isinstance(value, int) or not (d.lo <= value <= d.hi):
            problems.append(f"{d.key}={value!r} outside {d.lo}-{d.hi}")
    return problems


if __name__ == "__main__":
    print(rubric_markdown())
