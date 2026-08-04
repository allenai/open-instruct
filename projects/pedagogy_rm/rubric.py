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
        "correct",
        "Could a student take away anything wrong from this turn?",
        # Rewritten after the first agent run. It used to ask whether the turn
        # was TRUE, and five raters from five labs read that as "contains no
        # false statement" - so they returned 3 for 85% of turns and agreed
        # with the human at kappa 0.06. The human was marking things that are
        # not false but still leave a wrong impression: a question floating a
        # wrong operation, a conflict described as an "overlap", filler that
        # says nothing about the problem. Those are named below, because a
        # scale whose failure modes are listed can be pointed at, and one that
        # says "imprecise" cannot.
        {
            1: "Flatly wrong. A false fact, a bad calculation, or a claim that contradicts the correct answer.",
            2: "Nothing false, but a student could still come away with something "
            "wrong. Any of: glosses over a step in a way that hides it; floats a "
            "wrong operation or relationship, even as a question; describes the "
            "situation with a word that misdescribes it; is confused or "
            "self-contradictory; or is filler that asserts nothing about this "
            "problem at all.",
            3: "Nothing to take away wrongly. Every claim is true, precise, and "
            "about THIS problem. This is a high bar - if any phrase makes you "
            "pause, it is a 2.",
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

BY_KEY = {d.key: d for d in DIMENSIONS}


def rubric_markdown() -> str:
    """The rater-facing rubric.

    Generated from the same objects the loader validates against, so the
    document raters read and the schema the code enforces cannot drift apart -
    which they did last time, when the rubric lived only in a hand-written file.
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
    for d in DIMENSIONS:
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
