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
        "Is everything the turn asserts about this problem true?",
        {
            1: "Contains a factual or mathematical error, or an analogy that does "
            "not map onto this problem. Fluent and confident does not help.",
            2: "Nothing false, but something is imprecise or could mislead.",
            3: "Everything asserted is true and precise.",
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
        "Who does the thinking - the tutor or the student?",
        {
            1: "The tutor does it. Explains or performs the reasoning outright.",
            2: "Mixed. Explains part, leaves part.",
            3: "The student does it. The turn is a question or a prompt to try "
            "something, and the reasoning is left to them.",
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


def validate(record: dict) -> list[str]:
    """Problems with one rater's record, as a list of human-readable strings."""
    problems = []
    for d in DIMENSIONS:
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
