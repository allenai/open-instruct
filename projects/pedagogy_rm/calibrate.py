"""Turn your labels into a written account of how YOU apply the rubric.

    python -m projects.pedagogy_rm.calibrate \
        --labels data/labels/sophia.json --units data/label_slices/slice_1.json \
        --out data/calibration.md --holdout 10

WHY THIS EXISTS. A rubric fixes what the questions are. It does not fix where one
rater draws the line between 2 and 3, and that line is most of the disagreement.
Handing agents the rubric alone gets you six models' opinions about teaching;
handing them the rubric plus evidence of how you actually scored gets you six
estimates of YOUR scoring, which is the thing the probe is supposed to learn.

HOW. A strong model reads the rubric and every one of your labelled examples,
including the turn text, and writes down the decision rules that reproduce your
scores - especially the boundary cases and any tendency the rubric does not
mention. That document, not this script, is what the raters are given.

THE HOLDOUT IS THE POINT. A fraction of your labels is withheld from calibration
and never shown to any agent. Without it there is no way to tell whether the
agents learned your standard or merely memorised your examples, and "the agents
agree with me on the items they were shown" is not evidence of anything. The
holdout is what ``agreement.py`` later scores them against.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random

from projects.pedagogy_rm.rubric import DIMENSIONS, rubric_markdown

PROMPT = """Below is a rubric for rating tutor turns, then {n} turns rated by ONE \
human rater. Your job is to write the instructions that would let someone else \
reproduce this rater's scores.

Write a document with one section per dimension. In each section state:

- Where this rater actually puts the boundary between each pair of adjacent \
scores, in terms you could apply to a new turn. Quote short fragments from the \
examples as evidence.
- Any rule they appear to follow that the rubric does not state.
- Any case where their score is not what the rubric's wording alone would \
predict. These are the most useful things you can find; do not smooth them over.
- Their distribution: if a dimension is nearly always 3, say so, and say what \
makes the exceptions exceptional.

Be concrete. "Judge how much is given away" is useless - the rubric already says \
that. "Naming the concept counts as 2, but naming it AND saying which option it \
rules out is 3" is what we need.

If the evidence for a dimension is thin or contradictory, say that plainly \
rather than inventing a rule. A wrong rule is worse than a missing one.

End with a short section "Where this rater is inconsistent", listing any \
dimension whose examples cannot be reconciled. That tells us which dimensions to \
distrust later.

=== RUBRIC ===

{rubric}

=== RATED EXAMPLES ===

{examples}
"""


def render_example(unit: dict, record: dict, index: int) -> str:
    scores = "  ".join(f"{d.key}={record[d.key]}" for d in DIMENSIONS if d.key in record)
    flag = f"\nRATER FLAGGED: {record['flag']}" if record.get("flag") else ""
    return (
        f"--- example {index} ---\n"
        f"QUESTION: {' '.join(unit['question'].split())}\n"
        f"STUDENT SAID: {' '.join(unit['student_before'].split())}\n"
        f"TUTOR TURN: {' '.join(unit['tutor_turn'].split())}\n"
        f"SCORES: {scores}{flag}\n"
    )


def load(labels_path: str, units_path: str) -> tuple[list[dict], dict[str, dict]]:
    with open(units_path) as handle:
        units = {u["id"]: u for u in json.load(handle)["units"]}
    with open(labels_path) as handle:
        records = json.load(handle)["labels"]
    complete = [r for r in records if all(d.key in r for d in DIMENSIONS) and r["id"] in units]
    return complete, units


async def main_async(args) -> None:
    from projects.pedagogy_rm import gateway  # noqa: PLC0415

    records, units = load(args.labels, args.units)
    if len(records) < args.holdout + 5:
        raise SystemExit(f"only {len(records)} complete labels; need more than --holdout ({args.holdout}) plus a few")

    random.Random(args.seed).shuffle(records)
    holdout, train = records[: args.holdout], records[args.holdout :]
    print(f"{len(records)} complete labels: {len(train)} for calibration, {len(holdout)} held out")

    examples = "\n".join(render_example(units[r["id"]], r, i + 1) for i, r in enumerate(train))
    prompt = PROMPT.format(n=len(train), rubric=rubric_markdown(), examples=examples)

    client = gateway.make_client()
    reply = await client.chat.completions.create(
        model=args.model, messages=[{"role": "user", "content": prompt}], temperature=0.2
    )
    text = (reply.choices[0].message.content or "").strip()

    with open(args.labels) as handle:
        rater = json.load(handle).get("rater", "?")
    with open(args.out, "w") as handle:
        handle.write(f"<!-- calibrate.py: {len(train)} labels by {rater}, via {args.model} -->\n\n")
        handle.write(text + "\n")
    with open(args.holdout_out, "w") as handle:
        json.dump({"schema": "pedagogy-rm/holdout-v1", "ids": [r["id"] for r in holdout]}, handle, indent=1)

    print(f"wrote {args.out} ({len(text.split())} words)")
    print(f"wrote {args.holdout_out} — {len(holdout)} ids no agent will be shown")
    print("\nREAD IT before running the agents. If it describes rules you do not")
    print("recognise as yours, the agents will copy those rules faithfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--out", default="data/calibration.md")
    parser.add_argument("--holdout-out", default="data/holdout.json")
    parser.add_argument("--holdout", type=int, default=10, help="labels withheld from every agent")
    parser.add_argument("--model", default="openai-group/gpt-5.6-terra")
    parser.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
