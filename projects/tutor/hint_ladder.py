"""Does anything short of the answer move the student on its OWN problem?

WHY THIS EXISTS. Four training runs produced a flat ``teacher_acc``, and the
transfer probe then found nothing carried from one problem to another
(specificity +0.0001 +/- 0.0050). Both are nulls, and a null only becomes a
finding once you know the measurement could have registered the effect. The
transfer probe's positive control settled that much: told the answer outright,
the 0.5B moves from 0.259 to 0.517, about 13 SE. So the channel is alive.

But that control tested the two extremes and nothing in between, and it tested
TRANSFER - a dialogue about A, scored on B - which is strictly harder than what
training actually asks for. Training asks that a dialogue about A help on A. That
has never been measured here with a control, which is the gap this closes.

THE LADDER. Five rungs on the same item, ordered by how much of the answer they
contain, each scored as P(gold):

    0  nothing                     the floor
    1  the knowledge unit, named   "This problem is about X."
    2  the misconception, named    "A common mistake is Y."
    3  a real tutoring dialogue    non-leaking ones only
    4  the answer, stated          the ceiling, and not teaching

Rungs 1 and 2 come free from units.py's annotations; rung 3 from the traces.
Every rung is measured on the SAME items, so every comparison is paired and the
standard errors are on per-item differences rather than on group means.

HOW TO READ IT. Rung 4 is a sanity check, not a result - it should be large, and
if it is not, the student is too weak to measure and nothing else on the page
means anything. Rung 3 is the experiment. It is the training objective, stated as
a number.

    rung 3 moves, rung 4 large      the dialogues teach; the reward channel was
                                    the broken part, and the measured gain can
                                    replace the judge outright
    rung 3 flat, rungs 1-2 move     the format works and the teacher's output is
                                    the weak part - a different, easier problem
    only rung 4 moves               nothing but telling registers. A frozen
                                    multiple-choice student cannot see teaching,
                                    and no reward model fixes that. Stop.

Run it on both students. The 0.5B is the one the run history used, so it is the
comparable number; the 3B tests whether that student was simply too weak to hold
a partial understanding for a hint to complete.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics

from projects.tutor import units as units_mod
from projects.tutor.student import LocalChoiceStudent
from projects.tutor.transfer_probe import CachingStudent, StubStudent, load_dialogues

RUNGS = ("nothing", "unit named", "misconception named", "real dialogue", "answer stated")


def build_rungs(item: dict, unit: units_mod.Unit, dialogue: str) -> list[str]:
    """The five hints for one item, weakest first.

    Rungs 1 and 2 are deliberately terse. A paragraph would confound the thing
    being tested with the mere presence of text, and rung 3 already carries that
    confound - which is why rung 3 is read against rungs 1-2 and not only against
    the floor.
    """
    gold = item["choices"][item["gold_idx"]]
    return [
        "",
        f"This problem is about {unit.primary}.",
        f"A common mistake on this kind of problem is {unit.misconception}.",
        dialogue,
        f"The answer is {gold}.",
    ]


def select(items, units, dialogues, seed: int = 0):
    """Items where all five rungs exist, so the ladder is paired throughout.

    An item missing a misconception or a clean dialogue would have to be dropped
    from some rungs and not others, which would make the rungs incomparable -
    each would be an average over a different set of problems.
    """
    rng = random.Random(seed)
    chosen = []
    for item in items:
        key = units_mod.item_key(item)
        unit = units.get(key)
        if not unit or not unit.primary or not unit.misconception:
            continue
        clean = [d for d in dialogues.get(key, []) if not d.get("leaked")]
        if not clean or item.get("gold_idx") is None or not item.get("choices"):
            continue
        text = str(rng.choice(clean).get("completion") or "").strip()
        if text:
            chosen.append((item, unit, text))
    return chosen


async def run(args) -> None:
    with open(args.items) as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    units = units_mod.load(args.units)
    dialogues = load_dialogues(args.traces, args.tier or None)
    selected = select(items, units, dialogues, args.seed)
    if args.limit:
        selected = selected[: args.limit]
    print(f"{len(items)} items, {len(units)} annotated, {len(selected)} with all five rungs")
    if not selected:
        raise SystemExit("no items carry a unit, a misconception and a clean dialogue")

    if args.stub_student:
        print("\n*** STUB STUDENT - plumbing check only, these numbers mean nothing ***")
        student = CachingStudent(StubStudent(args.seed))
    else:
        print(f"loading {args.student_model} in-process")
        student = CachingStudent(LocalChoiceStudent(args.student_model))

    limiter = asyncio.Semaphore(args.concurrency)

    async def measure(item: dict, hint: str) -> float:
        async with limiter:
            return await student.prob_gold(item, hint)

    ladders = [build_rungs(item, unit, text) for item, unit, text in selected]
    columns = await asyncio.gather(
        *(
            asyncio.gather(*(measure(item, ladder[rung]) for (item, _, _), ladder in zip(selected, ladders)))
            for rung in range(len(RUNGS))
        )
    )

    print(f"\n=== hint ladder, {args.student_model} (n={len(selected)}, P(gold)) ===\n")
    floor = columns[0]
    rows = []
    for name, column in zip(RUNGS, columns):
        gains = [c - f for c, f in zip(column, floor)]
        mean = statistics.fmean(gains)
        se = statistics.stdev(gains) / math.sqrt(len(gains)) if len(gains) > 1 else float("nan")
        sigmas = mean / se if se else 0.0
        marker = "" if name == "nothing" else f"  gain {mean:+.4f} +/- {se:.4f}  = {sigmas:+.1f} SE"
        print(f"  {name:<22} {statistics.fmean(column):.3f}{marker}")
        rows.append({"rung": name, "mean": statistics.fmean(column), "gain": mean, "se": se, "sigmas": sigmas})

    verdict(rows)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"n": len(selected), "model": args.student_model, "rungs": rows}, handle, indent=2)
        print(f"\nwrote {args.out}")


def verdict(rows: list[dict]) -> None:
    """Say what the numbers mean, so a later reader cannot quietly pick a story.

    Three SE is the bar for calling a rung live. It is deliberately above the
    usual two: five rungs are being read off one run, and at two SE one of them
    clearing by chance is not a remote possibility.
    """
    live = {r["rung"]: r["sigmas"] >= 3 for r in rows[1:]}
    print()
    if not live["answer stated"]:
        print(
            "  -> INSTRUMENT TOO WEAK. The student does not follow even a stated\n"
            "     answer, so it cannot register a hint of any size. No conclusion\n"
            "     about teaching can be drawn; use a stronger student."
        )
    elif live["real dialogue"]:
        print(
            "  -> THE DIALOGUES TEACH. A real, non-leaking dialogue moves the\n"
            "     student on its own problem. The environment carries signal, the\n"
            "     reward channel was the broken part, and this measured gain can\n"
            "     serve as the reward directly instead of a judge."
        )
    elif live["unit named"] or live["misconception named"]:
        print(
            "  -> THE FORMAT WORKS, THE DIALOGUES DO NOT. A bare sentence moves the\n"
            "     student where a whole tutoring dialogue does not, so the channel\n"
            "     is open and the teacher's output is what is weak. Fix the teacher,\n"
            "     not the setup."
        )
    else:
        print(
            "  -> ONLY TELLING WORKS. Nothing short of the answer registers. A\n"
            "     frozen multiple-choice student cannot distinguish teaching from\n"
            "     telling, so any reward built on its correctness is maximised by\n"
            "     leaking. No reward model repairs this - change the environment\n"
            "     or stop."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True)
    parser.add_argument("--units", required=True, help="output of units.py")
    parser.add_argument("--traces", required=True, help="gen_traces.jsonl")
    parser.add_argument("--tier", default="policy")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stub-student", action="store_true", help="plumbing check, no GPU")
    parser.add_argument("--out")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
