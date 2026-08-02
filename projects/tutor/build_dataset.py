"""Turn screened problems into rows open-instruct will train on.

    python -m projects.tutor.build_dataset \
        --items data/state_tests/train_items.jsonl \
        --out data/tutor_train.jsonl

Input is one JSON object per line with at least ``question``, ``choices`` and
``gold_idx``; ``grade``, ``subject``, ``state`` and ``choose_pick`` are used when
present. Output is the four-column RLVR format.

SCREEN THE ITEMS FIRST. The reward only has a gradient on problems the student
fails alone - a problem it already answers correctly produces the same outcome
for every member of the group and therefore no gradient at all. An unscreened
attempt at this opened at 97% solved with three quarters of its groups
contributing nothing; screened, the same setup ran at 27%. This script does not
screen; it assumes the input is already the kept set.

THE SPLIT IS BY SOURCE, not at random. ``--held-out PA`` holds out every item
from one state - a different test, authored by different people, in a different
year. A random slice of one corpus shares its authors and its house style with
the training set, so it measures memorisation of items and nothing else.

HOW BIG. Training wants 300-500 items. Below about 200 the tutor sees each
problem six to thirteen times and can learn per-problem hints instead of how to
tutor. The held-out set wants about 200, because that is where the minimum
detectable difference (~0.07) drops below the effects worth chasing.
"""

from __future__ import annotations

import argparse
import json

from open_instruct.scored_rewards import data
from projects.tutor.student import opening_line

TEACHER_SYSTEM = (
    "You are a patient tutor working through a problem WITH a middle-school student.\n"
    "Your goal is to teach, not to answer. In each turn do whatever helps most: explain "
    "the underlying idea, give a small worked example from a similar situation, ask a "
    "question that makes them think, correct a misconception, or supply background "
    "knowledge they do not have. Give a direct hint ONLY if they are truly stuck, and "
    "always respond to their last message.\n"
    "Rules:\n"
    "1. NEVER reveal the answer. Do not name, quote, or rule out any option.\n"
    "2. Keep each turn to 2-3 sentences so it stays a conversation.\n"
    "3. Plain language a 12-year-old understands. No preamble."
)


def user_view(item: dict) -> str:
    """What the tutor sees. Deliberately EXCLUDES the gold answer.

    The options are shown because the student can see them and the tutor has to
    know what it must not name.
    """
    choices = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(item["choices"]))
    return f"A student is stuck on this question:\n\n{item['question']}\n{choices}"


def build(items, *, env: str | None = "tutor_student", turns: int = 3, opener: bool = True) -> list[dict]:
    return data.build_rows(
        items,
        system=TEACHER_SYSTEM,
        user=user_view,
        verifier=data.PASSTHROUGH,
        env_name=env,
        env_kwargs=(lambda item: {"max_turns": turns}) if env else None,
        max_steps=turns,
        opening=(lambda item: opening_line(item, seed=abs(hash(item["question"])) % (2**31))) if opener else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True, help="jsonl of screened problems")
    parser.add_argument("--out", required=True, help="where to write the RLVR jsonl")
    parser.add_argument("--eval-out", default=None, help="where to write the held-out split")
    parser.add_argument("--split-key", default="state", help="field to hold out on")
    parser.add_argument("--held-out", nargs="*", default=[], help="values of --split-key to hold out")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--env", default="tutor_student", help="empty string for single-turn")
    parser.add_argument("--no-opener", action="store_true", help="do not put the student's first line in the prompt")
    parser.add_argument("--push", default=None, help="also push to this Hub repo id")
    args = parser.parse_args()

    with open(args.items) as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    train, held = data.split_by(items, args.split_key, args.held_out) if args.held_out else (items, [])

    rows = build(train, env=args.env or None, turns=args.turns, opener=not args.no_opener)
    data.write_jsonl(rows, args.out)
    print(f"{len(rows)} training rows -> {args.out}")

    if held:
        if args.eval_out:
            # the anchor reads raw items, not rows: it needs the gold answer and
            # runs its own three conditions
            data.write_jsonl(held, args.eval_out)
            print(f"{len(held)} held-out items -> {args.eval_out}")
        else:
            print(f"{len(held)} held-out items dropped (pass --eval-out to keep them)")

    if args.push:
        data.push(rows, args.push)
        print(f"pushed to {args.push}")


if __name__ == "__main__":
    main()
