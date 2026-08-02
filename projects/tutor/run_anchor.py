"""The anchor: does the tutor actually help, on problems it never trained on?

    python -m projects.tutor.run_anchor \
        --items data/state_tests/eval_items.jsonl \
        --tutor-model  <checkpoint or base>  --tutor-url  http://localhost:8000/v1 \
        --student-model Qwen/Qwen2.5-0.5B-Instruct --student-url http://localhost:8001/v1

Run it before training and after, on the same items with the same student, and
compare. Nothing here is part of the reward, and nothing in the reward is here.
That separation is the only reason the number can say "this did not work".

WHAT IT PRINTS.

    baseline     the frozen student's accuracy with no tutoring
    treated      its accuracy given this problem's tutor turns
    swapped      its accuracy given ANOTHER problem's tutor turns
    gain         treated - baseline
    specificity  treated - swapped
    leaked       fraction of dialogues that gave the answer away
    clean_leaked treated, restricted to dialogues that did NOT leak

Read the last two together with the first. A rising ``treated`` while
``clean_leaked`` stays flat is not teaching improving - it is leaking falling,
and the leaked solves disappearing from the numerator. That decomposition is
what four runs of this project turned on: leak rates halved every time and the
honest solve rate never moved.

Read ``gain`` next to ``specificity``. In the most careful measurement here,
across nine evals of 235 items: no hint 0.489, a hint written for a DIFFERENT
problem 0.495, the tutor's own hint 0.552. So 91% of a +0.063 gain was
question-specific - the help is real, it is just small, and training did not
grow it.

And read both next to the standard error, which is printed. At n=40 the smallest
trustworthy difference is about 0.16, which is larger than any effect here.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from open_instruct.scored_rewards.anchor import Anchor, moved
from projects.tutor import leak
from projects.tutor.build_dataset import TEACHER_SYSTEM, user_view
from projects.tutor.student import ChoiceStudent


def tutor_policy(model: str, base_url: str | None, api_key: str | None, max_tokens: int = 256, concurrency: int = 16):
    """Ask the tutor for its turns on each item. One shot, no dialogue.

    Single-turn on purpose: the anchor has to be identical across runs and
    across the three conditions, and a dialogue is not reproducible enough to
    be a measuring instrument. It measures the tutor's advice, not its
    conversational skill.
    """
    import openai  # noqa: PLC0415

    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    limiter = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> str:
        async with limiter:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": user_view(item)}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()

    async def policy(items):
        return list(await asyncio.gather(*(one(i) for i in items)))

    return policy


def solved_outcome(student: ChoiceStudent):
    """The outcome channel. Length-normalised log-prob over the options.

    Stricter than the student's real ability, and deliberately unchanged since
    the first run: every baseline in this project's history was taken with it.
    """

    async def outcome(item: dict, hint: str) -> float:
        picked = await student.choose(item["question"], item["choices"], hint=hint)
        return float(picked == item["gold_idx"])

    return outcome


async def run(args) -> None:
    with open(args.items) as handle:
        items = [json.loads(line) for line in handle if line.strip()][: args.n]
    student = ChoiceStudent(args.student_model, args.student_url, args.api_key)

    anchor = Anchor(
        items=items,
        policy=tutor_policy(args.tutor_model, args.tutor_url, args.api_key, args.max_tokens, args.concurrency),
        outcome=solved_outcome(student),
        extra_metrics={"leaked": lambda item, text: float(leak.leaked_item(text, item))},
        concurrency=args.concurrency,
    )
    result = await anchor.run()
    print(result)
    print(json.dumps(result.to_dict(), indent=2))

    if args.compare_to:
        with open(args.compare_to) as f:
            previous = json.load(f)
        before = type(result)(
            n=int(previous["anchor/n"]),
            baseline=previous["anchor/baseline"],
            treated=previous["anchor/treated"],
            swapped=previous["anchor/swapped"],
        )
        print(moved(before, result))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True, help="held-out items, one JSON object per line")
    parser.add_argument("--n", type=int, default=235, help="~200 is where the noise floor gets useful")
    parser.add_argument("--tutor-model", required=True)
    parser.add_argument("--tutor-url", default=None)
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--student-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", default=None, help="write the metrics as JSON")
    parser.add_argument("--compare-to", default=None, help="a previous --out, to print the movement in SE")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
