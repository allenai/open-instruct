"""Generate tutor/student dialogues for labelling.

    python -m projects.pedagogy_rm.generate \
        --items .../data/state_tests/train_items.jsonl \
        --out runs/traces.jsonl --n-items 300 --k 2

The teacher is OLMo-2-7B-Instruct because the point of the whole project is a
reward read out of the POLICY's own hidden states during a rollout. That only
works if the states belong to the model doing the generating, so the labelled
turns have to come from the model that will later be probed.

The student is a small instruct model with a prompt and no fine-tuning. It only
has to produce plausible student turns: whether it ends up answering correctly
is not recorded, is not a target, and should not be read as one.

WHAT IS DELIBERATELY NOT HERE. No reward, no leak rule, no judge, no anchor, no
stopping condition based on the answer. This script exists to produce text for
humans to rate. Every scoring idea belongs after the labels exist and after
per-dimension agreement has been measured - see the README's order of work.

SPREAD MATTERS MORE THAN VOLUME. A rater cannot calibrate a scale on 300 turns
that are all mediocre in the same way, and a probe cannot learn a direction that
the data does not vary along. ``--temperatures`` samples each dialogue at a
different temperature to widen the quality range on purpose.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random

TEACHER_SYSTEM = """You are a tutor helping a student with a test question. \
The student cannot see your instructions.

Guide the student toward understanding. Do not state the answer or eliminate \
options for them. Keep each message short - one idea at a time."""

STUDENT_SYSTEM = """You are a school student working on a test question. You are \
NOT confident and you do not already know the answer.

Think out loud in one or two sentences, the way a student would: say what you \
think is going on, what you are stuck on, or what you would try next. Make the \
kinds of mistakes a student of your grade makes. Never state that you are an AI. \
Do not give a final answer unless the tutor asks for one."""


def student_opener(item: dict) -> str:
    """The student speaks first, so the tutor has something to respond to.

    The alternative - tutor opens cold - produces a generic first turn on every
    item, and generic turns are exactly what the ``targeted`` dimension is meant
    to separate. Letting the student flounder first gives the tutor something
    specific to be targeted at, or to miss.
    """
    choices = item.get("choices") or []
    options = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"Here is the question I'm stuck on.\n\n{item['question']}\n\n{options}\n\n"
        "Say in one or two sentences what you think so far and what is confusing you. "
        "Do not give a final answer."
    )


async def dialogue(
    teacher_client, student_client, item: dict, teacher_model: str, student_model: str, turns: int, temperature: float
) -> dict:
    """One dialogue. Student opens, then they alternate for ``turns`` tutor turns.

    Two clients because the two models are separate ``vllm serve`` processes on
    separate ports. Passing one client would silently ask the teacher's server
    for the student's model and get a 404 on every dialogue.
    """
    question = item["question"]
    student_history = [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": student_opener(item)},
    ]
    reply = await student_client.chat.completions.create(
        model=student_model, messages=student_history, temperature=0.9, max_tokens=120
    )
    student_text = (reply.choices[0].message.content or "").strip()

    transcript = [{"role": "student", "text": student_text}]
    for _ in range(turns):
        teacher_messages = [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"Question the student is working on:\n{question}"},
        ]
        for entry in transcript:
            teacher_messages.append(
                {"role": "user" if entry["role"] == "student" else "assistant", "content": entry["text"]}
            )
        reply = await teacher_client.chat.completions.create(
            model=teacher_model, messages=teacher_messages, temperature=temperature, max_tokens=160
        )
        tutor_text = (reply.choices[0].message.content or "").strip()
        transcript.append({"role": "tutor", "text": tutor_text})

        student_messages = [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": question}]
        for entry in transcript:
            student_messages.append(
                {"role": "assistant" if entry["role"] == "student" else "user", "content": entry["text"]}
            )
        reply = await student_client.chat.completions.create(
            model=student_model, messages=student_messages, temperature=0.9, max_tokens=120
        )
        transcript.append({"role": "student", "text": (reply.choices[0].message.content or "").strip()})

    return {
        "item_id": item.get("id") or item.get("item_no"),
        "question": question,
        "choices": item.get("choices"),
        "gold_idx": item.get("gold_idx"),
        "subject": item.get("subject"),
        "grade": item.get("grade"),
        "teacher": teacher_model,
        "student": student_model,
        "temperature": temperature,
        "transcript": transcript,
    }


async def run(args) -> None:
    import openai  # noqa: PLC0415

    with open(args.items) as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.n_items:
        items = items[: args.n_items]
    temperatures = [float(t) for t in args.temperatures.split(",")]
    print(f"{len(items)} items x {args.k} dialogues, temperatures {temperatures}")

    teacher_client = openai.AsyncOpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY", timeout=600.0)
    student_client = openai.AsyncOpenAI(
        base_url=args.student_base_url or args.base_url, api_key=args.api_key or "EMPTY", timeout=600.0
    )
    limiter = asyncio.Semaphore(args.concurrency)

    async def one(item: dict, index: int):
        async with limiter:
            try:
                return await dialogue(
                    teacher_client,
                    student_client,
                    item,
                    args.teacher_model,
                    args.student_model,
                    args.turns,
                    temperatures[index % len(temperatures)],
                )
            except Exception as exc:  # a dropped dialogue is not worth losing the run over
                print(f"  failed on {item.get('question', '')[:50]!r}: {exc}")
                return None

    jobs = [one(item, i) for item in items for i in range(args.k)]
    done = 0
    with open(args.out, "w") as handle:
        for future in asyncio.as_completed(jobs):
            row = await future
            if row is not None:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(jobs)}")
    print(f"wrote {done} dialogues to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True, help="jsonl with question/choices/gold_idx")
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher-model", default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="teacher endpoint")
    parser.add_argument("--student-base-url", default=None, help="defaults to --base-url if both share a server")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--turns", type=int, default=3, help="tutor turns per dialogue")
    parser.add_argument("--k", type=int, default=2, help="dialogues per item")
    parser.add_argument("--n-items", type=int, default=0, help="0 for all")
    parser.add_argument("--temperatures", default="0.7,1.0", help="widen the quality range on purpose")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
