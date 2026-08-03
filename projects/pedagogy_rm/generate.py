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
that are all mediocre in the same way, and a probe cannot learn a direction the
data does not vary along. The first run made this concrete: one prompt at two
temperatures gave 1,800 turns with a median of 94 words and almost no short
ones, so ``concise`` would have been a constant and ``elicits`` nearly one. Two
knobs widen it deliberately:

    --temperatures  varies how careful the turn is
    STYLES          varies what KIND of turn it is - a question, a lecture, one
                    sentence, or unconstrained

The styles are not four prompts hoping for variety. Each one targets a rubric
dimension that would otherwise be degenerate: ``socratic`` and ``explain`` sit at
opposite ends of ``elicits``, ``brief`` and ``explain`` at opposite ends of
``concise``. Sampling per dialogue means the same item appears in several styles,
so a rater sees the contrast rather than having to imagine it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random

TEACHER_SYSTEM = """You are a tutor helping a student with a test question. \
The student cannot see your instructions.

Guide the student toward understanding. Do not state the answer and do not \
eliminate options for them."""

# Appended to the teacher's system prompt, one per dialogue. See the module
# docstring: these exist to put spread into dimensions that would otherwise be
# constant, not to find the single best tutor prompt.
# Generous on purpose. Capping the student at 60 tokens to force brevity backfired
# badly: it left student turns ending mid-sentence, and OLMo then CONTINUED the
# sentence instead of replying to it, tripling mid-sentence tutor turns to 9%.
# Brevity is enforced afterwards by ``first_sentences``, which cannot cut a word
# in half.
STUDENT_MAX_TOKENS = 140
STUDENT_SENTENCES = 2

STYLES: dict[str, str] = {
    "plain": "Reply in at most three sentences.",
    "brief": "Reply with a single short sentence. No preamble, no summary.",
    "socratic": "Reply with exactly one question and nothing else. Never explain.",
    "explain": "Explain the underlying idea thoroughly before asking anything.",
}

# Instructions alone did not work: told to be brief and unsure, the 1.5B still
# produced 50-word textbook paragraphs that defined terms back at the tutor. A
# student who never says anything specific enough to be wrong about leaves the
# ``targeted`` dimension nothing to point at, so the voice is now demonstrated
# rather than described.
STUDENT_SYSTEM = """You are a school student working on a test question. You do \
not know the answer and you are not confident.

Write like these examples - short, unsure, specific, and often wrong:

  "Is it B? I feel like it's the one about pressure but I can't tell why."
  "I keep wanting to add them but that gives a number way too big."
  "I don't really get what 'per capita' is doing here."
  "We did something like this with the graph, but this one has no graph."

Rules: ONE OR TWO SHORT SENTENCES. Never define a term. Never explain a concept \
back. Never list what you know. Do not thank the tutor or praise them. Do not \
write like a textbook. Do not give a final answer unless the tutor asks."""


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


async def say(client, model: str, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, bool]:
    """One completion, plus whether it was cut off at the cap.

    Truncation has to be tracked, not ignored. The first run capped tutor turns
    at 160 tokens and 11% of them ended mid-word, which is unlabelable: a rater
    scoring ``concise`` on a turn the cap truncated is scoring the cap.
    """
    reply = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    choice = reply.choices[0]
    return (choice.message.content or "").strip(), choice.finish_reason == "length"


def first_sentences(text: str, n: int) -> str:
    """The first ``n`` complete sentences, never a partial one.

    This is what keeps the student short, because asking a 1.5B for one or two
    sentences and showing it examples both failed - it wrote textbook paragraphs
    either way. Trimming afterwards always works and, unlike a token cap, cannot
    leave a dangling clause for the teacher to finish.
    """
    out: list[str] = []
    start = 0
    for i, char in enumerate(text):
        if char in ".!?" and (i + 1 == len(text) or text[i + 1].isspace()):
            out.append(text[start : i + 1].strip())
            start = i + 1
            if len(out) >= n:
                break
    trimmed = " ".join(out).strip()
    # No terminator at all means one long run-on, possibly cut at the cap. Keep
    # it whole rather than returning nothing, but drop a half-finished last word.
    return trimmed or text.strip().rsplit(" ", 1)[0]


def teacher_view(question: str, transcript: list[dict], style: str) -> list[dict]:
    """The teacher's messages, strictly alternating user/assistant.

    The question is folded into the FIRST user message rather than sent as its
    own. Two user messages in a row is not something a chat template promises to
    handle, and OLMo's did not: the first run produced turns that began
    mid-sentence, in the middle of a word.
    """
    messages = [{"role": "system", "content": TEACHER_SYSTEM + "\n\n" + STYLES[style]}]
    for i, entry in enumerate(transcript):
        text = entry["text"]
        if i == 0:
            text = f"Question I'm working on:\n{question}\n\n{text}"
        messages.append({"role": "user" if entry["role"] == "student" else "assistant", "content": text})
    return messages


def student_view(item: dict, transcript: list[dict]) -> list[dict]:
    """The student's messages, strictly alternating, opening with the same prompt."""
    messages = [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": student_opener(item)}]
    for entry in transcript:
        messages.append({"role": "assistant" if entry["role"] == "student" else "user", "content": entry["text"]})
    return messages


async def dialogue(
    teacher_client,
    student_client,
    item: dict,
    teacher_model: str,
    student_model: str,
    turns: int,
    temperature: float,
    style: str,
) -> dict:
    """One dialogue. Student opens, then they alternate for ``turns`` tutor turns.

    Two clients because the two models are separate ``vllm serve`` processes on
    separate ports. Passing one client would silently ask the teacher's server
    for the student's model and get a 404 on every dialogue.
    """
    question = item["question"]
    text, _ = await say(student_client, student_model, student_view(item, []), 0.9, STUDENT_MAX_TOKENS)
    transcript: list[dict] = [{"role": "student", "text": first_sentences(text, STUDENT_SENTENCES)}]

    for _ in range(turns):
        # 400 tokens is well above what any style should need, so a turn that
        # still hits it is a runaway rather than a good turn spoiled by the cap.
        text, truncated = await say(
            teacher_client, teacher_model, teacher_view(question, transcript, style), temperature, 400
        )
        transcript.append({"role": "tutor", "text": text, "truncated": truncated})
        text, _ = await say(student_client, student_model, student_view(item, transcript), 0.9, STUDENT_MAX_TOKENS)
        transcript.append({"role": "student", "text": first_sentences(text, STUDENT_SENTENCES)})

    return {
        "style": style,
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
    styles = [s for s in args.styles.split(",") if s]
    unknown = set(styles) - set(STYLES)
    if unknown:
        raise SystemExit(f"unknown styles {sorted(unknown)}; have {sorted(STYLES)}")
    # Each (temperature, style) combination is walked in order rather than
    # sampled, so the k dialogues for an item differ from each other by
    # construction instead of by luck.
    grid = [(t, s) for s in styles for t in temperatures]
    print(f"{len(items)} items x {args.k} dialogues over {len(grid)} (temp, style) cells: {grid}")

    teacher_client = openai.AsyncOpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY", timeout=600.0)
    student_client = openai.AsyncOpenAI(
        base_url=args.student_base_url or args.base_url, api_key=args.api_key or "EMPTY", timeout=600.0
    )
    limiter = asyncio.Semaphore(args.concurrency)

    async def one(item: dict, index: int):
        async with limiter:
            temperature, style = grid[index % len(grid)]
            try:
                return await dialogue(
                    teacher_client,
                    student_client,
                    item,
                    args.teacher_model,
                    args.student_model,
                    args.turns,
                    temperature,
                    style,
                )
            except Exception as exc:  # a dropped dialogue is not worth losing the run over
                print(f"  failed on {item.get('question', '')[:50]!r}: {exc}")
                return None

    # Global counter, not a per-item one: indexing by the within-item repeat
    # would mean k=4 against an 8-cell grid never reaches cells 4-7, so two of
    # the four styles would silently never be generated.
    jobs = [one(item, n) for n, (item, _) in enumerate((it, i) for it in items for i in range(args.k))]
    done = cut = turns = 0
    with open(args.out, "w") as handle:
        for future in asyncio.as_completed(jobs):
            row = await future
            if row is not None:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(jobs)}")
                cut += sum(1 for t in row["transcript"] if t.get("truncated"))
                turns += sum(1 for t in row["transcript"] if t["role"] == "tutor")
    print(f"wrote {done} dialogues to {args.out}")
    print(f"truncated tutor turns: {cut}/{turns} — build_label_set drops these")


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
    parser.add_argument("--styles", default="plain,brief,socratic,explain", help=f"any of {sorted(STYLES)}")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
