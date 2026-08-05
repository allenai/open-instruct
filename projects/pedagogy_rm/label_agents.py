"""Six model raters, calibrated to your labels, scoring the rest of the pool.

    python -m projects.pedagogy_rm.label_agents \
        --units data/label_slices/slice_1.json --out-dir data/labels \
        --calibration data/calibration.md --examples data/labels/sophia.json \
        --holdout data/holdout.json

EACH RATER IS A DIFFERENT MODEL FAMILY, and that is not decoration. Agreement
between six samples of one model measures that model's self-consistency, which
is high for reasons that have nothing to do with the question being well posed.
Agreement between OpenAI, Anthropic, Google, xAI, DeepSeek and Qwen is much
harder to get for a bad reason, so it is worth something as evidence.

WHAT EACH RATER IS GIVEN, in order:

1. the rubric, generated from ``rubric.py`` so it cannot drift from the schema;
2. the calibration document, which is where YOUR boundaries live;
3. your labelled examples, as few-shot demonstrations;
4. one turn to rate.

Order matters. The rubric alone yields a model's own notion of good teaching.
The calibration and examples are what redirect it towards reproducing yours.

THE HOLDOUT IS EXCLUDED FROM THE EXAMPLES, not from the work. Agents rate those
items like any other; they are simply never shown your answers for them. That is
the only reason the eventual agreement number means anything.

FLAGGING IS ALLOWED, and worth reading. An agent that flags a dimension on many
turns is reporting the same thing a human would - that the question does not fit
- and that is information about the rubric, not noise to be suppressed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from projects.pedagogy_rm.rubric import BY_KEY, DIMENSIONS, rubric_markdown, validate

#: Six families. Picked for independence of failure, not for leaderboard rank.
DEFAULT_RATERS: dict[str, str] = {
    "gpt": "openai-group/gpt-5.6-terra",
    # sonnet-5 and opus-4-8 exist on the gateway but 403 with an IAM denial;
    # sonnet-4-6 is the strongest Anthropic model the key can actually reach.
    "claude": "claude-group/claude-sonnet-4-6",
    "gemini": "gemini-group/gemini-3.1-pro",
    "grok": "xai-group/grok-4.5",
    "deepseek": "bedrock-oss-group/deepseek-v3.2",
    "qwen": "bedrock-oss-group/qwen3-next-80b",
}

SYSTEM = """You are rating a single tutor turn against a fixed rubric, matching \
the judgement of one specific human rater as closely as you can.

Your own opinion about good teaching matters only where the rubric and the \
calibration notes are silent. Where they disagree with you, follow them.

Reply with ONLY a JSON object, no prose and no code fence:

{{"{keys}": <int>, "flag": "<empty string, or a short reason the rubric did not fit>"}}

Every dimension must be present and must be an integer in its stated range. Use \
"flag" for a turn the rubric genuinely cannot describe - not for one you find \
hard to score. A hard turn still gets your best number."""

TASK = """QUESTION: {question}

STUDENT SAID, immediately before: {student}

TUTOR TURN TO RATE: {tutor}"""


def build_messages(
    unit: dict, rubric: str, calibration: str, shots: list[tuple[dict, dict]], active: tuple = DIMENSIONS
) -> list[dict]:
    system = SYSTEM.format(keys='": <int>, "'.join(d.key for d in active))
    context = f"=== RUBRIC ===\n\n{rubric}"
    if calibration:
        context += (
            "\n\n=== HOW THE HUMAN RATER APPLIES IT ===\n\n"
            "These notes were derived from that rater's own labels. Where they are\n"
            "more specific than the rubric, they win.\n\n" + calibration
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": context}]
    messages.append({"role": "assistant", "content": "Understood. Send a turn and I will reply with JSON only."})
    for unit_shot, record in shots:
        messages.append(
            {
                "role": "user",
                "content": TASK.format(
                    question=" ".join(unit_shot["question"].split()),
                    student=" ".join(unit_shot["student_before"].split()),
                    tutor=" ".join(unit_shot["tutor_turn"].split()),
                ),
            }
        )
        answer = {d.key: record[d.key] for d in active if d.key in record}
        answer["flag"] = record.get("flag", "")
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    messages.append(
        {
            "role": "user",
            "content": TASK.format(
                question=" ".join(unit["question"].split()),
                student=" ".join(unit["student_before"].split()),
                tutor=" ".join(unit["tutor_turn"].split()),
            ),
        }
    )
    return messages


def parse(text: str) -> dict | None:
    """The JSON object in a reply, tolerating fences and stray prose around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        blob = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return blob if isinstance(blob, dict) else None


async def rate_one(
    client, model: str, unit: dict, rubric: str, calibration: str, shots, retries: int, active=DIMENSIONS
) -> dict | None:
    messages = build_messages(unit, rubric, calibration, shots, active)
    for attempt in range(retries + 1):
        try:
            reply = await client.chat.completions.create(
                model=model, messages=messages, temperature=0.0 if attempt == 0 else 0.3
            )
            blob = parse(reply.choices[0].message.content or "")
        except Exception as exc:
            if attempt == retries:
                print(f"    {model} errored on {unit['id']}: {str(exc)[:110]}")
                return None
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        if blob is None:
            continue
        record = {"id": unit["id"]}
        for dim in active:
            value = blob.get(dim.key)
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                value = int(value.strip())
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            record[dim.key] = max(dim.lo, min(dim.hi, value))  # clamp rather than discard an otherwise good record
        if blob.get("flag"):
            record["flag"] = str(blob["flag"])[:200]
        if not validate(record, active):
            return record
    return None


async def run_rater(name: str, model: str, units: list[dict], rubric, calibration, shots, args, active) -> None:
    from projects.pedagogy_rm import gateway  # noqa: PLC0415

    out_path = os.path.join(args.out_dir, f"agent_{name}.json")
    done: dict[str, dict] = {}
    # Tracked per dimension, because a partial re-rate reveals only THAT
    # dimension's answers. One flat list would retire units from the validation
    # set for dimensions whose answers the agent was never shown.
    previous_shots: dict[str, list[str]] = {}
    if os.path.exists(out_path) and not args.overwrite:
        with open(out_path) as handle:
            blob = json.load(handle)
        done = {r["id"]: r for r in blob.get("labels", [])}
        stored = blob.get("shots") or {}
        previous_shots = stored if isinstance(stored, dict) else {d.key: list(stored) for d in DIMENSIONS}
    # Re-rating a subset means every unit is due again, and the new scores are
    # merged into the existing records rather than replacing them - the other
    # dimensions have already been validated and should not move.
    partial = len(active) < len(DIMENSIONS)
    todo = list(units) if partial else [u for u in units if u["id"] not in done]
    if not todo:
        print(f"  {name}: already complete ({len(done)})")
        return

    client = gateway.make_client()
    limiter = asyncio.Semaphore(args.concurrency)
    failures = 0

    async def one(unit):
        nonlocal failures
        async with limiter:
            record = await rate_one(client, model, unit, rubric, calibration, shots, args.retries, active)
            if record is None:
                failures += 1
            return record

    results = await asyncio.gather(*(one(u) for u in todo))
    for record in results:
        if not record:
            continue
        if partial and record["id"] in done:
            done[record["id"]].update({k: v for k, v in record.items() if k != "id"})
        else:
            done[record["id"]] = record
    with open(out_path, "w") as handle:
        json.dump(
            {
                "schema": "pedagogy-rm/labels-v1",
                "rater": name,
                "model": model,
                # Recorded so agreement.py can refuse to score these units against
                # the human. An agent shown the answer reproduces it, which reads
                # as perfect agreement and means nothing.
                "shots": {
                    d.key: sorted(
                        set(previous_shots.get(d.key, [])) | ({u["id"] for u, _ in shots} if d in active else set())
                    )
                    for d in DIMENSIONS
                },
                "labels": list(done.values()),
            },
            handle,
            indent=1,
        )
    flagged = sum(1 for r in done.values() if r.get("flag"))
    print(f"  {name:<9} {len(done):>4} labels, {failures} unparseable, {flagged} flagged  -> {out_path}")


async def main_async(args) -> None:
    with open(args.units) as handle:
        units = json.load(handle)["units"]
    if args.limit:
        units = units[: args.limit]

    calibration = ""
    if args.calibration and os.path.exists(args.calibration):
        with open(args.calibration) as handle:
            calibration = handle.read()
    elif args.calibration:
        raise SystemExit(
            f"{args.calibration} does not exist. Run calibrate.py first - without it the agents\n"
            "rate by their own standards, which is not what this is for. Pass --calibration '' to\n"
            "override deliberately."
        )

    # `active` is resolved before the shots are chosen, and the order matters. An example is
    # usable when it carries the dimensions being rated, not when it carries all five of
    # DIMENSIONS: a run rating four of them against labels that deliberately omit `concise`
    # matched nothing under the old test and printed "0 few-shot examples", which reads like
    # a missing file rather than a filter that could never pass. Silently unguided agents are
    # the worst possible failure here, because the output looks the same.
    active = DIMENSIONS if not args.dimensions else tuple(BY_KEY[k] for k in args.dimensions.split(","))
    if len(active) < len(DIMENSIONS):
        print(f"re-rating only {[d.key for d in active]}; other dimensions are left as they are")

    shots: list[tuple[dict, dict]] = []
    if args.examples:
        by_id = {u["id"]: u for u in units}
        held = set()
        if args.holdout and os.path.exists(args.holdout):
            with open(args.holdout) as handle:
                held = set(json.load(handle)["ids"])
        with open(args.examples) as handle:
            for record in json.load(handle)["labels"]:
                if record["id"] in held or record["id"] not in by_id:
                    continue
                if all(d.key in record for d in active):
                    shots.append((by_id[record["id"]], record))
        shots = shots[: args.max_shots]
        print(f"{len(shots)} few-shot examples, {len(held)} holdout ids withheld from every rater")
        if args.examples and not shots:
            raise SystemExit(
                "no usable few-shot examples: every labelled record is either held out or missing "
                f"one of {[d.key for d in active]}. Rating unguided would produce the agents' own "
                "standards rather than yours, which is not what this is for."
            )
    raters = DEFAULT_RATERS if not args.raters else {k: DEFAULT_RATERS[k] for k in args.raters.split(",")}
    print(f"{len(units)} units x {len(raters)} raters: {', '.join(raters)}")
    os.makedirs(args.out_dir, exist_ok=True)
    rubric = rubric_markdown(active)
    await asyncio.gather(
        *(run_rater(n, m, units, rubric, calibration, shots, args, active) for n, m in raters.items())
    )
    print("\nNext: agreement.py over the agent files, and against your holdout.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", required=True)
    parser.add_argument("--out-dir", default="data/labels")
    parser.add_argument("--calibration", default="data/calibration.md")
    parser.add_argument("--examples", default="", help="your labels, used as few-shot demonstrations")
    parser.add_argument("--holdout", default="data/holdout.json")
    parser.add_argument("--raters", default="", help=f"subset of {sorted(DEFAULT_RATERS)}")
    parser.add_argument("--dimensions", default="", help="re-rate only these, merging into existing files")
    parser.add_argument("--max-shots", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="off by default, so a rerun resumes")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
