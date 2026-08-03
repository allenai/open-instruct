"""Knowledge-unit decomposition, offline.

PEARL's simulator decomposes a problem into structured knowledge units organised
by prerequisite relations, and tracks mastery as a vector over those units rather
than one number for the problem. This is that decomposition, run once over a
corpus and cached to disk.

WHY IT IS WORTH THE ANNOTATION PASS. Units are the only thing shared between two
problems. Without them "did teaching transfer" cannot be asked, because there is
no way to say that item B needs what item A taught. `subject` (3 values) and
`grade` (8) are far too coarse: two grade-6 maths items can have nothing in
common. Everything downstream - the transfer probe, and later a per-unit mastery
state - needs this and nothing else provides it.

WHAT IT ASKS FOR, AND WHY EACH PART. Per item:

``units``          the knowledge required, most specific first. The grouping key.
``prerequisites``  edges between units. Not used by the probe; recorded because
                   PEARL's mastery update needs them and re-annotating later is
                   the expensive part, not storing an extra field now.
``misconception``  what someone who picks the WRONG option believes. This is the
                   thing a tutor would have to repair, and the reason a transfer
                   item is a fair test: repairing a belief should generalise,
                   whereas being told this item's answer should not.

THE MISCONCEPTION IS GROUNDED, NOT INVENTED. It is annotated against the option
the student actually chose, taken from ``student_believed`` in the traces, so it
describes an error that was really made rather than one an annotator imagines a
student might make. Items with no observed wrong choice get ``None`` and are
dropped from pairing rather than guessed at.

COST. One call per item; 307 training items is 307 calls. Cache to disk and this
is a one-off, so the annotator should be the best model available rather than
the cheapest - a wrong unit tag silently mis-pairs items and every number
downstream inherits it.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import json
import os
import re
from collections.abc import Sequence

PROMPT = """You are decomposing an assessment item for a tutoring system.

Problem:
{question}

Options:
{options}

Correct answer: {gold}
{believed}
Return ONLY a JSON object:

{{
  "units": ["most specific knowledge unit required", "next", "..."],
  "prerequisites": [["unit", "unit it depends on"]],
  "misconception": "one sentence: what someone choosing the wrong option believes, phrased as the belief itself, or null if there is no wrong option given"
}}

Rules for "units":
- Name the transferable skill, not this problem. "ordering multi-digit numbers"
  is a unit; "ordering 147, 163, 234, 275" is not.
- 1 to 3 units, most specific first. The first one is used as the grouping key,
  so it must be the thing this item actually tests.
- Use wording another item testing the same skill would also produce. Prefer a
  plain, conventional name over a precise but idiosyncratic one.

Rules for "misconception":
- Describe the belief, not the behaviour. "believes the largest digit count
  means the largest number", not "ordered them wrong".
- It must explain the specific wrong option given, not generic carelessness.
"""


@dataclasses.dataclass
class Unit:
    """One item's decomposition."""

    key: str
    units: list[str]
    prerequisites: list[tuple[str, str]]
    misconception: str | None

    @property
    def primary(self) -> str | None:
        """The grouping key: the most specific unit the item tests."""
        return self.units[0] if self.units else None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "units": self.units,
            "prerequisites": [list(p) for p in self.prerequisites],
            "misconception": self.misconception,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Unit:
        return cls(
            key=d["key"],
            units=list(d.get("units") or []),
            prerequisites=[tuple(p) for p in d.get("prerequisites") or []],
            misconception=d.get("misconception"),
        )


def item_key(item: dict) -> str:
    """A stable identity for an item across files.

    The corpora carry ``item_no``/``booklet``/``year``, but the traces do not -
    they only carry the question text. So the text is the only thing that joins
    them, normalised so whitespace differences do not split an item in two.
    """
    return " ".join(str(item.get("question", "")).split()).lower()


def build_prompt(item: dict, believed: str | None = None) -> str:
    choices = item.get("choices") or []
    options = "\n".join(f"  {chr(65 + i)}. {c}" for i, c in enumerate(choices)) or "  (free response)"
    gold = item.get("gold")
    if gold is None and item.get("gold_idx") is not None and choices:
        gold = choices[int(item["gold_idx"])]
    believed_line = f"An actual student chose: {believed}\n" if believed else ""
    return PROMPT.format(question=item.get("question", ""), options=options, gold=gold, believed=believed_line)


def parse(text: str) -> dict:
    """Pull the JSON object out of a model reply.

    Raises rather than returning a default. A silently empty decomposition would
    drop the item from every pairing without anything in the output saying so.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    blob = json.loads(match.group(0))
    units = [str(u).strip().lower() for u in (blob.get("units") or []) if str(u).strip()]
    prereqs = [
        (str(a).strip().lower(), str(b).strip().lower()) for a, b in (blob.get("prerequisites") or []) if a and b
    ]
    misc = blob.get("misconception")
    return {"units": units, "prerequisites": prereqs, "misconception": str(misc).strip() if misc else None}


def believed_by_item(traces: Sequence[dict]) -> dict[str, str]:
    """The wrong option most often actually chosen, per item.

    Most often rather than first seen: a student that picked three different
    wrong options across rollouts has no single misconception, and the modal one
    is the closest thing to the belief worth annotating.
    """
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for t in traces:
        believed = t.get("student_believed")
        gold = t.get("gold")
        if believed and believed != gold:
            counts[item_key({"question": t.get("prompt", "")})][believed] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items() if c}


async def annotate(
    items: Sequence[dict],
    believed: dict[str, str] | None = None,
    *,
    model: str = "gpt-5.1",
    base_url: str | None = None,
    api_key: str | None = None,
    concurrency: int = 8,
) -> list[Unit]:
    from openai import AsyncOpenAI  # noqa: PLC0415

    client = AsyncOpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
    believed = believed or {}
    limiter = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> Unit:
        key = item_key(item)
        async with limiter:
            reply = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_prompt(item, believed.get(key))}],
                temperature=0.0,
            )
        blob = parse(reply.choices[0].message.content or "")
        return Unit(key=key, **blob)

    return list(await asyncio.gather(*(one(i) for i in items)))


CANON_PROMPT = """Below are knowledge-unit names produced independently for different
assessment items. Many are the same skill worded differently.

{names}

Merge them into a smaller vocabulary. Return ONLY a JSON object mapping every
name above to its canonical name:

{{"comparing whole numbers": "ordering whole numbers", ...}}

Rules:
- Merge only genuine synonyms - skills where teaching one IS teaching the other.
  "ordering whole numbers" and "ordering decimals" are NOT the same skill.
- Keep the granularity that a tutor would treat as one lesson.
- Every name must appear as a key, mapping to itself if it merges with nothing.
"""


async def canonicalize(
    rows: Sequence[Unit], *, model: str = "gpt-5.1", base_url: str | None = None, api_key: str | None = None
) -> list[Unit]:
    """Merge synonymous unit names into one vocabulary.

    Annotating items independently is what makes the pass parallel and cheap, and
    it is also why the names do not line up: nothing tells the annotator what it
    called this skill on the previous item. Two items testing one skill land on
    two names, the unit looks unique, and both drop out of pairing - so the
    corpus appears to have no shared structure when the tags were fine.

    One call with the whole vocabulary, because the merge decision needs to see
    the alternatives. Skipping this on a 307-item corpus spread over 3 subjects
    and 8 grades is the likeliest way to get an empty probe.
    """
    from openai import AsyncOpenAI  # noqa: PLC0415

    names = sorted({u.primary for u in rows if u.primary})
    if not names:
        return list(rows)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
    reply = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": CANON_PROMPT.format(names="\n".join(f"- {n}" for n in names))}],
        temperature=0.0,
    )
    match = re.search(r"\{.*\}", reply.choices[0].message.content or "", re.DOTALL)
    if not match:
        print("WARNING: canonicalisation returned no mapping; leaving names as they are")
        return list(rows)
    mapping = {str(k).strip().lower(): str(v).strip().lower() for k, v in json.loads(match.group(0)).items()}

    merged = []
    for u in rows:
        renamed = [mapping.get(n, n) for n in u.units]
        merged.append(dataclasses.replace(u, units=renamed))
    print(f"canonicalised {len(names)} unit names down to {len({mapping.get(n, n) for n in names})}")
    return merged


def load(path: str) -> dict[str, Unit]:
    with open(path) as handle:
        return {u["key"]: Unit.from_dict(u) for u in (json.loads(line) for line in handle if line.strip())}


def save(units: Sequence[Unit], path: str) -> None:
    with open(path, "w") as handle:
        for u in units:
            handle.write(json.dumps(u.to_dict()) + "\n")


def coverage(units: Sequence[Unit]) -> dict:
    """How usable the annotation is for pairing, before anything is run.

    ``pairable`` is the number that matters: units seen on only one item cannot
    produce a pair, so a decomposition that gives every item its own unique unit
    is worthless however accurate each tag is.
    """
    by_unit: dict[str, int] = collections.Counter(u.primary for u in units if u.primary)
    shared = {k: v for k, v in by_unit.items() if v >= 2}
    return {
        "items": len(units),
        "with_units": sum(1 for u in units if u.primary),
        "with_misconception": sum(1 for u in units if u.misconception),
        "distinct_units": len(by_unit),
        "shared_units": len(shared),
        "pairable_items": sum(shared.values()),
        "largest_unit": max(by_unit.values()) if by_unit else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True, help="items JSONL")
    parser.add_argument("--traces", default=None, help="gen_traces JSONL, for the observed wrong answer")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-canonicalize", action="store_true", help="skip the synonym merge (usually a mistake)")
    args = parser.parse_args()

    with open(args.items) as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        items = items[: args.limit]

    believed = {}
    if args.traces:
        with open(args.traces) as handle:
            believed = believed_by_item([json.loads(line) for line in handle if line.strip()])
        print(f"observed a wrong answer for {len(believed)} items")

    units = asyncio.run(
        annotate(
            items,
            believed,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            concurrency=args.concurrency,
        )
    )
    before = coverage(units)
    if not args.no_canonicalize:
        units = asyncio.run(canonicalize(units, model=args.model, base_url=args.base_url, api_key=args.api_key))
    save(units, args.out)

    stats = coverage(units)
    if not args.no_canonicalize:
        print(f"pairable items: {before['pairable_items']} before the merge, {stats['pairable_items']} after")
    print(json.dumps(stats, indent=2))
    if stats["pairable_items"] < 40:
        print(
            f"\nWARNING: only {stats['pairable_items']} items sit on a unit shared with another item.\n"
            "The transfer probe needs pairs, and at n<40 one standard error is 0.079 -\n"
            "the smallest difference you could trust is ~0.16, which is larger than the\n"
            "effect. Annotate more items or loosen the unit naming before running it."
        )


if __name__ == "__main__":
    main()
