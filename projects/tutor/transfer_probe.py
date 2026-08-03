"""Does tutoring about problem A move the student on a DIFFERENT problem B?

This is a measurement, not a training change, and it is meant to be run before
committing to a redesign. It needs no tutor generation - it re-scores dialogues
that already exist - so the only model running is the answering student.

WHY ASK THIS. Five runs moved `teacher_acc` by nothing, and `why_flat.md`
measured the reason: rated tutoring quality is uncorrelated with the student
solving (r = -0.012) while rated LEAKING correlates at +0.291. The outcome has
one channel, information content, and the tutor's only way to use it is to give
the answer away.

A transfer item closes that channel by construction rather than by penalty.
Telling the student "the answer is copper" resolves item A and is worth exactly
nothing on item B. Only a stated principle - why the belief behind the wrong
option is wrong - can carry across. So:

    if transfer is flat        the environment has ONE channel and it is leakage.
                               No reward shaping fixes that; the outcome measure
                               has to change, or the project reports the negative
                               result. Either way, do not run another GRPO job.
    if transfer is positive    there is a pedagogy channel that the current
                               outcome measure cannot see, and it is worth
                               rebuilding the reward around it.

Both answers are worth having and neither costs a training run.

THE THREE CONDITIONS, all measured on item B:

    baseline   B alone
    treated    B, given a real dialogue about A, where A shares B's knowledge unit
    swapped    B, given a real dialogue about a DIFFERENT unit

``swapped`` is not optional. A dialogue - any dialogue - primes carefulness,
supplies vocabulary and demonstrates working, and that is worth something on any
item. `treated - swapped` is the part that required A and B to be about the same
thing, which is the only part that means teaching.

WHAT WOULD INVALIDATE THE RESULT, checked and reported rather than assumed:

- **A leaked dialogue helping B.** If dialogues that gave away A's answer lift B
  as much as clean ones do, the pairing is contaminated - the two items likely
  share an answer or the units are too broad - and the headline is meaningless.
  ``--report`` splits on this and it is the first row to read.
- **The unit tags being wrong.** Silent, and everything inherits it. Spot-check
  the pairs printed by ``--show-pairs`` before believing a number.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
from collections.abc import Sequence

from open_instruct.scored_rewards.anchor import Anchor, AnchorResult
from projects.tutor import units as units_mod
from projects.tutor.student import ChoiceStudent


class TransferStudent(ChoiceStudent):
    """The answering channel, told the context is about a different problem.

    ``ChoiceStudent.PROMPT`` says ``Fact: {hint}`` because there the hint IS about
    the question. Here it is a conversation about something else, and calling
    that a fact about B invites the model to read across as if it were. Subclassed
    rather than edited so the anchor's own instrument stays byte-identical - every
    number in the run history was taken with it.
    """

    PROMPT = (
        "Here is a tutoring conversation about a different problem:\n"
        "{hint}\n\n"
        "Now answer this question.\nQuestion: {question}\nAnswer:"
    )


def build_pairs(
    items: Sequence[dict],
    units: dict[str, units_mod.Unit],
    dialogues: dict[str, list[dict]],
    *,
    seed: int = 0,
    max_per_unit: int = 12,
    sources_per_target: int = 3,
) -> list[dict]:
    """Pair items that share a knowledge unit, where the source has a dialogue.

    ``sources_per_target`` is what makes the probe resolvable. One source per
    target gives 56 pairs on this corpus, and at n=56 one standard error is 0.067
    so nothing smaller than a 0.13 difference can be read - larger than the effect
    being looked for. Pairing each target against several same-unit sources
    multiplies n at no annotation cost, because the dialogues already exist.

    Those extra measurements are not independent - they reuse the target item - so
    the true standard error lies between the one implied by the pair count and the
    one implied by the count of distinct targets. Both are printed, and a result
    significant only on the smaller of the two is not proven.

    ``max_per_unit`` caps how many pairs one unit contributes. Without it a
    single broad unit ("basic arithmetic") holding 80 items dominates the mean,
    and the result describes that unit rather than the corpus.
    """
    rng = random.Random(seed)
    by_unit: dict[str, list[dict]] = collections.defaultdict(list)
    for item in items:
        key = units_mod.item_key(item)
        unit = units.get(key)
        if unit and unit.primary:
            by_unit[unit.primary].append(item)

    pairs: list[dict] = []
    for unit_name, group in sorted(by_unit.items()):
        if len(group) < 2:
            continue
        candidates = []
        for target in group:
            target_key = units_mod.item_key(target)
            sources = [
                i for i in group if units_mod.item_key(i) != target_key and dialogues.get(units_mod.item_key(i))
            ]
            if not sources:
                continue
            rng.shuffle(sources)
            for source in sources[:sources_per_target]:
                candidates.append(
                    {
                        "unit": unit_name,
                        "target": target,
                        "target_key": target_key,
                        "source_key": units_mod.item_key(source),
                        "source_question": source.get("question", ""),
                    }
                )
        rng.shuffle(candidates)
        pairs.extend(candidates[:max_per_unit])
    rng.shuffle(pairs)
    return pairs


def pick_dialogue(entries: Sequence[dict], rng: random.Random, prefer_clean: bool = True) -> dict:
    """One dialogue about the source item.

    Prefers a non-leaking one: a dialogue that gave A's answer away is a poor
    test of transfer, since there is nothing to transfer but a number that does
    not apply to B. Both kinds are reported separately either way.
    """
    clean = [e for e in entries if not e.get("leaked")]
    pool = clean if (prefer_clean and clean) else list(entries)
    return rng.choice(pool)


def load_dialogues(path: str, tier: str | None = "policy") -> dict[str, list[dict]]:
    by_item: dict[str, list[dict]] = collections.defaultdict(list)
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if tier and row.get("tier") != tier:
                continue
            text = str(row.get("completion") or "").strip()
            if text:
                by_item[units_mod.item_key({"question": row.get("prompt", "")})].append(row)
    return dict(by_item)


def make_swap(units: dict[str, units_mod.Unit], seed: int = 0):
    """A foreign dialogue from a DIFFERENT unit.

    Anchor's default rotation is wrong here: pairs are grouped by unit and then
    shuffled, so rotating lands on a same-unit dialogue often enough to eat the
    specificity term. This picks explicitly across units.
    """

    def swap(pairs: Sequence[dict], outputs: Sequence[str]) -> list[str]:
        rng = random.Random(seed)
        indices = list(range(len(pairs)))
        out: list[str] = []
        for pair in pairs:
            foreign = [j for j in indices if pairs[j]["unit"] != pair["unit"] and outputs[j]]
            out.append(outputs[rng.choice(foreign)] if foreign else "")
        return out

    return swap


async def run(args) -> None:
    with open(args.items) as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    units = units_mod.load(args.units)
    dialogues = load_dialogues(args.traces, tier=args.tier)
    print(f"{len(items)} items, {len(units)} annotated, dialogues for {len(dialogues)} items")

    pairs = build_pairs(
        items,
        units,
        dialogues,
        seed=args.seed,
        max_per_unit=args.max_per_unit,
        sources_per_target=args.sources_per_target,
    )
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(
            "no pairs. Either the unit tags are all unique (check `units.py --out`'s "
            "coverage report) or no source item has a dialogue in the traces."
        )

    rng = random.Random(args.seed)
    chosen = {p["source_key"]: pick_dialogue(dialogues[p["source_key"]], rng) for p in pairs}
    leaked = {k: bool(v.get("leaked")) for k, v in chosen.items()}
    targets = len({p["target_key"] for p in pairs})
    print(
        f"{len(pairs)} pairs over {len({p['unit'] for p in pairs})} units, {targets} distinct targets\n"
        f"  1 SE = {(0.25 / len(pairs)) ** 0.5:.3f} by pairs, {(0.25 / targets) ** 0.5:.3f} by distinct targets"
    )

    if args.show_pairs:
        for p in pairs[: args.show_pairs]:
            print(f"\n  unit: {p['unit']}")
            print(f"    taught on: {p['source_question'][:90]}")
            print(f"    tested on: {p['target'].get('question', '')[:90]}")

    if args.dry_run:
        return

    student = TransferStudent(args.student_model, args.student_url, args.api_key)

    async def policy(ps: Sequence[dict]) -> list[str]:
        return [str(chosen[p["source_key"]].get("completion") or "") for p in ps]

    async def outcome(pair: dict, text: str) -> float:
        item = pair["target"]
        picked = await student.choose(item["question"], item["choices"], hint=text)
        return float(picked == item["gold_idx"])

    anchor = Anchor(
        items=pairs, policy=policy, outcome=outcome, concurrency=args.concurrency, swap=make_swap(units, args.seed)
    )
    result = await anchor.run()
    print("\n=== transfer ===")
    print(result)
    verdict(result, targets)

    if args.report:
        await split_by_leak(pairs, chosen, leaked, student, units, args)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({**result.to_dict(), "pairs": len(pairs)}, handle, indent=2)
        print(f"\nwrote {args.out}")


async def split_by_leak(pairs, chosen, leaked, student, units, args) -> None:
    """The contamination check. Read this before the headline.

    A dialogue that leaked A's answer should carry NOTHING to B. If it carries as
    much as a clean dialogue, the two items share more than a knowledge unit and
    the pairing is not measuring transfer.
    """
    groups = {
        "clean source": [p for p in pairs if not leaked[p["source_key"]]],
        "leaked source": [p for p in pairs if leaked[p["source_key"]]],
    }
    print("\n=== by whether the source dialogue leaked ===")
    for name, subset in groups.items():
        if len(subset) < 20:
            print(f"  {name:<15} n={len(subset)} - too few to read")
            continue

        async def policy(ps: Sequence[dict]) -> list[str]:
            return [str(chosen[p["source_key"]].get("completion") or "") for p in ps]

        async def outcome(pair: dict, text: str) -> float:
            item = pair["target"]
            picked = await student.choose(item["question"], item["choices"], hint=text)
            return float(picked == item["gold_idx"])

        sub = await Anchor(
            items=subset,
            policy=policy,
            outcome=outcome,
            concurrency=args.concurrency,
            swap=make_swap(units, args.seed),
        ).run()
        print(f"  {name:<15} {sub}")


def verdict(result: AnchorResult, distinct_targets: int | None = None) -> None:
    """State the reading in standard errors, so it cannot be talked up later.

    Judged against the CONSERVATIVE standard error - the one implied by distinct
    target items rather than by pairs - because pairs sharing a target are not
    independent observations and the pair-count SE flatters the result.
    """
    se = result.standard_error
    if distinct_targets:
        se = max(se, (0.25 / distinct_targets) ** 0.5)
    gain, spec = result.gain, result.specificity
    print(f"\n  gain        {gain:+.3f} = {gain / se:+.1f} SE")
    print(f"  specificity {spec:+.3f} = {spec / se:+.1f} SE   (conservative SE = {se:.3f})")
    if abs(spec) < 2 * se:
        print(
            "\n  -> NO transferable teaching signal at this n.\n"
            "     Same-unit tutoring is worth no more than tutoring about anything\n"
            "     else, so the outcome measure still has only the one channel.\n"
            "     Do not run another GRPO job against it."
        )
    else:
        print(
            "\n  -> There IS a unit-specific transfer effect.\n"
            "     Leakage cannot produce this, so it is a pedagogy channel the\n"
            "     current outcome measure cannot see. Worth rebuilding around."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--units", required=True, help="output of units.py")
    parser.add_argument("--traces", required=True, help="gen_traces.jsonl")
    parser.add_argument("--tier", default="policy", help="which tier of dialogue to use; empty for all")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--student-url", default="http://localhost:8001/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-unit", type=int, default=12)
    parser.add_argument("--sources-per-target", type=int, default=3, help="same-unit dialogues tested per target item")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--show-pairs", type=int, default=5, help="print this many pairs to eyeball the unit tags")
    parser.add_argument("--report", action="store_true", help="also split by whether the source dialogue leaked")
    parser.add_argument("--dry-run", action="store_true", help="build pairs and stop; no model needed")
    parser.add_argument("--out", default=None)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
