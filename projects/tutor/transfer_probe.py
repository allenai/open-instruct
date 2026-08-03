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
import math
import random
import statistics
from collections.abc import Sequence

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


class StubStudent:
    """A fake answerer, for checking the plumbing without a GPU.

    Not a measurement and cannot be mistaken for one - it answers by looking for
    the gold string in the text, which is the leakage channel and nothing else.
    Its only job is to make the full run() path executable on a laptop, so that a
    crash in the leak split or the output writing is found here rather than forty
    minutes into a cluster job.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    async def score_choices(self, question: str, choices: Sequence[str], hint: str = "") -> list[float]:
        lowered = hint.lower()
        rng = random.Random(f"{self.seed}{question}{len(hint)}")
        scores = [rng.uniform(-3.0, -1.0) for _ in choices]
        for i, choice in enumerate(choices):
            if str(choice).strip().lower() in lowered:
                scores[i] += 2.0
        return scores


class CachingStudent:
    """Scores each (question, hint) once and serves both outcome measures from it.

    The binary outcome and P(gold) come from the same option log-probabilities, so
    computing them separately would double the inference for no new information.
    The cache also makes the contamination check nearly free, since it re-uses the
    baseline condition the headline already paid for.
    """

    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[tuple[str, str], list[float]] = {}

    async def scores(self, question: str, choices: Sequence[str], hint: str) -> list[float]:
        key = (question, hint)
        if key not in self.cache:
            self.cache[key] = await self.inner.score_choices(question, choices, hint)
        return self.cache[key]

    async def correct(self, item: dict, hint: str) -> float:
        scores = await self.scores(item["question"], item["choices"], hint)
        return float(max(range(len(scores)), key=scores.__getitem__) == item["gold_idx"])

    async def prob_gold(self, item: dict, hint: str) -> float:
        """Softmax over the options, restricted to the options offered.

        Continuous because the binary indicator throws away most of the signal: it
        moves only when a hint flips the argmax, so an effect that shifts belief
        without crossing the boundary reads as exactly zero. On a four-way item the
        indicator's variance is near its maximum, and a plausible transfer effect
        here is a couple of points - smaller than the binary measure can resolve at
        any n this corpus can supply.
        """
        scores = await self.scores(item["question"], item["choices"], hint)
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = sum(weights)
        return weights[item["gold_idx"]] / total if total else float("nan")


def build_pairs_by_similarity(
    items: Sequence[dict],
    units: dict[str, units_mod.Unit],
    dialogues: dict[str, list[dict]],
    embeddings: tuple,
    *,
    threshold: float = 0.82,
    sources_per_target: int = 3,
    seed: int = 0,
) -> list[dict]:
    """Pair items whose PRIMARY units are close in embedding space.

    Primary only, deliberately. Items carry two or three units and matching on any
    of them pairs on a shared prerequisite instead of a shared lesson - it put a
    stamps-and-rates problem with a rounding problem at similarity 1.000, because
    both happened to list place value. The first unit is the thing the item
    actually tests, so it is the only one that makes "teaching A should help on B"
    a fair claim.

    THE THRESHOLD IS A GRANULARITY DIAL, and it trades power against validity:

        0.88   66 targets   tight; near-duplicate skills only
        0.82   81 targets   still clean - simple vs compound interest,
                            rounding to tens vs to hundreds
        0.75  113 targets   starts admitting neighbours rather than the same skill
        0.70  144 targets   too loose to call it the same lesson

    Only the SOURCE needs a dialogue; the target just needs to be an item. That is
    what lets held-out eval items serve as targets, and it is most of where the
    usable n comes from.
    """
    import numpy as np  # noqa: PLC0415

    index, matrix = embeddings
    rng = random.Random(seed)

    usable, primaries = [], []
    for item in items:
        key = units_mod.item_key(item)
        unit = units.get(key)
        if unit and unit.primary and unit.primary in index:
            usable.append(item)
            primaries.append(index[unit.primary])
    if not usable:
        return []

    vectors = matrix[np.asarray(primaries)]
    similarity = (vectors @ vectors.T).astype(float)
    np.fill_diagonal(similarity, -1.0)

    keys = [units_mod.item_key(i) for i in usable]
    source_positions = [i for i, k in enumerate(keys) if dialogues.get(k)]

    pairs: list[dict] = []
    for t, target in enumerate(usable):
        scored = [(similarity[t, s], s) for s in source_positions if s != t and similarity[t, s] >= threshold]
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        for sim, s in scored[:sources_per_target]:
            pairs.append(
                {
                    "unit": units[keys[t]].primary,
                    "source_unit": units[keys[s]].primary,
                    "similarity": round(float(sim), 4),
                    "target": target,
                    "target_key": keys[t],
                    "source_key": keys[s],
                    "source_question": usable[s].get("question", ""),
                }
            )
    rng.shuffle(pairs)
    return pairs


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

    if args.embeddings:
        pairs = build_pairs_by_similarity(
            items,
            units,
            dialogues,
            units_mod.load_embeddings(args.embeddings),
            threshold=args.pair_threshold,
            sources_per_target=args.sources_per_target,
            seed=args.seed,
        )
    else:
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
            "no pairs. Exact unit names rarely repeat - pass --embeddings (written by\n"
            "units.py --embed) so pairing goes by similarity instead of string equality."
        )

    rng = random.Random(args.seed)
    chosen = {p["source_key"]: pick_dialogue(dialogues[p["source_key"]], rng) for p in pairs}
    targets = len({p["target_key"] for p in pairs})
    sims = [p.get("similarity") for p in pairs if p.get("similarity") is not None]
    print(
        f"{len(pairs)} pairs, {targets} distinct targets, {len({p['unit'] for p in pairs})} units"
        + (f", similarity {min(sims):.3f}-{max(sims):.3f}" if sims else "")
    )

    if args.show_pairs:
        for p in pairs[: args.show_pairs]:
            sim = f" (sim {p['similarity']:.3f})" if p.get("similarity") is not None else ""
            print(f"\n  {p.get('source_unit', p['unit'])} -> {p['unit']}{sim}")
            print(f"    taught on: {p['source_question'][:88]}")
            print(f"    tested on: {p['target'].get('question', '')[:88]}")

    if args.dry_run:
        return

    if args.stub_student:
        print("\n*** STUB STUDENT - plumbing check only, these numbers mean nothing ***")
        student = CachingStudent(StubStudent(args.seed))
    else:
        student = CachingStudent(TransferStudent(args.student_model, args.student_url, args.api_key))

    texts = [str(chosen[p["source_key"]].get("completion") or "") for p in pairs]
    swapped = make_swap(units, args.seed)(pairs, texts)

    limiter = asyncio.Semaphore(args.concurrency)

    async def both(pair: dict, text: str) -> tuple[float, float]:
        async with limiter:
            return await student.correct(pair["target"], text), await student.prob_gold(pair["target"], text)

    conditions = await asyncio.gather(
        asyncio.gather(*(both(p, "") for p in pairs)),
        asyncio.gather(*(both(p, t) for p, t in zip(pairs, texts))),
        asyncio.gather(*(both(p, s) for p, s in zip(pairs, swapped))),
    )
    base, treat, swap = conditions

    print("\n=== transfer ===")
    results = {}
    for label, k in (("solved (binary)", 0), ("P(gold) (continuous)", 1)):
        results[label] = report(label, [x[k] for x in base], [x[k] for x in treat], [x[k] for x in swap])

    if args.report:
        await leak_contrast(pairs, dialogues, student, args)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"pairs": len(pairs), "targets": targets, "measures": results}, handle, indent=2)
        print(f"\nwrote {args.out}")


def report(label: str, base: list[float], treated: list[float], swapped: list[float]) -> dict:
    """Means, plus a PAIRED standard error taken from the data.

    The anchor quotes ``0.5/sqrt(n)``, the worst case for an unpaired proportion.
    Here every condition is measured on the same target with the same instrument,
    so the quantity of interest is a per-pair difference and its standard error is
    the spread of those differences - typically well under the unpaired bound,
    because most pairs simply do not move and contribute nothing to the variance.
    Quoting the unpaired figure would hide a real effect behind a bound that does
    not apply.
    """
    n = len(base)
    gains = [t - b for t, b in zip(treated, base)]
    specs = [t - s for t, s in zip(treated, swapped)]

    def se(xs: list[float]) -> float:
        return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")

    gain, spec = statistics.fmean(gains), statistics.fmean(specs)
    gse, sse = se(gains), se(specs)
    print(f"\n  {label}  (n={n})")
    print(
        f"    baseline {statistics.fmean(base):.3f}   treated {statistics.fmean(treated):.3f}"
        f"   swapped {statistics.fmean(swapped):.3f}"
    )
    print(f"    gain        {gain:+.4f} +/- {gse:.4f}  = {gain / gse:+.1f} SE" if gse else "")
    print(f"    specificity {spec:+.4f} +/- {sse:.4f}  = {spec / sse:+.1f} SE" if sse else "")
    verdict_line(spec, sse)
    return {"n": n, "baseline": statistics.fmean(base), "gain": gain, "gain_se": gse, "spec": spec, "spec_se": sse}


def verdict_line(spec: float, se: float) -> None:
    if not se or math.isnan(se):
        return
    if abs(spec) < 2 * se:
        print("    -> no unit-specific transfer at this n")
    elif spec > 0:
        print("    -> unit-specific transfer PRESENT; leakage cannot produce this")
    else:
        print("    -> same-unit dialogue is WORSE than a foreign one; investigate before believing it")


async def leak_contrast(pairs, dialogues, student, args) -> None:
    """The contamination check, paired.

    A dialogue that gave A's answer away should carry NOTHING to B, because A's
    answer is not B's. If leaked sources move B as much as clean ones, the pair
    shares more than a knowledge unit - overlapping answers, or units drawn so
    broadly that the items are near-duplicates - and the headline is measuring
    that rather than teaching.

    Restricted to source items that have BOTH a clean and a leaked dialogue, so
    the two arms differ in leakage and in nothing else. Splitting the headline's
    own sources instead gives two unrelated groups, and on this corpus leaves the
    leaked arm at n=16, far too small to read - which is worse than useless for a
    check whose whole purpose is to invalidate a result.
    """
    subset = []
    for pair in pairs:
        entries = dialogues[pair["source_key"]]
        clean = [e for e in entries if not e.get("leaked")]
        leaked = [e for e in entries if e.get("leaked")]
        if clean and leaked:
            subset.append((pair, clean[0], leaked[0]))

    print("\n=== contamination check: same pair, clean vs leaked source dialogue ===")
    if len(subset) < 20:
        print(f"  only {len(subset)} source items have both kinds; cannot read this")
        return

    limiter = asyncio.Semaphore(args.concurrency)

    async def measure(pair: dict, text: str) -> float:
        async with limiter:
            return await student.prob_gold(pair["target"], text)

    base, with_clean, with_leaked = await asyncio.gather(
        asyncio.gather(*(measure(p, "") for p, _, _ in subset)),
        asyncio.gather(*(measure(p, str(c.get("completion") or "")) for p, c, _ in subset)),
        asyncio.gather(*(measure(p, str(lk.get("completion") or "")) for p, _, lk in subset)),
    )
    n = len(subset)
    clean_gain = [c - b for c, b in zip(with_clean, base)]
    leaked_gain = [lk - b for lk, b in zip(with_leaked, base)]
    se = statistics.stdev([c - lk for c, lk in zip(clean_gain, leaked_gain)]) / math.sqrt(n) if n > 1 else float("nan")
    b, c, lk = statistics.fmean(base), statistics.fmean(with_clean), statistics.fmean(with_leaked)
    print(f"  n={n}, P(gold), paired 1 SE on the difference = {se:.4f}")
    print(f"  baseline           {b:.3f}")
    print(f"  clean source       {c:.3f}   gain {statistics.fmean(clean_gain):+.4f}")
    print(f"  leaked source      {lk:.3f}   gain {statistics.fmean(leaked_gain):+.4f}")
    if statistics.fmean(leaked_gain) >= statistics.fmean(clean_gain) - 2 * se:
        print(
            "\n  -> CONTAMINATED. Leaking about A should be worthless on B, and it\n"
            "     is not. The pairs share more than a knowledge unit; treat the\n"
            "     headline as unproven until the pairing is tightened."
        )
    else:
        print("\n  -> clean. Leaking about A does not carry to B, which is what the pairing assumed.")


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
    parser.add_argument("--embeddings", default=None, help=".npz from units.py --embed; enables similarity pairing")
    parser.add_argument("--pair-threshold", type=float, default=0.82, help="min primary-unit cosine similarity")
    parser.add_argument("--max-per-unit", type=int, default=12)
    parser.add_argument("--sources-per-target", type=int, default=3, help="same-unit dialogues tested per target item")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--show-pairs", type=int, default=5, help="print this many pairs to eyeball the unit tags")
    parser.add_argument("--report", action="store_true", help="also split by whether the source dialogue leaked")
    parser.add_argument("--dry-run", action="store_true", help="build pairs and stop; no model needed")
    parser.add_argument("--stub-student", action="store_true", help="run the full path with a fake answerer")
    parser.add_argument("--out", default=None)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
