"""Look at generated dialogues before spending anyone's afternoon labelling them.

    python -m projects.pedagogy_rm.inspect_traces --traces runs/traces.jsonl --samples 3

Generation is the cheap step and labelling is the expensive one, so every defect
caught here is worth several caught later. The first 600 dialogues had three that
only showed up on reading them: turns truncated mid-word, turns beginning
mid-sentence from a chat-template problem, and a length distribution so narrow
that two rubric dimensions would have been constants.

WHAT TO LOOK FOR, in the order it matters:

    spread      A dimension the data does not vary along cannot be labelled
                usefully or learned. If every style has the same median length,
                ``concise`` is dead on arrival.
    defects     Truncated or mid-sentence turns are unlabelable, not merely
                imperfect.
    leak rate   Should be neither 0% nor most of them. If no turn gives the
                answer away, ``leak`` has no positive class.
    the text    Read some. Statistics did not catch the student lecturing
                instead of being stuck; reading three dialogues did.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics


def percentile(values: list[int], q: float) -> int:
    return sorted(values)[min(int(len(values) * q), len(values) - 1)] if values else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--samples", type=int, default=2, help="dialogues to print in full")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.traces) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    tutor = [(r.get("style", "?"), t) for r in rows for t in r["transcript"] if t["role"] == "tutor"]
    usable = [(s, t) for s, t in tutor if not t.get("truncated")]
    student = [t["text"] for r in rows for t in r["transcript"] if t["role"] == "student"]
    print(f"{len(rows)} dialogues, {len(tutor)} tutor turns, {len(usable)} usable after truncation")
    print(f"student turns: median {statistics.median(len(s.split()) for s in student):.0f} words\n")

    print(f"  {'style':<10} {'n':>5} {'median':>7} {'p10':>5} {'p90':>5} {'question':>9} {'trunc':>6}")
    print("  " + "-" * 52)
    for style in sorted({s for s, _ in tutor}):
        words = [len(t["text"].split()) for s, t in usable if s == style]
        if not words:
            continue
        asks = sum(1 for s, t in usable if s == style and t["text"].rstrip().endswith("?")) / len(words)
        cut = sum(1 for s, t in tutor if s == style and t.get("truncated"))
        print(
            f"  {style:<10} {len(words):>5} {statistics.median(words):>7.0f} "
            f"{percentile(words, 0.1):>5} {percentile(words, 0.9):>5} {asks:>8.0%} {cut:>6}"
        )

    leaks = checked = 0
    for row in rows:
        choices, gold_idx = row.get("choices"), row.get("gold_idx")
        if not choices or gold_idx is None:
            continue
        gold = str(choices[gold_idx]).strip().lower()
        if len(gold) < 4:
            continue
        for turn in row["transcript"]:
            if turn["role"] == "tutor" and not turn.get("truncated"):
                checked += 1
                leaks += gold.lower() in turn["text"].lower()
    midsentence = sum(1 for _, t in usable if t["text"][:1].islower())
    print(
        f"\n  gold stated verbatim   {leaks}/{checked} ({leaks / max(checked, 1):.0%})  — wants to be neither 0% nor most"
    )
    print(f"  starts mid-sentence    {midsentence}/{len(usable)}  — above a few percent means a chat-template problem")
    print(f"  empty                  {sum(1 for _, t in usable if not t['text'].strip())}/{len(usable)}")

    rng = random.Random(args.seed)
    for row in rng.sample(rows, min(args.samples, len(rows))):
        print("\n" + "=" * 78)
        gold = (row.get("choices") or ["?"])[row["gold_idx"]] if row.get("gold_idx") is not None else "?"
        print(f"[{row.get('style')} @ {row.get('temperature')}]  gold: {gold}")
        print("Q: " + " ".join(row["question"].split())[:200])
        for turn in row["transcript"][:4]:
            mark = " (TRUNCATED)" if turn.get("truncated") else ""
            print(f"\n{turn['role'].upper()}{mark}: {' '.join(turn['text'].split())[:400]}")


if __name__ == "__main__":
    main()
