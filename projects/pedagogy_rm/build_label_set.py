"""Turn dialogues into labelling slices, with deliberate overlap.

    python -m projects.pedagogy_rm.build_label_set \
        --traces runs/traces.jsonl --out-dir data/label_slices \
        --n 600 --raters 4 --overlap 0.15

WHY OVERLAP IS NOT OPTIONAL. The previous round split 928 items across six
raters with ZERO shared items, so per-dimension agreement could never be
computed - and agreement turned out to be the thing that mattered, since a probe
cannot beat the noise in its own labels. A fixed fraction of every slice is
therefore shared by all raters. At 15% of 600 that is 90 items rated by everyone,
which is ample for a weighted kappa per dimension and costs each rater a few
extra minutes.

WHAT A RATER SEES. One unit is the question, the student's immediately preceding
turn, and one tutor turn. Nothing after it, and no other tutor turns from the
same dialogue - later context would let a rater score the turn by how the
conversation turned out, which is the student-outcome signal this project is
deliberately not using.

SAMPLING. At most one turn per dialogue by default. Turns from the same dialogue
share a question, a student and a temperature, so they are not independent
samples of tutor behaviour, and a rater who has seen one anchors on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

from projects.pedagogy_rm.rubric import rubric_markdown


def units(trace: dict, max_per_dialogue: int, rng: random.Random) -> list[dict]:
    """Labelling units from one dialogue: (question, prior student turn, tutor turn)."""
    transcript = trace.get("transcript") or []
    found = []
    for i, entry in enumerate(transcript):
        if entry.get("role") != "tutor":
            continue
        prior = next((t["text"] for t in reversed(transcript[:i]) if t.get("role") == "student"), "")
        tutor_text = (entry.get("text") or "").strip()
        if not tutor_text or not prior:
            continue
        # A turn cut off at the token cap is unlabelable: rating it for
        # ``concise`` rates the cap, and ``actionable`` is undefined when the
        # instruction is the part that got cut.
        if entry.get("truncated"):
            continue
        # And a turn whose CONTEXT was corrupted is just as unusable. When a
        # tutor turn is truncated the student continues its sentence rather than
        # replying, so the student turn after it is not a student turn at all -
        # and it is exactly what a rater reads to judge ``targeted``.
        if any(t.get("truncated") for t in transcript[:i] if t.get("role") == "tutor"):
            continue
        found.append(
            {
                "id": unit_id(trace, i),
                "question": trace["question"],
                "choices": trace.get("choices"),
                "gold": (trace.get("choices") or [None] * 99)[trace["gold_idx"]]
                if trace.get("gold_idx") is not None and trace.get("choices")
                else None,
                "student_before": prior,
                "tutor_turn": tutor_text,
                "turn_index": i,
                "subject": trace.get("subject"),
                "grade": trace.get("grade"),
                "temperature": trace.get("temperature"),
                "style": trace.get("style"),
            }
        )
    if max_per_dialogue and len(found) > max_per_dialogue:
        found = rng.sample(found, max_per_dialogue)
    return found


def unit_id(trace: dict, index: int) -> str:
    """Stable across regeneration, so labels survive rebuilding the slices.

    Keyed on the text rather than on position: a rebuild that reorders or
    resamples dialogues would otherwise silently reassign every label.
    """
    payload = f"{trace['question']}|{index}|{trace['transcript'][index]['text']}"
    return "u" + hashlib.sha1(payload.encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=600, help="total units to label")
    parser.add_argument("--raters", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.15, help="fraction every rater labels")
    parser.add_argument("--max-per-dialogue", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.traces) as handle:
        traces = [json.loads(line) for line in handle if line.strip()]
    pool = [u for trace in traces for u in units(trace, args.max_per_dialogue, rng)]
    rng.shuffle(pool)
    pool = pool[: args.n]
    if not pool:
        raise SystemExit("no labelling units - check --traces")

    n_shared = int(len(pool) * args.overlap)
    shared, rest = pool[:n_shared], pool[n_shared:]
    slices: list[list[dict]] = [list(shared) for _ in range(args.raters)]
    for i, unit in enumerate(rest):
        slices[i % args.raters].append(unit)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "RUBRIC.md"), "w") as handle:
        handle.write(rubric_markdown() + "\n")
    for i, chunk in enumerate(slices, start=1):
        rng.shuffle(chunk)  # so the shared items are not all at the top
        with open(os.path.join(args.out_dir, f"slice_{i}.json"), "w") as handle:
            json.dump({"schema": "pedagogy-rm/v1", "shared": n_shared, "units": chunk}, handle, indent=1)

    print(f"{len(traces)} dialogues -> {len(pool)} units")
    print(f"{args.raters} slices of {len(slices[0])}, of which {n_shared} shared by all")
    print(f"wrote {args.out_dir}/slice_1..{args.raters}.json and RUBRIC.md")


if __name__ == "__main__":
    main()
