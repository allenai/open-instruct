"""Turn the generated dialogues into prompts open-instruct can train on.

    python -m projects.pedagogy_rm.build_rl_data --out data/rl

One row is one moment: a question, and the student's last message. The policy
writes the next tutor turn and the head scores it. Single turn, no simulated
student, because that is exactly the distribution the head was fitted on - a
scorer fitted on single turns and applied to a whole dialogue is reading a number
that means nothing, and the previous project got a reward correlating +0.291 with
leakage that way.

THE PROMPT IS THE SCORING CONTEXT, EXACTLY. `extract_hidden.context_messages`
builds the three messages the head saw, and these rows use the same function
rather than a lookalike. If the policy were prompted one way and judged as though
it had answered another, it would be optimising a different question from the one
being measured, and nothing would report that.

QUESTIONS ARE HELD OUT WHOLE. The eval split shares no question with train, so a
policy cannot score on it by having seen the same problem. `probe.py` grouped its
folds the same way, for the same reason.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from open_instruct.scored_rewards.data import PASSTHROUGH, write_jsonl
from projects.pedagogy_rm.extract_hidden import context_messages


def moments(traces: list[dict], max_per_question: int) -> list[dict]:
    """Every point where a student has just spoken and a tutor is about to.

    Deduplicated on the student's message: the dialogues were generated over a
    temperature and style grid, so the same opening recurs across them, and the
    duplicates would weight those questions more heavily for no reason.
    """
    seen: set[tuple[str, str]] = set()
    per_question: dict[str, int] = {}
    out = []
    for trace in traces:
        transcript = trace["transcript"]
        for i, turn in enumerate(transcript):
            if turn.get("role") != "student" or i + 1 >= len(transcript):
                continue
            if transcript[i + 1].get("role") != "tutor":
                continue
            question, before = trace["question"], turn.get("text", "").strip()
            if not before or (question, before) in seen:
                continue
            if per_question.get(question, 0) >= max_per_question:
                continue
            seen.add((question, before))
            per_question[question] = per_question.get(question, 0) + 1
            choices = trace.get("choices") or []
            gold_idx = trace.get("gold_idx")
            out.append(
                {
                    "question": question,
                    "student_before": before,
                    "choices": choices,
                    "gold": choices[int(gold_idx)] if choices and gold_idx is not None else "",
                    "subject": trace.get("subject", ""),
                    "grade": trace.get("grade", ""),
                    "turn_index": i,
                }
            )
    return out


def rows_for(items: list[dict]) -> list[dict]:
    """RLVR rows whose prompt is byte-identical to the head's scoring context."""
    rows = []
    for item in items:
        rows.append(
            {
                "messages": context_messages(item),
                "ground_truth": json.dumps(item, ensure_ascii=False),
                # No per-sample verifier: the group scorer supplies the whole
                # reward, and passthrough is upstream's no-op returning 0.0.
                "dataset": PASSTHROUGH,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", default="data/traces.jsonl")
    parser.add_argument("--out", default="data/rl", help="directory for train.jsonl and eval.jsonl")
    parser.add_argument("--held-out", type=int, default=50, help="questions reserved for evaluation")
    parser.add_argument("--max-per-question", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.traces) as handle:
        traces = [json.loads(line) for line in handle if line.strip()]
    items = moments(traces, args.max_per_question)

    questions = sorted({i["question"] for i in items})
    random.Random(args.seed).shuffle(questions)
    held = set(questions[: args.held_out])
    train = [i for i in items if i["question"] not in held]
    evaluation = [i for i in items if i["question"] in held]

    os.makedirs(args.out, exist_ok=True)
    write_jsonl(rows_for(train), f"{args.out}/train.jsonl")
    write_jsonl(rows_for(evaluation), f"{args.out}/eval.jsonl")
    print(f"{len(items)} moments from {len(traces)} dialogues over {len(questions)} questions")
    print(f"  train {len(train):>4} rows, {len(questions) - len(held):>3} questions -> {args.out}/train.jsonl")
    print(f"  eval  {len(evaluation):>4} rows, {len(held):>3} questions -> {args.out}/eval.jsonl")


if __name__ == "__main__":
    main()
