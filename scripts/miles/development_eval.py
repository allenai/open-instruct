"""Score saved math/GSM8K development responses in the pinned olmo-eval environment.

Copy this module into custom evaluator bundles. Gold labels may be a scalar or
a flat list of acceptable answers; malformed labels are errors, never zero rewards.
"""

import re

from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.extract import MathExtractor

NUMBER = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def gold_answers(label):
    answers = label if isinstance(label, list) else [label]
    if not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError("Development gold must be a nonempty string or a flat list of nonempty strings")
    return answers


def score_one(row, item):
    gold = gold_answers(item["label"])
    text = row["text"].split("</think>")[-1].strip().removeprefix("<answer>").removesuffix("</answer>").strip()
    if item["verifier"] == "math":
        answers = MathExtractor.extract_answer(text)
        output = LMOutput(text=text, extracted_answer=answers[0] if answers else None)
        output.metadata["all_extracted_answers"] = answers
        score = MinervaMathScorer().process_score(
            Instance(question=item["prompt"], gold_answer=gold[0], metadata={"all_gold_answers": gold}), output
        )
    elif item["verifier"] == "gsm8k":
        matches = NUMBER.findall(re.sub(r"(?<=\d),(?=\d)", "", text))
        answers = matches[-1] if matches else None
        score = float(answers in gold)
    else:
        raise ValueError(f"Unsupported development verifier: {item['verifier']}")
    closed = "</think>" in row["text"] and bool(row["text"].rsplit("</think>", 1)[1].strip())
    finish = row["finish"]["type"]
    if finish not in ("stop", "length"):
        raise ValueError(f"Unsupported generation finish: {finish}")
    return dict(
        row,
        score=score,
        scoring={"extracted": answers},
        closed_think=closed,
        capped=finish == "length",
        completed_final_score=score * float(closed and finish == "stop"),
    )
