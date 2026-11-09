"""Versioned, local long-context answer contracts (no model judges or tools).

Only explicit final-answer regions are interpreted. Metrics describe correctness
of the supplied references, not the factual validity of a synthesized question.
"""

import math
import re
from decimal import Decimal, InvalidOperation

from open_instruct.miles.rewards import deterministic_verifiers as common
from open_instruct.miles.rewards import task_verifiers

KIND = "long_context_v1"
SCORER_REVISION = "long_context_v1.2"
METRICS = {"exact", "docqa", "decimal", "choice", "lines_f1", "ndcg", "line_accuracy"}


def normalize(value):
    return " ".join(value.lower().split())


def validate_target(target):
    if not isinstance(target, dict) or set(target) != {"version", "metric", "answers", "extraction"}:
        raise ValueError("Expected a complete long-context target")
    if target["version"] != 1 or target["metric"] not in METRICS:
        raise ValueError("Unknown long-context contract")
    answers = target["answers"]
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError("Require nonempty string references")
    if target["extraction"] not in {"final", "answer_marker", "therefore", "choice"}:
        raise ValueError("Unknown answer extraction")
    metric = target["metric"]
    if metric in {"decimal", "choice", "line_accuracy", "exact", "docqa"} and len(answers) != 1:
        raise ValueError("This contract requires one reference")
    if metric == "choice" and re.fullmatch(r"[A-Z]", answers[0]) is None:
        raise ValueError("Choice reference must be a single letter")
    if metric == "decimal" and (
        len(answers[0]) > 128 or re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?", answers[0]) is None
    ):
        raise ValueError("Require a bounded finite numeric reference")
    if metric == "ndcg" and len(set(map(normalize, answers))) != len(answers):
        raise ValueError("Ranking reference contains duplicate items")
    return metric


def extract(text, mode):
    final = common.answer_text(text)
    # Unclosed explicit thinking is not a final answer. Olmo's opener can also
    # live in the prompt; core.reward_final_answer_only handles that case.
    if "<think>" in final:
        return ""
    if mode == "answer_marker":
        markers = [(final.rfind(m), m) for m in ("[Answer]", "[答案]")]
        pos, marker = max(markers)
        return final[pos + len(marker) :].strip() if pos >= 0 else final
    if mode == "therefore":
        marker = "Therefore, the answer is "
        if marker in final:
            final = final.rsplit(marker, 1)[1].strip()
        return final.removesuffix(".")
    if mode == "choice":
        # A letter inside prose (e.g. "I think") is not an asserted choice.
        # Parentheses delimit an explicit choice even before option text. A bare
        # letter instead needs punctuation, a line ending or the end of the answer.
        token = r"(?:\(([A-Z])\)|([A-Z])\)?)"
        matches = list(
            re.finditer(
                r"\bThe correct answer is\s*:?\s*"
                r"(?:\(([A-Z])\)(?=[\s.,;:!?]|$)|([A-Z])"
                r"(?:\)(?=[\s.,;:!?]|$)|(?=[ \t]*[.,;:!?\n]|[ \t]*$)))",
                final,
            )
        )
        if matches:
            # Option text may follow a choice, but another bracketed choice on
            # that answer line is ambiguous (e.g. "(C) and (D)"). Do not scan
            # subsequent explanation lines for alternatives.
            if any(re.match(r"[^\r\n]*\([A-Z]\)", final[match.end() :]) for match in matches):
                return ""
            choices = [match[1] or match[2] for match in matches]
            return choices[0] if len(set(choices)) == 1 else ""
        match = re.fullmatch(r":?\s*" + token + r"[.,]?", final)
        return (match[1] or match[2]) if match else ""
    return final


def score(prediction, target):
    metric = validate_target(target)
    final = extract(prediction, target["extraction"])
    gold = target["answers"]
    if not final:
        return 0.0
    if metric == "exact":
        return float(final.strip() == gold[0].strip())
    if metric == "docqa":
        # Deliberately whole-answer equality, never substring credit for a list
        # of guesses or an incorrect name/number containing the correct one.
        return float(normalize(final).rstrip(".") == normalize(gold[0]).rstrip("."))
    if metric == "choice":
        return float(final == gold[0])
    if metric == "decimal":
        if len(final) > 128 or re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?", final) is None:
            return 0.0
        try:
            actual, expected = Decimal(final), Decimal(gold[0])
            return float(abs(actual - expected) <= max(Decimal("0.000001"), abs(expected) * Decimal("0.0001")))
        except InvalidOperation:
            return 0.0
    if metric == "line_accuracy":
        expected, actual = gold[0].strip().splitlines(), final.strip().splitlines()
        # Positional comparison penalizes missing, extra and reordered lines.
        # Unlike set overlap, repeated source lines are separate obligations.
        return sum(a == b for a, b in zip(expected, actual)) / max(len(expected), len(actual))
    actual = [normalize(line) for line in final.splitlines() if line.strip()]
    expected = list(map(normalize, gold))
    if metric == "lines_f1":
        a, b = set(actual), set(expected)
        return 2 * len(a & b) / (len(a) + len(b))
    # Linear relevance and log2 discount. Deduplicate in first-occurrence order:
    # appending an earlier bad guess cannot move it out of the scored prefix.
    # This deliberately fixes the upstream wrapper's last-occurrence behavior;
    # trec_eval itself rejects duplicate document IDs rather than choosing one.
    relevance = {value: len(expected) - i for i, value in enumerate(expected)}
    ranked = list(dict.fromkeys(actual))[: len(expected)]
    dcg = sum(relevance.get(value, 0) / math.log2(i + 2) for i, value in enumerate(ranked))
    ideal = sum((len(expected) - i) / math.log2(i + 2) for i in range(len(expected)))
    return dcg / ideal


class LongContextVerifier:
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        pass

    async def async_call(self, tokens, prediction, label, **kwargs):
        value = score(prediction, label)
        return common.Result(
            value,
            diagnostics={
                "kind": KIND,
                "scorer_revision": SCORER_REVISION,
                "status": "ok",
                "metric": label["metric"],
                "full_reference_match": value == 1.0,
            },
        )
