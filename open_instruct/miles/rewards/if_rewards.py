"""Score instruction-following responses strictly and screen IF training rows.
The verifier reuses Open Instruct's IFEvalG checkers but pays 1 only when every
constraint holds, and reads a hidden count relation as the exact count the
prompt states. The row screens flag constraint sets that no response can
satisfy, kwargs that make a checker random, and rows that the answer "a" fully
passes, so data preparation can drop them before training.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import json
import random
import re
from typing import Any

from langdetect import detector_factory

from open_instruct import logger_utils
from open_instruct.ground_truth_utils import (
    VerificationResult,
    VerifierConfig,
    VerifierFunction,
    remove_thinking_section,
)
from open_instruct.IFEvalG import instructions_registry

logger = logger_utils.setup_logger(__name__)

# langdetect loads its language profiles lazily and not thread-safely; the default
# async verifier path scores in a thread pool, so load them once at import.
detector_factory.init_factory()

# Constraint types the Phase 1 audit found unsatisfied in every run
# (runs/mopd-phase1-log.md, 2026-10-20): copy-type checks demand the response
# equal a prompt span while always paired with another constraint, and the other
# two need exact sentence/word positions the policy never met.
KNOWN_UNSAT_TYPES = frozenset(
    {
        "new:copy_span_idx",
        "copy:copy",
        "copy:copying_simple",
        "copy:copying_multiple",
        "count:counting_composition",
        "keywords:keyword_specific_position",
    }
)

_HIDDEN_RELATION_ID = "keywords:word_count_different_numbers"
_EXACT = "exactly"
_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
_RELATION_PHRASES = {"less than": ("less than", "fewer than"), "at least": ("at least",)}
_TRIVIAL_ANSWER = "a"


@dataclasses.dataclass
class StrictIFConfig(VerifierConfig):
    all_or_nothing: bool = True
    infer_hidden_relation: bool = True


def constraint_bundle(label: Any) -> dict:
    """The ``{"instruction_id": [...], "kwargs": [...]}`` dict inside an IF target."""
    if isinstance(label, list):
        if not label:
            raise ValueError("IF target list is empty")
        return constraint_bundle(label[0])
    if isinstance(label, str):
        try:
            parsed = ast.literal_eval(label)
        except (ValueError, SyntaxError):
            parsed = json.loads(label)
        return constraint_bundle(parsed)
    if not isinstance(label, dict) or "instruction_id" not in label:
        raise ValueError("IF target must hold instruction_id and kwargs")
    return label


def _relation_stated(query: str, relation: str, frequency: int) -> bool:
    numbers = [str(frequency)]
    if 0 <= frequency < len(_NUMBER_WORDS):
        numbers.append(_NUMBER_WORDS[frequency])
    return any(
        re.search(rf"\b{phrase}\s+{number}\b", query, flags=re.IGNORECASE)
        for phrase in _RELATION_PHRASES.get(relation, ())
        for number in numbers
    )


def resolve_kwargs(instruction_id: str, kwargs: dict | None, query: str | None) -> dict:
    """Checker kwargs without None values; a count relation the prompt never states becomes exact."""
    args = {key: value for key, value in (kwargs or {}).items() if value is not None}
    if (
        instruction_id == _HIDDEN_RELATION_ID
        and query is not None
        and args.get("relation") in _RELATION_PHRASES
        and isinstance(args.get("frequency"), int)
        and not _relation_stated(query, args["relation"], args["frequency"])
    ):
        args["relation"] = _EXACT
    return args


def build_checkers(bundle: dict, query: str | None, *, infer_hidden_relation: bool = True) -> list[tuple[str, Any]]:
    ids, kwargs_list = bundle["instruction_id"], bundle["kwargs"]
    if len(ids) != len(kwargs_list):
        raise ValueError("IF target needs one kwargs entry per instruction")
    checkers = []
    for instruction_id, kwargs in zip(ids, kwargs_list, strict=True):
        args = resolve_kwargs(instruction_id, kwargs, query if infer_hidden_relation else None)
        checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        checker.build_description(**args)
        checkers.append((instruction_id, checker))
    return checkers


def constraint_results(answer: str, checkers: list[tuple[str, Any]]) -> list[bool]:
    return [bool(answer.strip()) and bool(checker.check_following(answer)) for _, checker in checkers]


class StrictIFVerifier(VerifierFunction):
    """IFEval-style verifier that pays only for full compliance by default."""

    def __init__(self, verifier_config: StrictIFConfig | None = None) -> None:
        self.config = verifier_config or StrictIFConfig()
        super().__init__("strict_ifeval", weight=1.0, verifier_config=self.config)

    @classmethod
    def get_config_class(cls) -> type:
        return StrictIFConfig

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        config = self.config
        answer = remove_thinking_section(prediction)
        if not prediction.strip() or not answer:
            return VerificationResult(score=0.0)
        checkers = build_checkers(constraint_bundle(label), query, infer_hidden_relation=config.infer_hidden_relation)
        passed = constraint_results(answer, checkers)
        if not passed:
            return VerificationResult(score=0.0)
        if config.all_or_nothing:
            return VerificationResult(score=float(all(passed)))
        return VerificationResult(score=sum(passed) / len(passed))


# ---------------------------------------------------------------------------
# Row screens for data preparation.


def _args(checker: Any) -> dict:
    return checker.get_instruction_args() or {}


def _nondeterministic(instruction_id: str, kwargs: dict) -> bool:
    """True when building the checker twice from the same kwargs gives different arguments."""
    state = random.getstate()
    try:
        built = []
        for seed in (0, 1):
            random.seed(seed)
            checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
            checker.build_description(**kwargs)
            built.append(_args(checker))
    finally:
        random.setstate(state)
    return built[0] != built[1]


def _interval(relation: str, number: int) -> tuple[float, float]:
    if relation == "less than":
        return (0, number - 1)
    if relation == "at least":
        return (number, float("inf"))
    return (number, number)


def _intersect(intervals: list[tuple[float, float]]) -> tuple[float, float]:
    return (max(low for low, _ in intervals), min(high for _, high in intervals))


def _word_counts(checkers: list[tuple[str, Any]]) -> dict[str, list[tuple[str, tuple[float, float]]]]:
    """Whole-word count bounds per lowercased word, from checkers that count whole words."""
    bounds = collections.defaultdict(list)
    for instruction_id, checker in checkers:
        args = _args(checker)
        if instruction_id == "keywords:word_once":
            bounds[args["keyword"].lower()].append((instruction_id, (1, 1)))
        elif instruction_id == _HIDDEN_RELATION_ID:
            interval = _interval(args["relation"], args["frequency"])
            bounds[args["keyword"].lower()].append((instruction_id, interval))
        elif instruction_id == "count:count_increment_word":
            bounds[args["keyword1"].lower()].append((instruction_id, (1, 1)))
            bounds[args["keyword2"].lower()].append((instruction_id, (2, 2)))
        elif instruction_id == "keywords:exclude_word_harder":
            bounds[args["keyword"].lower()].append((instruction_id, (0, 0)))
        elif instruction_id == "keywords:forbidden_words":
            for word in args["forbidden_words"]:
                bounds[word.lower()].append((instruction_id, (0, 0)))
        elif instruction_id in ("first_word:first_word_answer", "first_word:first_word_sent"):
            bounds[args["first_word"].lower()].append((instruction_id, (1, float("inf"))))
        elif instruction_id in ("last_word:last_word_answer", "last_word:last_word_sent"):
            bounds[args["last_word"].lower()].append((instruction_id, (1, float("inf"))))
        elif instruction_id == "startend:end_checker":
            for word in re.findall(r"\w+", args["end_phrase"].lower()):
                bounds[word].append((instruction_id, (1, float("inf"))))
    return bounds


def _contradictions(checkers: list[tuple[str, Any]]) -> list[str]:
    """Constraint pairs that no single response can satisfy together."""
    problems = []
    by_id = collections.defaultdict(list)
    for instruction_id, checker in checkers:
        by_id[instruction_id].append(_args(checker))
    ids = set(by_id)

    def flag(rule: str) -> None:
        problems.append(f"contradiction:{rule}")

    capital, lower = "change_case:english_capital" in ids, "change_case:english_lowercase" in ids
    if capital and lower:
        flag("english_capital+english_lowercase")
    languages = {args["language"] for args in by_id.get("language:response_language", [])}
    if len(languages) > 1:
        flag("response_language+response_language")
    if (capital or lower) and languages - {"en"}:
        flag("english_case+response_language")
    if lower and any(
        args["capital_relation"] == "at least" and args["capital_frequency"] >= 1
        for args in by_id.get("change_case:capital_word_frequency", [])
    ):
        flag("english_lowercase+capital_word_frequency")

    # Literals matched case-sensitively must fit the required case.
    literals = [args["section_spliter"] for args in by_id.get("detectable_format:multiple_sections", [])]
    literals += [args["starter"] for args in by_id.get("detectable_format:constrained_start", [])]
    literals += [args["keyword"] for args in by_id.get("keywords:keyword_specific_position", [])]
    if "detectable_format:constrained_response" in ids:
        literals.append("My answer is")
    if lower and any(any(c.isupper() for c in literal) for literal in literals):
        flag("english_lowercase+case_sensitive_literal")
    if capital and any(any(c.islower() for c in literal) for literal in literals):
        flag("english_capital+case_sensitive_literal")

    # Counts of words, sentences and paragraphs.
    counts = collections.defaultdict(list)
    for args in by_id.get("length_constraints:number_words", []):
        counts["words"].append(_interval(args["relation"], args["num_words"]))
    for args in by_id.get("length_constraints:number_sentences", []):
        counts["sentences"].append(_interval(args["relation"], args["num_sentences"]))
    for args in by_id.get("length_constraints:number_paragraphs", []):
        counts["star_paragraphs"].append((args["num_paragraphs"], args["num_paragraphs"]))
    if "paragraphs:paragraphs" in ids:
        counts["star_paragraphs"].append((2, 2))
    if "paragraphs:paragraphs2" in ids:
        counts["blank_paragraphs"].append((2, 2))
    for args in by_id.get("length_constraints:nth_paragraph_first_word", []):
        counts["blank_paragraphs"].append((args["num_paragraphs"], args["num_paragraphs"]))
    ranges = {name: _intersect(intervals) for name, intervals in counts.items()}
    for name, (low, high) in ranges.items():
        if low > high:
            flag(f"{name}_count")
    if "words" in ranges:
        most_words = ranges["words"][1]
        for name in ("sentences", "star_paragraphs", "blank_paragraphs"):
            if name in ranges and ranges[name][0] > most_words:
                flag(f"words+{name}")

    # Whole-word counts for the same word.
    for word, word_bounds in _word_counts(checkers).items():
        low, high = _intersect([interval for _, interval in word_bounds])
        if low > high:
            flag("+".join(sorted({instruction_id for instruction_id, _ in word_bounds})) + f":{word}")

    # Required first and last words.
    first_words = {args["first_word"].lower() for args in by_id.get("first_word:first_word_answer", [])}
    if len(first_words | {args["first_word"].lower() for args in by_id.get("first_word:first_word_sent", [])}) > 1:
        flag("first_word")
    last_words = {args["last_word"].lower() for args in by_id.get("last_word:last_word_answer", [])}
    for args in by_id.get("startend:end_checker", []):
        # LastWordAnswerChecker drops punctuation inside the final token, so do the same.
        phrase_tokens = args["end_phrase"].lower().split()
        if phrase_tokens:
            last_words.add(re.sub(r"[^\w\s]", "", phrase_tokens[-1]))
    if len(last_words | {args["last_word"].lower() for args in by_id.get("last_word:last_word_sent", [])}) > 1:
        flag("last_word")
    if "keywords:start_end" in ids and first_words and last_words and first_words != last_words:
        flag("start_end+first_last_word")

    # Punctuation bans against required literals.
    end_phrases = [args["end_phrase"] for args in by_id.get("startend:end_checker", [])]
    if "punctuation:punctuation_dot" in ids and (
        "detectable_content:postscript" in ids or any("." in phrase for phrase in end_phrases)
    ):
        flag("punctuation_dot+required_dot")
    if "punctuation:no_comma" in ids and any("," in phrase for phrase in end_phrases):
        flag("no_comma+end_checker")
    if "punctuation:punctuation_exclamation" in ids and any("!" in phrase for phrase in end_phrases):
        flag("punctuation_exclamation+end_checker")

    # A response wrapped in quotes starts and ends with a quote token.
    if "startend:quotation" in ids:
        for other in ("keywords:start_end", "first_word:first_word_answer"):
            if other in ids:
                flag(f"quotation+{other.split(':')[1]}")

    # Copy-type checks fix the whole response; the other constraints must accept it.
    copy_candidates = []
    for args in by_id.get("copy:copy", []) + by_id.get("copy:copying_simple", []):
        copy_candidates.append(args["prompt_to_repeat"])
    for args in by_id.get("copy:copying_multiple", []):
        copy_candidates.append("******".join([args["prompt_to_repeat"]] * args["N"]))
    for args in by_id.get("new:copy_span_idx", []):
        copy_candidates.append(args["prompt_to_repeat"][args["n_start"] : args["n_end"]])
    if copy_candidates and not any(
        all(checker.check_following(text) for _, checker in checkers)
        for candidate in copy_candidates
        for text in (candidate.strip(), candidate.strip().lower(), candidate.strip().upper())
    ):
        flag("copy+other")
    return problems


def if_row_problems(
    row: dict, *, known_unsat: frozenset[str] = KNOWN_UNSAT_TYPES, infer_hidden_relation: bool = True
) -> list[str]:
    """Reasons to drop one IF row; empty when the row is usable.

    ``row`` holds the source ``prompt`` and its IF ``ground_truth`` (a target
    string, or a list whose first element is one).
    """
    try:
        bundle = constraint_bundle(row["ground_truth"])
    except (ValueError, SyntaxError, json.JSONDecodeError) as error:
        return [f"malformed_target:{error}"]
    prompt = row.get("prompt")
    query = prompt if isinstance(prompt, str) else None
    problems = []
    for instruction_id, kwargs in zip(bundle["instruction_id"], bundle["kwargs"], strict=True):
        if instruction_id not in instructions_registry.INSTRUCTION_DICT:
            problems.append(f"unknown_instruction:{instruction_id}")
            continue
        if instruction_id in known_unsat:
            problems.append(f"known_unsat_type:{instruction_id}")
        args = resolve_kwargs(instruction_id, kwargs, query if infer_hidden_relation else None)
        try:
            if _nondeterministic(instruction_id, args):
                problems.append(f"nondeterministic_kwargs:{instruction_id}")
        except Exception as error:
            problems.append(f"build_error:{instruction_id}:{type(error).__name__}")
            continue
        if instruction_id == "length_constraints:nth_paragraph_first_word" and not (
            1 <= args.get("nth_paragraph", 0) <= args.get("num_paragraphs", 0)
        ):
            problems.append(f"invalid_kwargs:{instruction_id}")
        if any(
            isinstance(args.get(key), int) and args[key] <= 0 and args.get(relation) == "less than"
            for key, relation in (
                ("num_words", "relation"),
                ("num_sentences", "relation"),
                ("frequency", "relation"),
                ("capital_frequency", "capital_relation"),
            )
        ):
            problems.append(f"invalid_kwargs:{instruction_id}")
    if any(problem.startswith(("unknown_instruction:", "build_error:")) for problem in problems):
        return problems
    checkers = build_checkers(bundle, query, infer_hidden_relation=infer_hidden_relation)
    problems += _contradictions(checkers)
    if all(constraint_results(_TRIVIAL_ANSWER, checkers)):
        problems.append("trivial_answer_passes")
    return problems


def normalize_if_target(prompt: str, ground_truth: str) -> str:
    """Rewrite a hidden count relation as ``exactly`` so the stored target matches the prompt.

    The result keeps Open Instruct's target encoding: a Python-literal list whose first
    element is the constraint dict or its JSON string.
    """
    bundle = constraint_bundle(ground_truth)
    kwargs = []
    for instruction_id, args in zip(bundle["instruction_id"], bundle["kwargs"], strict=True):
        if (
            instruction_id == _HIDDEN_RELATION_ID
            and args is not None
            and resolve_kwargs(instruction_id, args, prompt).get("relation") == _EXACT
        ):
            args = {**args, "relation": _EXACT}
        kwargs.append(args)
    if kwargs == bundle["kwargs"]:
        return ground_truth
    rewritten = {**bundle, "kwargs": kwargs}
    inner = ast.literal_eval(ground_truth)[0]
    return str([json.dumps(rewritten) if isinstance(inner, str) else rewritten])
