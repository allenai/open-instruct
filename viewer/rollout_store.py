from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, BinaryIO

from viewer._compat import (
    BROWSECOMP_CORRECTNESS_JUDGE_PROMPT,
    apply_browsecomp_format_gates,
    apply_re_search_format_gates,
    format_reference_answer,
    normalize_answer,
)

STEP_PREFIX = re.compile(rb'^\{"step":\s*(\d+)')
ANSWER_HEADING = re.compile(
    r"(?im)(?:^|\n)\s*(?:#{1,4}\s*)?(?:\*\*)?(?P<label>(?:final\s+)?answer)\s*[:\-]\s*(?:\*\*)?\s*(?P<value>[^\n]*)"
)
XML_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
IM_START = re.compile(r"<\|im_start\|>(?P<role>[A-Za-z0-9_-]*)[ \t]*\n?")
IM_END = re.compile(r"<\|im_end\|>")
USER_CHAT_TURN = re.compile(r"<\|im_start\|>user[ \t]*\n(?P<content>.*?)<\|im_end\|>", re.DOTALL)
REASONING_OR_CALL = re.compile(
    r"<think>(?P<reasoning>.*?)</think>|<tool_call>(?P<call>.*?)</tool_call>", re.IGNORECASE | re.DOTALL
)
TOOL_RESPONSE = re.compile(r"<tool_response>(?P<result>.*?)</tool_response>", re.IGNORECASE | re.DOTALL)
FUNCTION_NAME = re.compile(r"<function=(?P<name>[^>\s]+)")
HARD_CORRUPTION_PATTERNS = (
    ("replacement characters", re.compile(r"�")),
    ("mojibake", re.compile(r"(?:ï¿½|â(?:€|€™|€œ|€œ|€“|€”|€¦)|Ã[\x80-\xBF])")),
    ("raw byte token", re.compile(r"(?:<0x[0-9A-Fa-f]{2}>\s*){2,}")),
)
DEGENERATE_REPETITION_PATTERNS = (
    ("repeated word", re.compile(r"\b([A-Za-z]{2,20})(?:\s+\1){6,}\b", re.IGNORECASE)),
    ("repeated character", re.compile(r"([A-Za-z0-9])\1{11,}")),
    ("repeated punctuation", re.compile(r"([!?.,;:/\\#]{2,8})\1{7,}")),
)
TOKENIZER_ARTIFACT = re.compile(r"(?:Ġ|Ċ|▁)")
RAW_CHAT_SENTINEL = re.compile(r"<\|[^|>]+\|>")
ALPHABETIC_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
FILE_NAME = re.compile(r"^(?P<run>.+?)_(?P<filtered>filtered_)?rollouts_(?P<shard>\d+)\.jsonl$")
# Terminations that make the judge return zero without ever scoring the answer.
INCOMPLETE_TERMINATIONS = frozenset(
    {"max_steps", "response_limit", "context_limit", "reset_failure", "generation_failure"}
)
# A mixed group this lopsided barely escaped the zero-variance filter.
NEAR_MISS_MARGIN = 3
HARD_GROUP_MAX_PASS_RATE = 0.25
EASY_GROUP_MIN_PASS_RATE = 0.75
TRAJECTORY_MATCH_LIMIT = 1_000
TRAJECTORY_MATCH_CATEGORIES = ("reasoning", "tool_call", "tool_result", "final_output")

GROUP_CATEGORIES: list[dict[str, str]] = [
    {"id": "all_groups", "label": "All groups", "description": "Every prompt group in this artifact step"},
    {"id": "all_wrong_group", "label": "All wrong", "description": "Pass rate is exactly 0%"},
    {"id": "hard_group", "label": "Hard", "description": "Pass rate is above 0% and at most 25%"},
    {"id": "learning_group", "label": "Learning zone", "description": "Pass rate is above 25% and below 75%"},
    {"id": "easy_group", "label": "Easy", "description": "Pass rate is at least 75% but below 100%"},
    {"id": "all_correct_group", "label": "All correct", "description": "Pass rate is exactly 100%"},
]

# Review queues, in sidebar order. "section" only drives grouping in the UI.
CATEGORIES: list[dict[str, str]] = [
    # What happened to this one trajectory.
    {
        "id": "judged_correct",
        "label": "Judged correct",
        "section": "Outcome",
        "description": "The judge scored the answer and gave reward",
    },
    {
        "id": "judged_incorrect",
        "label": "Judged incorrect",
        "section": "Outcome",
        "description": "The judge read a real answer and rejected it",
    },
    {
        "id": "incomplete",
        "label": "Incomplete (never judged)",
        "section": "Outcome",
        "description": "Zero reward without the judge ever scoring an answer",
    },
    {
        "id": "incomplete_response_limit",
        "label": "· ran out of response budget",
        "section": "Outcome",
        "description": "Exhausted response_length inside the tool loop",
    },
    {
        "id": "incomplete_max_steps",
        "label": "· hit the tool-call cap",
        "section": "Outcome",
        "description": "Reached max_steps tool calls",
    },
    {
        "id": "incomplete_context_limit",
        "label": "· ran out of context",
        "section": "Outcome",
        "description": "Reached the context window limit",
    },
    {
        "id": "incomplete_generation_failure",
        "label": "· generation request failed",
        "section": "Outcome",
        "description": "A request-local model-serving failure was masked and scored zero",
    },
    {
        "id": "incomplete_no_terminal_message",
        "label": "· no terminal message",
        "section": "Outcome",
        "description": "Produced no final assistant message at all",
    },
    {
        "id": "incomplete_unclean_stop",
        "label": "· did not stop cleanly",
        "section": "Outcome",
        "description": "Final generation ended for a reason other than stop",
    },
    # Shape of the 32-sample prompt group this trajectory belongs to.
    {
        "id": "all_wrong_group",
        "label": "All-wrong groups",
        "section": "Group shape",
        "description": "Every sample of the prompt failed, so the group taught nothing",
    },
    {
        "id": "all_correct_group",
        "label": "All-correct groups",
        "section": "Group shape",
        "description": "Every sample of the prompt succeeded, so the group taught nothing",
    },
    {
        "id": "mixed_group",
        "label": "Mixed groups",
        "section": "Group shape",
        "description": "The prompt had both successes and failures, so it carried gradient",
    },
    {
        "id": "near_miss_group",
        "label": "Near-miss groups",
        "section": "Group shape",
        "description": f"Mixed groups where at most {NEAR_MISS_MARGIN} samples differ from the rest",
    },
    # Structural anomalies worth a human read.
    {
        "id": "review",
        "label": "All suspicious",
        "section": "Review flags",
        "description": "Any structural anomaly; excludes ordinary zero reward, length, and tool errors",
    },
    {
        "id": "format_error",
        "label": "Format Error",
        "section": "Review flags",
        "description": "Terminal turn failed the format rules used by this run's verifier",
    },
    {
        "id": "no_tool_calls",
        "label": "No Tool Calls",
        "section": "Review flags",
        "description": "The trajectory made zero search, visit, or bash calls",
    },
    {
        "id": "judge_negative_has_answer",
        "label": "Judge: negative has answer",
        "section": "Review flags",
        "description": "A format-valid judged response received zero despite containing an exact reference answer",
    },
    {
        "id": "judge_positive_no_answer",
        "label": "Judge: positive has no answer",
        "section": "Review flags",
        "description": "A format-valid judged response was rewarded without containing an exact reference answer",
    },
    {
        "id": "gibberish",
        "label": "Gibberish",
        "section": "Review flags",
        "description": "Any hard corruption, localized corruption, or token-salad signal",
    },
    {
        "id": "hard_corruption",
        "label": "· hard corruption",
        "section": "Review flags",
        "description": "Broken Unicode, tokenizer artifacts, raw sentinels, or control characters",
    },
    {
        "id": "localized_corruption",
        "label": "· localized corruption",
        "section": "Review flags",
        "description": "A lexical token unexpectedly mixes multiple writing systems",
    },
    {
        "id": "token_salad",
        "label": "· token salad",
        "section": "Review flags",
        "description": "Multiple mixed-script tokens, scattered scripts, or degenerate repetition",
    },
    {
        "id": "no_final_answer",
        "label": "No final answer",
        "section": "Review flags",
        "description": "No terminal model turn was captured",
    },
    {"id": "timeouts", "label": "Tool timeouts", "section": "Review flags", "description": "A tool call timed out"},
    # Common properties, kept as filters but never enough to flag a rollout.
    {
        "id": "token_capped",
        "label": "Token capped",
        "section": "Context",
        "description": "Length or non-stop termination",
    },
    {"id": "long", "label": "Long (64k+)", "section": "Context", "description": "At least 65,536 stored tokens"},
    {"id": "tool_errors", "label": "Tool errors", "section": "Context", "description": "Search or visit errors"},
    {
        "id": "filtered",
        "label": "Discarded",
        "section": "Context",
        "description": "Discarded before training by active sampling",
    },
    {
        "id": "healthy_looking",
        "label": "Healthy-looking",
        "section": "Context",
        "description": "No structural warning detected",
    },
]


class RolloutStoreError(RuntimeError):
    pass


@dataclass
class FileInfo:
    id: str
    path: Path
    run: str
    source: str
    shard: int
    size: int
    mtime: float
    # Reading a shard's step range means scanning back over a multi-megabyte record,
    # so boundaries stay unresolved until some caller asks about that run.
    first_step: int | None = None
    last_step: int | None = None

    @property
    def resolved(self) -> bool:
        return self.first_step is not None and self.last_step is not None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.path.name,
            "source": self.source,
            "shard": self.shard,
            "size": self.size,
            "mtime": self.mtime,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class RecordPointer:
    id: str
    file_id: str
    offset: int
    length: int


@dataclass(frozen=True)
class GibberishAssessment:
    hard_corruption: tuple[str, ...] = ()
    localized_corruption: tuple[str, ...] = ()
    token_salad: tuple[str, ...] = ()

    @property
    def reasons(self) -> list[str]:
        return list(self.hard_corruption + self.localized_corruption + self.token_salad)

    @property
    def tiers(self) -> list[str]:
        return [
            tier
            for tier, reasons in (
                ("hard_corruption", self.hard_corruption),
                ("localized_corruption", self.localized_corruption),
                ("token_salad", self.token_salad),
            )
            if reasons
        ]


@dataclass
class StepData:
    run: str
    source: str
    step: int
    records: list[dict[str, Any]]
    category_counts: Counter[str]
    total_records: int


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def compact(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}…"


def tail_preview(value: str, limit: int = 520) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return f"…{value[-(limit - 1) :]}"


def question_from_prompt(value: str) -> str:
    """Recover the query string that the reward function received."""
    value = value.strip()
    matches = list(USER_CHAT_TURN.finditer(value))
    if matches:
        return matches[-1].group("content").strip()
    if value.startswith("user: "):
        return value[len("user: ") :]
    return value


def _value_start(line: bytes, key: str) -> int | None:
    marker = json.dumps(key).encode() + b":"
    position = line.find(marker)
    if position < 0:
        return None
    position += len(marker)
    while position < len(line) and line[position] in b" \t\r\n":
        position += 1
    return position


def _value_end(line: bytes, start: int) -> int:
    if start >= len(line):
        raise ValueError("Missing JSON value")
    first = line[start]
    if first == ord('"'):
        escaped = False
        for position in range(start + 1, len(line)):
            current = line[position]
            if escaped:
                escaped = False
            elif current == ord("\\"):
                escaped = True
            elif current == ord('"'):
                return position + 1
        raise ValueError("Unterminated JSON string")
    if first in (ord("["), ord("{")):
        opening = first
        closing = ord("]") if opening == ord("[") else ord("}")
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(line)):
            current = line[position]
            if in_string:
                if escaped:
                    escaped = False
                elif current == ord("\\"):
                    escaped = True
                elif current == ord('"'):
                    in_string = False
                continue
            if current == ord('"'):
                in_string = True
            elif current == opening:
                depth += 1
            elif current == closing:
                depth -= 1
                if depth == 0:
                    return position + 1
        raise ValueError("Unterminated JSON collection")
    position = start
    while position < len(line) and line[position] not in b",}\r\n":
        position += 1
    return position


def extract_json_value(line: bytes, key: str, default: Any = None) -> Any:
    start = _value_start(line, key)
    if start is None:
        return default
    try:
        end = _value_end(line, start)
        return json.loads(line[start:end])
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return default


def json_array_length(line: bytes, key: str) -> int:
    start = _value_start(line, key)
    if start is None or start >= len(line) or line[start] != ord("["):
        return 0
    end = line.find(b"]", start + 1)
    if end < 0:
        return 0
    contents = line[start + 1 : end]
    if not contents.strip():
        return 0
    return contents.count(b",") + 1


def terminal_prose(value: str) -> str:
    """Return the complete saved terminal turn without hiding malformed structure."""
    return value.strip()


def answer_declarations(value: str) -> list[str]:
    declarations = [match.group("value").strip() for match in ANSWER_HEADING.finditer(value)]
    declarations.extend(match.strip() for match in XML_ANSWER.findall(value))
    return [declaration for declaration in declarations if declaration]


def unicode_script_family(character: str) -> str | None:
    """Map letters to coarse writing systems while ignoring modifiers and combining marks."""
    if not character.isalpha() or unicodedata.category(character).startswith("M"):
        return None
    name = unicodedata.name(character, "")
    if not name:
        return None
    if any(label in name for label in ("CJK", "HIRAGANA", "KATAKANA", "IDEOGRAPH")):
        return "CJK"
    for label, family in (
        ("LATIN", "Latin"),
        ("CYRILLIC", "Cyrillic"),
        ("GREEK", "Greek"),
        ("HANGUL", "Hangul"),
        ("ARABIC", "Arabic"),
        ("HEBREW", "Hebrew"),
        ("DEVANAGARI", "Devanagari"),
        ("BENGALI", "Bengali"),
        ("THAI", "Thai"),
        ("ARMENIAN", "Armenian"),
        ("GEORGIAN", "Georgian"),
    ):
        if label in name:
            return family
    return None


def mixed_script_tokens(value: str) -> list[str]:
    mixed = []
    for token in ALPHABETIC_TOKEN.findall(value):
        families = {family for character in token if (family := unicode_script_family(character)) is not None}
        if len(families) > 1:
            mixed.append(token)
    return mixed


def _has_scattered_script_salad(value: str) -> bool:
    families = [family for character in value if (family := unicode_script_family(character)) is not None]
    if len(families) < 100:
        return False
    counts = Counter(families)
    non_latin_families = {family for family in counts if family != "Latin"}
    non_latin_count = sum(counts[family] for family in non_latin_families)
    if counts["Latin"] < 100 or len(non_latin_families) < 3 or non_latin_count / len(families) > 0.2:
        return False

    short_non_latin_runs = 0
    current_family: str | None = None
    current_length = 0
    for character in value + " ":
        family = unicode_script_family(character)
        if family == current_family and family is not None:
            current_length += 1
            continue
        if current_family not in {None, "Latin"} and current_length <= 6:
            short_non_latin_runs += 1
        current_family = family
        current_length = int(family is not None)
    return short_non_latin_runs >= 3


def assess_gibberish(value: str) -> GibberishAssessment:
    hard = []
    token_salad = []
    for label, pattern in HARD_CORRUPTION_PATTERNS:
        if pattern.search(value):
            hard.append(label)
    if len(TOKENIZER_ARTIFACT.findall(value)) >= 3:
        hard.append("raw tokenizer artifacts")
    sentinels = RAW_CHAT_SENTINEL.findall(value)
    if sentinels:
        without_allowed_trailer = re.sub(r"<\|im_end\|>\s*$", "", value, count=1)
        if RAW_CHAT_SENTINEL.search(without_allowed_trailer):
            hard.append("unexpected chat sentinel")
    if any(unicodedata.category(character) in {"Cc", "Cs", "Co"} for character in value if character not in "\n\r\t"):
        hard.append("control or private-use characters")

    for label, pattern in DEGENERATE_REPETITION_PATTERNS:
        if pattern.search(value):
            token_salad.append(label)
    words = re.findall(r"\b\w+\b", value.casefold(), re.UNICODE)
    for width in range(2, 6):
        if any(
            words[index : index + width] * 4 == words[index : index + width * 4]
            for index in range(0, len(words) - width * 4 + 1)
        ):
            token_salad.append("repeated phrase")
            break

    mixed = mixed_script_tokens(value)
    localized = []
    if len(mixed) == 1:
        localized.append(f"mixed-script token: {mixed[0][:80]}")
    elif len(mixed) > 1:
        examples = ", ".join(token[:40] for token in mixed[:3])
        token_salad.append(f"multiple mixed-script tokens ({len(mixed)}): {examples}")
    if _has_scattered_script_salad(value):
        token_salad.append("scattered non-Latin script fragments")
    return GibberishAssessment(tuple(hard), tuple(localized), tuple(token_salad))


def gibberish_reasons(value: str) -> list[str]:
    """Compatibility wrapper for callers that only need the flattened reason list."""
    return assess_gibberish(value).reasons


def verifier_policy_from_dataset(line: bytes, configured_policy: str) -> str:
    if configured_policy != "auto":
        return configured_policy
    datasets = extract_json_value(line, "dataset", []) or []
    if isinstance(datasets, str):
        datasets = [datasets]
    names = {str(item).casefold() for item in datasets}
    if "re_search_llm" in names:
        return "llm"
    if "re_search" in names:
        return "exact"
    return "legacy"


def verifier_visible_response(terminal: str, policy: str) -> tuple[str | None, str]:
    """Return a format failure and the exact response inspected for correctness."""
    if policy == "llm_format_gates":
        return apply_browsecomp_format_gates(terminal)
    if policy == "exact":
        return apply_re_search_format_gates(terminal)
    return None, terminal.strip()


def contains_reference_answer(response: str, ground_truths: list[str]) -> bool:
    """Check exact-normalized phrase containment against every accepted label."""
    normalized_response = normalize_answer(response)
    padded_response = f" {normalized_response} "
    return any(
        normalized_label and f" {normalized_label} " in padded_response
        for normalized_label in (normalize_answer(label) for label in ground_truths)
    )


def _append_segment(segments: list[dict[str, Any]], kind: str, content: str, *, tool_name: str | None = None) -> None:
    content = content.strip()
    if not content:
        return
    segment: dict[str, Any] = {"kind": kind, "content": content}
    if tool_name:
        segment["tool_name"] = tool_name
    segments.append(segment)


def _split_turns(text: str) -> list[tuple[str, str]]:
    """Split a decoded trajectory on chat-template turn boundaries."""
    matches = list(IM_START.finditer(text))
    if not matches:
        return [("assistant", text)]
    turns: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        # A stored response begins mid-turn, continuing the assistant turn that the
        # prompt opened, so the leading chunk carries no <|im_start|> marker.
        turns.append(("assistant", text[: matches[0].start()]))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        turns.append((match.group("role") or "assistant", text[match.end() : end]))
    return turns


def _emit_assistant_text(segments: list[dict[str, Any]], text: str) -> None:
    """Emit assistant prose, treating a dangling <think> as unterminated reasoning.

    A rollout truncated mid-thought leaves an opening tag with no partner, which
    would otherwise be shown as a response containing the literal tag.
    """
    opening = text.find("<think>")
    if opening < 0:
        _append_segment(segments, "assistant_text", text)
        return
    _append_segment(segments, "assistant_text", text[:opening])
    _append_segment(segments, "reasoning", text[opening + len("<think>") :])


def _assistant_segments(content: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    remainder = content
    opening = remainder.find("<think>")
    closing = remainder.find("</think>")
    if closing >= 0 and (opening < 0 or closing < opening):
        # The chat template emits <think> at the end of the prompt, so the first
        # reasoning block of a stored response is closed but never opened.
        _append_segment(segments, "reasoning", remainder[:closing])
        remainder = remainder[closing + len("</think>") :]
    cursor = 0
    for match in REASONING_OR_CALL.finditer(remainder):
        _emit_assistant_text(segments, remainder[cursor : match.start()])
        if match.group("reasoning") is not None:
            _append_segment(segments, "reasoning", match.group("reasoning"))
        else:
            body = match.group("call")
            name = FUNCTION_NAME.search(body)
            _append_segment(segments, "tool_call", body, tool_name=name.group("name") if name else None)
        cursor = match.end()
    _emit_assistant_text(segments, remainder[cursor:])
    return segments


def _observation_segments(content: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    cursor = 0
    for match in TOOL_RESPONSE.finditer(content):
        _append_segment(segments, "user_text", content[cursor : match.start()])
        _append_segment(segments, "tool_result", match.group("result"))
        cursor = match.end()
    _append_segment(segments, "user_text", content[cursor:])
    return segments


def segment_trajectory(text: str) -> list[dict[str, Any]]:
    """Split a decoded trajectory into ordered reasoning, tool, and prose segments.

    Assistant turns yield ``reasoning``, ``tool_call``, and ``assistant_text``
    segments; every other role yields ``tool_result`` and ``user_text`` segments.
    """
    segments: list[dict[str, Any]] = []
    for role, content in _split_turns(text):
        content = IM_END.sub("", content)
        produced = _assistant_segments(content) if role == "assistant" else _observation_segments(content)
        for segment in produced:
            segment["role"] = role
            segments.append(segment)
    for index, segment in enumerate(segments):
        segment["index"] = index
        segment["char_len"] = len(segment["content"])
    return segments


def _ground_truth_list(value: Any) -> list[str]:
    """Normalize the stored reference answer, which may hold several accepted forms."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or not str(value).strip():
        return []
    return [str(value)]


def _ground_truth(value: Any) -> str:
    entries = _ground_truth_list(value)
    return entries[0] if entries else ""


def incomplete_reason(terminal: str, termination_reason: str, generation_finish_reason: str) -> str | None:
    """Explain why the verifier never scored this rollout, or None if it was eligible.

    Mirrors the short-circuits shared by the exact and LLM verifiers: those
    rollouts are handed a zero without the verifier seeing a final answer, so lumping them
    in with genuinely wrong answers hides why the reward was zero.
    """
    # Checked before the empty-message case: a rollout cut off by the budget also
    # has no terminal message, and "ran out of response budget" is the actionable
    # cause while "no terminal message" is only the symptom.
    if termination_reason in INCOMPLETE_TERMINATIONS:
        return termination_reason
    if not terminal.strip():
        return "no_terminal_message"
    if termination_reason == "generation_complete" and generation_finish_reason not in {"", "stop"}:
        return "unclean_stop"
    return None


def summarize_line(
    line: bytes, pointer: RecordPointer, source: str, response_limit: int, verifier_policy: str = "auto"
) -> dict[str, Any]:
    step = int(extract_json_value(line, "step", -1))
    reward = float(extract_json_value(line, "reward", 0.0) or 0.0)
    advantage = extract_json_value(line, "advantage")
    token_count = json_array_length(line, "response_tokens")
    finish_reason = str(extract_json_value(line, "finish_reason", ""))
    ground_truths = _ground_truth_list(extract_json_value(line, "ground_truth", ""))
    ground_truth = ground_truths[0] if ground_truths else ""
    terminal = str(extract_json_value(line, "terminal_model_text", "") or "")
    prose = terminal_prose(terminal)
    declarations = answer_declarations(prose)
    gibberish = assess_gibberish(prose)
    num_calls = int(extract_json_value(line, "num_calls", 0) or 0)
    timeouts = int(extract_json_value(line, "timeouts", 0) or 0)
    tool_errors = str(extract_json_value(line, "tool_errors", "") or "")
    visit_404s = tool_errors.count("HTTP error: 404 Not Found")
    # request_info serializes tool_call_stats before rollout_state, so the first
    # match is the per-call list rather than the copy nested in the env state.
    tool_stats = extract_json_value(line, "tool_call_stats", []) or []
    successful_tool_calls = sum(1 for item in tool_stats if isinstance(item, dict) and item.get("success"))
    termination_reason = str(extract_json_value(line, "termination_reason", "") or "")
    generation_finish_reason = str(extract_json_value(line, "generation_finish_reason", "") or "")
    unjudged_reason = incomplete_reason(terminal, termination_reason, generation_finish_reason)
    capped = termination_reason != "generation_failure" and (finish_reason != "stop" or token_count >= response_limit)
    missing = not bool(prose)
    verifier_policy = verifier_policy_from_dataset(line, verifier_policy)
    format_error_reason, inspected_response = verifier_visible_response(terminal, verifier_policy)
    ground_truth_mentioned = contains_reference_answer(inspected_response, ground_truths)
    verifier_reached = unjudged_reason is None and format_error_reason is None

    categories: list[str] = []
    reasons: list[str] = []
    suspicion_score = 0

    def flag(category: str, reason: str, weight: int) -> None:
        nonlocal suspicion_score
        if category not in categories:
            categories.append(category)
        reasons.append(reason)
        suspicion_score += weight

    # Outcome first: a zero reward from a rollout the verifier never saw is a length
    # or tooling failure, not a wrong answer, and the two need different fixes.
    if reward > 0:
        outcome = "judged_correct"
    elif verifier_reached:
        outcome = "judged_incorrect"
    else:
        outcome = "incomplete"
    categories.append(outcome)
    if unjudged_reason is not None:
        flag(f"incomplete_{unjudged_reason}", f"Never reached the verifier: {unjudged_reason.replace('_', ' ')}", 0)

    # Weight-zero flags stay visible as queues but must not make a rollout
    # "suspicious": at this scale they describe the norm, not an anomaly.
    if source == "filtered":
        flag("filtered", "Discarded before training: the prompt group had zero reward variance", 0)
    if capped:
        flag("token_capped", "Trajectory hit a length or non-stop boundary", 0)
    if token_count >= 65_536:
        flag("long", f"Stored trajectory has {token_count:,} tokens", 0)
    if tool_errors:
        flag("tool_errors", "One or more tool calls returned an error", 0)

    # Length/non-stop capped trajectories are already classified by the
    # termination boundary. They never reached the verifier, so surfacing a
    # secondary format defect in their truncated terminal text is misleading.
    if format_error_reason is not None and not capped:
        flag("format_error", f"Verifier format check failed: {format_error_reason}", 5)
    if num_calls == 0:
        flag("no_tool_calls", "Trajectory made no tool calls", 3)
    if gibberish.hard_corruption:
        flag("hard_corruption", f"Hard corruption: {', '.join(gibberish.hard_corruption)}", 0)
    if gibberish.localized_corruption:
        flag("localized_corruption", f"Localized corruption: {', '.join(gibberish.localized_corruption)}", 0)
    if gibberish.token_salad:
        flag("token_salad", f"Token salad: {', '.join(gibberish.token_salad)}", 0)
    if gibberish.reasons:
        flag("gibberish", f"Gibberish screen: {', '.join(gibberish.reasons)}", 5)
    if missing:
        flag("no_final_answer", "No terminal model turn was captured", 5)
    if timeouts:
        flag("timeouts", f"Observed {timeouts} tool timeouts", 2)
    if verifier_reached and reward <= 0 and ground_truth_mentioned:
        flag("judge_negative_has_answer", "Verifier rejected a response containing an exact reference answer", 5)
    if verifier_reached and reward > 0 and not ground_truth_mentioned:
        flag("judge_positive_no_answer", "Verifier rewarded a response without an exact reference answer", 5)
    if suspicion_score == 0:
        categories.append("healthy_looking")

    return {
        "id": pointer.id,
        "source": source,
        "step": step,
        "optimizer_step": step + 1,
        "sample_idx": int(extract_json_value(line, "sample_idx", -1)),
        "prompt_idx": int(extract_json_value(line, "prompt_idx", -1)),
        "prompt_id": extract_json_value(line, "prompt_id"),
        "model_step": extract_json_value(line, "model_step"),
        "filter_reason": extract_json_value(line, "filter_reason"),
        "reward": reward,
        "advantage": advantage,
        "finish_reason": finish_reason,
        "token_count": token_count,
        "prompt_token_count": json_array_length(line, "prompt_tokens"),
        "num_calls": num_calls,
        "successful_tool_calls": successful_tool_calls,
        "timeouts": timeouts,
        "tool_error_count": len([item for item in tool_errors.splitlines() if item.strip()]),
        "visit_404s": visit_404s,
        "outcome": outcome,
        "judged": verifier_reached,
        "incomplete_reason": unjudged_reason,
        "termination_reason": termination_reason,
        "generation_finish_reason": generation_finish_reason,
        "ground_truth": ground_truth,
        "ground_truths": ground_truths,
        "ground_truth_mentioned": ground_truth_mentioned,
        "verifier_policy": verifier_policy,
        "format_error_reason": format_error_reason,
        "answer_declarations": declarations[:8],
        "answer_declaration_count": len(declarations),
        "gibberish_tiers": gibberish.tiers,
        "gibberish_reasons": gibberish.reasons,
        "terminal_preview": tail_preview(prose),
        "categories": categories,
        "reasons": reasons,
        "suspicion_score": suspicion_score,
    }


def group_difficulty(correct: int, size: int) -> str:
    if correct == 0:
        return "all_wrong_group"
    if correct == size:
        return "all_correct_group"
    pass_rate = correct / size
    if pass_rate <= HARD_GROUP_MAX_PASS_RATE:
        return "hard_group"
    if pass_rate >= EASY_GROUP_MIN_PASS_RATE:
        return "easy_group"
    return "learning_group"


def annotate_groups(records: list[dict[str, Any]]) -> None:
    """Label each record with the shape of the prompt group it belongs to.

    Whether every sample of a prompt succeeded or failed together is the property
    that decides if the group taught the learner anything, and it is never stored
    per record, so it has to be recomputed from the step's rollouts.

    Discarded records all carry ``prompt_idx`` 0 because each rejected group is
    saved on its own, so ``prompt_id`` is the only usable key when present.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    filtered_occurrences: Counter[str] = Counter()
    current_filtered_key: str | None = None
    for record in records:
        if record.get("source") == "filtered":
            base = str(record.get("prompt_id") or record.get("dataset_index") or record.get("prompt_idx"))
            # Filtered groups are appended one at a time and sample_idx restarts
            # at zero for every candidate group. Include an occurrence number so
            # the same prompt being rejected twice in one step remains two groups.
            if record.get("sample_idx") == 0 or current_filtered_key is None:
                filtered_occurrences[base] += 1
                current_filtered_key = f"{base}#{filtered_occurrences[base]}"
            key = current_filtered_key
        else:
            key = str(record.get("prompt_id") or record.get("prompt_idx"))
        grouped.setdefault(key, []).append(record)
    for key, members in grouped.items():
        size = len(members)
        correct = sum(1 for member in members if member["reward"] > 0)
        if correct == 0:
            shape = "all_wrong_group"
        elif correct == size:
            shape = "all_correct_group"
        else:
            shape = "mixed_group"
        difficulty = group_difficulty(correct, size)
        near_miss = shape == "mixed_group" and min(correct, size - correct) <= NEAR_MISS_MARGIN
        for member in members:
            member["group_key"] = key
            member["group_size"] = size
            member["group_correct"] = correct
            member["group_pass_rate"] = correct / size
            member["group_shape"] = shape
            member["group_difficulty"] = difficulty
            member["categories"].append(shape)
            if difficulty != shape:
                member["categories"].append(difficulty)
            if near_miss:
                member["categories"].append("near_miss_group")
                member["reasons"].append(f"Near-miss group: only {min(correct, size - correct)} of {size} differ")


class RolloutStore:
    def __init__(
        self,
        root: str | Path,
        *,
        response_limit: int = 131_072,
        cache_steps: int = 16,
        tokenizer_name: str | None = None,
        eval_index: Any = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.response_limit = response_limit
        self.cache_steps = cache_steps
        self.tokenizer_name = tokenizer_name
        self.eval_index = eval_index
        self._lock = threading.RLock()
        self._files: list[FileInfo] = []
        self._file_by_id: dict[str, FileInfo] = {}
        self._cache: OrderedDict[tuple[str, str, int], StepData] = OrderedDict()
        self._pointer_by_id: dict[str, RecordPointer] = {}
        self._trace_cache: OrderedDict[str, str] = OrderedDict()
        self._tokenizer: Any = None
        self._metadata: dict[str, dict[str, Any]] = {}
        # Physical timestamped attempts are mapped onto logical W&B lineages.
        # If W&B is unavailable, each attempt remains its own logical run.
        self._attempt_to_run: dict[str, str] = {}
        self._run_info: dict[str, dict[str, Any]] = {}
        self.refresh()

    @property
    def files(self) -> list[FileInfo]:
        with self._lock:
            return list(self._files)

    def refresh(self) -> None:
        if not self.root.is_dir():
            raise RolloutStoreError(f"Rollout directory does not exist: {self.root}")
        discovered: list[FileInfo] = []
        metadata: dict[str, dict[str, Any]] = {}
        for metadata_path in self.root.rglob("*_metadata.jsonl"):
            try:
                with metadata_path.open(encoding="utf-8") as handle:
                    row = json.loads(handle.readline())
                metadata[str(row.get("run_name") or metadata_path.stem)] = row
            except (OSError, json.JSONDecodeError):
                continue
        # Recursive so that one root can hold many training runs, each in its own
        # subdirectory with its own filtered/ shards. The glob matches accepted
        # (<run>_rollouts_N.jsonl) and filtered (<run>_filtered_rollouts_N.jsonl)
        # names alike; FILE_NAME decides which source a shard belongs to.
        candidates = list(self.root.rglob("*_rollouts_*.jsonl"))
        previous = {item.id: item for item in self._files}
        for path in sorted(set(candidates)):
            match = FILE_NAME.match(path.name)
            if match is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == 0:
                continue
            source = "filtered" if match.group("filtered") else "accepted"
            file_id = hashlib.sha1(os.fsencode(path)).hexdigest()[:12]
            # Carry over already-resolved boundaries for shards that did not change.
            carried = previous.get(file_id)
            unchanged = carried is not None and carried.size == stat.st_size and carried.mtime == stat.st_mtime
            discovered.append(
                FileInfo(
                    id=file_id,
                    path=path,
                    run=match.group("run"),
                    source=source,
                    shard=int(match.group("shard")),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    first_step=carried.first_step if unchanged else None,
                    last_step=carried.last_step if unchanged else None,
                )
            )
        discovered.sort(key=lambda item: (item.run, item.source, item.shard, str(item.path)))
        attempt_directories: dict[str, str] = {}
        for item in discovered:
            attempt_directories.setdefault(item.run, self._run_directory([item]))
        attempt_to_run: dict[str, str] = {}
        run_info: dict[str, dict[str, Any]] = {}
        for attempt, directory in attempt_directories.items():
            lineage = self.eval_index.lineage(attempt, directory) if self.eval_index is not None else None
            if lineage is None and getattr(self.eval_index, "registered_only", False):
                continue
            logical_run = lineage.get("logical_run") if lineage else None
            logical_run = logical_run or (f"wandb:{lineage['id']}" if lineage else attempt)
            attempt_to_run[attempt] = logical_run
            info = run_info.setdefault(
                logical_run,
                {
                    "label": lineage["name"] if lineage else attempt,
                    "wandb_run_id": (lineage.get("wandb_run_id", lineage["id"]) if lineage else None),
                    "registry_id": lineage.get("registry_id") if lineage else None,
                    "classification": lineage.get("classification") if lineage else None,
                    "visibility": lineage.get("visibility") if lineage else None,
                    "tags": lineage.get("tags", {}) if lineage else {},
                    "attempts": [],
                    "directory": directory,
                },
            )
            info["attempts"].append(attempt)
        if getattr(self.eval_index, "registered_only", False):
            discovered = [item for item in discovered if item.run in attempt_to_run]
            metadata = {attempt: value for attempt, value in metadata.items() if attempt in attempt_to_run}
        for info in run_info.values():
            attempt_key = (
                self.eval_index.precedence
                if self.eval_index is not None and hasattr(self.eval_index, "precedence")
                else self._attempt_start
            )
            info["attempts"].sort(key=attempt_key)
        fingerprint = [(item.id, item.size, item.mtime) for item in discovered]
        old_fingerprint = [(item.id, item.size, item.mtime) for item in self._files]
        with self._lock:
            self._files = discovered
            self._file_by_id = {item.id: item for item in discovered}
            self._metadata = metadata
            self._attempt_to_run = attempt_to_run
            self._run_info = run_info
            if fingerprint != old_fingerprint:
                self._cache.clear()
                self._pointer_by_id.clear()
                self._trace_cache.clear()

    def _resolve_run(self, run: str) -> list[FileInfo]:
        """Read step boundaries for one run's shards, deferring that cost until needed.

        Scanning back to a shard's last record costs megabytes of network reads, so
        resolving every run at startup would make the viewer slow to open.
        """
        with self._lock:
            pending = [item for item in self._files_for_run(run) if not item.resolved]
        for file_info in pending:
            try:
                with file_info.path.open("rb") as handle:
                    first = self._step_at(handle, 0)
                    last = self._step_at(handle, self._last_line_start(handle, file_info.size))
            except (OSError, ValueError):
                first = last = None
            with self._lock:
                if first is None or last is None:
                    self._files = [item for item in self._files if item.id != file_info.id]
                    self._file_by_id.pop(file_info.id, None)
                else:
                    file_info.first_step = first
                    file_info.last_step = last
        with self._lock:
            return [item for item in self._files_for_run(run) if item.resolved]

    def _files_for_run(self, run: str) -> list[FileInfo]:
        return [item for item in self._files if self._attempt_to_run.get(item.run, item.run) == run]

    def _attempt_start(self, attempt: str) -> int:
        match = re.search(r"__(\d+)$", attempt)
        return int(match.group(1)) if match else 0

    def _preferred_attempt(self, files: list[FileInfo]) -> str | None:
        """Newest process attempt wins where resumed rollout ranges overlap."""
        key = (
            self.eval_index.precedence
            if self.eval_index is not None and hasattr(self.eval_index, "precedence")
            else self._attempt_start
        )
        return max((item.run for item in files), key=key, default=None)

    def _verifier_policy(self, run: str) -> str:
        """Resolve the terminal format that actually governed one logical run."""
        tags = self._run_info.get(run, {}).get("tags", {})
        verifier = str(tags.get("verifier", "")).casefold()
        if "exact" in verifier:
            return "exact"
        if "llm" in verifier:
            judge = str(tags.get("judge", "")).casefold()
            return "llm_format_gates" if "format gates" in judge else "llm"
        return "auto"

    def meta(self) -> dict[str, Any]:
        with self._lock:
            runs = []
            for run in sorted(self._run_info, key=lambda item: self._run_info[item]["label"]):
                files = self._files_for_run(run)
                accepted = [item for item in files if item.source == "accepted"]
                filtered = [item for item in files if item.source == "filtered"]
                known = [item for item in files if item.resolved]
                info = self._run_info[run]
                attempts = info["attempts"]
                newest_attempt = max(
                    attempts,
                    key=(
                        self.eval_index.precedence
                        if self.eval_index is not None and hasattr(self.eval_index, "precedence")
                        else self._attempt_start
                    ),
                )
                runs.append(
                    {
                        "name": run,
                        "label": info["label"],
                        "wandb_run_id": info["wandb_run_id"],
                        "registry_id": info.get("registry_id"),
                        "classification": info.get("classification"),
                        "visibility": info.get("visibility"),
                        "tags": info.get("tags", {}),
                        "attempts": attempts,
                        # Null until the run is opened; /api/steps resolves on demand.
                        "first_step": min((item.first_step for item in known), default=None),
                        "last_step": max((item.last_step for item in known), default=None),
                        "resolved": len(known) == len(files),
                        "updated": max(item.mtime for item in files),
                        "accepted_files": sum(item.source == "accepted" for item in files),
                        "filtered_files": sum(item.source == "filtered" for item in files),
                        "accepted_first_step": min(
                            (item.first_step for item in accepted if item.resolved), default=None
                        ),
                        "accepted_last_step": max(
                            (item.last_step for item in accepted if item.resolved), default=None
                        ),
                        "filtered_first_step": min(
                            (item.first_step for item in filtered if item.resolved), default=None
                        ),
                        "filtered_last_step": max(
                            (item.last_step for item in filtered if item.resolved), default=None
                        ),
                        "metadata": self._metadata.get(newest_attempt, {}),
                        "attempt_metadata": [
                            self._metadata.get(attempt, {"run_name": attempt}) for attempt in attempts
                        ],
                        "files": [item.public() for item in files],
                    }
                )
            default_run = max(runs, key=lambda item: item["updated"])["name"] if runs else None
            return {
                "root": str(self.root),
                "default_run": default_run,
                "response_limit": self.response_limit,
                "runs": runs,
                "categories": CATEGORIES,
                "group_categories": GROUP_CATEGORIES,
            }

    def steps(self, run: str) -> dict[str, Any]:
        files = self._resolve_run(run)
        if not files:
            raise RolloutStoreError(f"Unknown run: {run}")
        first_step = min(item.first_step for item in files)
        last_step = max(item.last_step for item in files)
        source_ranges = {}
        for source in ("accepted", "filtered"):
            source_files = [item for item in files if item.source == source]
            if source_files:
                source_ranges[source] = {
                    "first_step": min(item.first_step for item in source_files),
                    "last_step": max(item.last_step for item in source_files),
                }
        evaluations = self._evaluations(run, files, first_step, last_step)
        return {
            "run": run,
            "first_step": first_step,
            "last_step": last_step,
            "steps": list(range(first_step, last_step + 1)),
            "evaluated_steps": [item["artifact_step"] for item in evaluations],
            "evaluations": evaluations,
            "source_ranges": source_ranges,
            "ranges": [item.public() for item in files],
        }

    def _evaluations(self, run: str, files: list[FileInfo], first_step: int, last_step: int) -> list[dict[str, Any]]:
        """Validation scores for the logical lineage, clipped to retained rollouts."""
        if self.eval_index is None:
            return []
        directory = self._run_directory(files)
        info = self._run_info.get(run, {})
        attempts = info.get("attempts") or [run]
        return [
            item
            for item in self.eval_index.evaluations(attempts[-1], directory, run_id=info.get("wandb_run_id"))
            if first_step <= item["artifact_step"] <= last_step
        ]

    @staticmethod
    def _run_directory(files: list[FileInfo]) -> str:
        """Name of the directory holding a run, ignoring the filtered/ subdirectory."""
        for file_info in files:
            parent = file_info.path.parent
            return (parent.parent if parent.name == "filtered" else parent).name
        return ""

    def query(
        self,
        *,
        run: str,
        step: int,
        source: str = "accepted",
        category: str = "review",
        search: str = "",
        sort: str = "suspicion",
        group_key: str = "",
        page: int = 1,
        page_size: int = 24,
    ) -> dict[str, Any]:
        if source not in {"accepted", "filtered", "both"}:
            raise RolloutStoreError(f"Invalid source: {source}")
        sources = ["accepted", "filtered"] if source == "both" else [source]
        records: list[dict[str, Any]] = []
        for selected_source in sources:
            records.extend(self._load_step(run, selected_source, step).records)
        category_counts = Counter()
        for record in records:
            category_counts.update(record["categories"])
            if record["suspicion_score"] > 0:
                category_counts["review"] += 1
        filtered = records
        if category == "review":
            filtered = [record for record in filtered if record["suspicion_score"] > 0]
        elif category != "all":
            filtered = [record for record in filtered if category in record["categories"]]
        if group_key:
            filtered = [record for record in filtered if record.get("group_key") == group_key]
        search_value = normalize_text(search)
        if search_value:
            filtered = [
                record
                for record in filtered
                if search_value
                in normalize_text(
                    " ".join(
                        [
                            record.get("ground_truth") or "",
                            record.get("terminal_preview") or "",
                            record.get("prompt_id") or "",
                            " ".join(record.get("answer_declarations") or []),
                        ]
                    )
                )
            ]
        sorters = {
            "suspicion": lambda record: (-record["suspicion_score"], -record["token_count"]),
            "tokens": lambda record: (-record["token_count"], -record["suspicion_score"]),
            "calls": lambda record: (-record["num_calls"], -record["suspicion_score"]),
            "sample": lambda record: (record["prompt_idx"], record["sample_idx"]),
            "reward": lambda record: (record["reward"], -record["suspicion_score"]),
        }
        if sort not in sorters:
            raise RolloutStoreError(f"Invalid sort: {sort}")
        filtered.sort(key=sorters[sort])
        page_size = max(1, min(page_size, 50))
        page = max(1, page)
        start = (page - 1) * page_size
        end = start + page_size
        rewards = [record["reward"] for record in records]
        tokens = [record["token_count"] for record in records]
        return {
            "run": run,
            "step": step,
            "optimizer_step": step + 1,
            "source": source,
            "category": category,
            "page": page,
            "page_size": page_size,
            "total": len(filtered),
            "has_more": end < len(filtered),
            "records": filtered[start:end],
            "category_counts": dict(category_counts),
            "stats": {
                "records": len(records),
                "reward_rate": sum(rewards) / len(rewards) if rewards else None,
                "average_tokens": sum(tokens) / len(tokens) if tokens else None,
                "suspicious": sum(record["suspicion_score"] > 0 for record in records),
            },
        }

    def groups(
        self,
        *,
        run: str,
        step: int,
        source: str = "accepted",
        category: str = "all_groups",
        search: str = "",
        sort: str = "reward",
        page: int = 1,
        page_size: int = 24,
    ) -> dict[str, Any]:
        """Return one compact summary per prompt group for a rollout step."""
        if source not in {"accepted", "filtered", "both"}:
            raise RolloutStoreError(f"Invalid source: {source}")
        sources = ["accepted", "filtered"] if source == "both" else [source]
        records: list[dict[str, Any]] = []
        for selected_source in sources:
            records.extend(self._load_step(run, selected_source, step).records)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault((record["source"], record["group_key"]), []).append(record)

        summaries: list[dict[str, Any]] = []
        category_counts: Counter[str] = Counter()
        for (group_source, key), members in grouped.items():
            first = members[0]
            size = len(members)
            correct = sum(member["reward"] > 0 for member in members)
            difficulty = group_difficulty(correct, size)
            category_counts[difficulty] += 1
            tokens = [member["token_count"] for member in members]
            summaries.append(
                {
                    "id": f"{group_source}:{key}",
                    "group_key": key,
                    "source": group_source,
                    "step": step,
                    "optimizer_step": step + 1,
                    "prompt_id": first.get("prompt_id"),
                    "prompt_idx": first.get("prompt_idx"),
                    "ground_truth": first.get("ground_truth"),
                    "ground_truths": first.get("ground_truths"),
                    "size": size,
                    "correct": correct,
                    "pass_rate": correct / size,
                    "difficulty": difficulty,
                    "average_tokens": sum(tokens) / size,
                    "token_capped": sum("token_capped" in member["categories"] for member in members),
                    "format_errors": sum("format_error" in member["categories"] for member in members),
                    "incomplete": sum(member["outcome"] == "incomplete" for member in members),
                    "suspicious": sum(member["suspicion_score"] > 0 for member in members),
                    "average_calls": sum(member["num_calls"] for member in members) / size,
                    "terminal_preview": next(
                        (member["terminal_preview"] for member in members if member["terminal_preview"]), ""
                    ),
                }
            )

        filtered = summaries
        if category != "all_groups":
            valid_categories = {item["id"] for item in GROUP_CATEGORIES}
            if category not in valid_categories:
                raise RolloutStoreError(f"Invalid group category: {category}")
            filtered = [group for group in filtered if group["difficulty"] == category]
        search_value = normalize_text(search)
        if search_value:
            filtered = [
                group
                for group in filtered
                if search_value
                in normalize_text(
                    " ".join(
                        [
                            group.get("ground_truth") or "",
                            group.get("prompt_id") or "",
                            group.get("terminal_preview") or "",
                        ]
                    )
                )
            ]

        sorters = {
            "suspicion": lambda group: (-group["suspicious"], group["pass_rate"]),
            "tokens": lambda group: (-group["average_tokens"], group["pass_rate"]),
            "calls": lambda group: (-group["average_calls"], group["pass_rate"]),
            "sample": lambda group: (group["source"], group["prompt_idx"], group["group_key"]),
            "reward": lambda group: (group["pass_rate"], -group["suspicious"]),
        }
        if sort not in sorters:
            raise RolloutStoreError(f"Invalid sort: {sort}")
        filtered.sort(key=sorters[sort])

        page_size = max(1, min(page_size, 50))
        page = max(1, page)
        start = (page - 1) * page_size
        end = start + page_size
        pass_rates = [group["pass_rate"] for group in summaries]
        trajectory_rewards = [record["reward"] for record in records]
        return {
            "run": run,
            "step": step,
            "optimizer_step": step + 1,
            "source": source,
            "category": category,
            "page": page,
            "page_size": page_size,
            "total": len(filtered),
            "has_more": end < len(filtered),
            "groups": filtered[start:end],
            "category_counts": dict(category_counts),
            "stats": {
                "groups": len(summaries),
                "trajectories": len(records),
                "mean_group_pass_rate": sum(pass_rates) / len(pass_rates) if pass_rates else None,
                "trajectory_reward_rate": (
                    sum(trajectory_rewards) / len(trajectory_rewards) if trajectory_rewards else None
                ),
                "average_tokens": (
                    sum(record["token_count"] for record in records) / len(records) if records else None
                ),
                "suspicious": sum(group["suspicious"] > 0 for group in summaries),
            },
        }

    def detail(self, record_id: str) -> dict[str, Any]:
        pointer, file_info, line = self._record_line(record_id)
        logical_run = self._attempt_to_run.get(file_info.run, file_info.run)
        summary = summarize_line(
            line, pointer, file_info.source, self.response_limit, self._verifier_policy(logical_run)
        )
        # Group shape is only known for a whole step, so reuse the cached annotation
        # from the listing the card was opened from.
        with self._lock:
            cached = self._cache.get((logical_run, file_info.source, summary["step"]))
        if cached is not None:
            annotated = next((item for item in cached.records if item["id"] == pointer.id), None)
            if annotated is not None:
                summary = dict(annotated)
        terminal = str(extract_json_value(line, "terminal_model_text", "") or "")
        verifier_input = extract_json_value(line, "verifier_input")
        verifier_skipped_reason = extract_json_value(line, "verifier_skipped_reason")
        judge_output = extract_json_value(line, "judge_output")
        verifier_input_source = "saved" if isinstance(verifier_input, str) else None
        verifier_skipped_reason_source = "saved" if verifier_skipped_reason is not None else None
        if not verifier_skipped_reason and isinstance(judge_output, dict):
            verifier_skipped_reason = judge_output.get("skipped")
            if verifier_skipped_reason is not None:
                verifier_skipped_reason_source = "saved"
        if not isinstance(verifier_input, str) and verifier_skipped_reason is None:
            verifier_input, verifier_skipped_reason = self._reconstruct_verifier_input(
                line=line,
                file_info=file_info,
                terminal_response=terminal,
                termination_reason=summary["termination_reason"],
                generation_finish_reason=summary["generation_finish_reason"],
                reference_answer=summary["ground_truth"],
            )
            if verifier_input is not None:
                verifier_input_source = "reconstructed"
            if verifier_skipped_reason is not None:
                verifier_skipped_reason_source = "reconstructed"
        raw_prompt = extract_json_value(line, "raw_prompt")
        decoded_response = extract_json_value(line, "decoded_response")
        tool_outputs = str(extract_json_value(line, "tool_outputs", "") or "")
        tool_errors = str(extract_json_value(line, "tool_errors", "") or "")
        return {
            **summary,
            "file": file_info.public(),
            "raw_prompt": raw_prompt,
            "terminal_response": terminal,
            "verifier_input": verifier_input if isinstance(verifier_input, str) else None,
            "verifier_input_source": verifier_input_source,
            "verifier_skipped_reason": (str(verifier_skipped_reason) if verifier_skipped_reason is not None else None),
            "verifier_skipped_reason_source": verifier_skipped_reason_source,
            "judge_output": judge_output if isinstance(judge_output, dict) else None,
            "decoded_response_preview": (decoded_response[:200_000] if isinstance(decoded_response, str) else None),
            "decoded_response_truncated": isinstance(decoded_response, str) and len(decoded_response) > 200_000,
            "tool_outputs": tool_outputs[:80_000],
            "tool_outputs_truncated": len(tool_outputs) > 80_000,
            "tool_errors": tool_errors[:40_000],
            "tool_errors_truncated": len(tool_errors) > 40_000,
            "trace_available": bool(json_array_length(line, "response_tokens")),
        }

    def _reconstruct_verifier_input(
        self,
        *,
        line: bytes,
        file_info: FileInfo,
        terminal_response: str,
        termination_reason: str,
        generation_finish_reason: str,
        reference_answer: str,
    ) -> tuple[str | None, str | None]:
        """Rebuild an old judge request in memory without mutating its artifact."""
        logical_run = self._attempt_to_run.get(file_info.run, file_info.run)
        tags = self._run_info.get(logical_run, {}).get("tags", {})
        verifier = str(tags.get("verifier", "")).casefold()
        if "llm" not in verifier:
            return None, None
        if not terminal_response.strip():
            return None, "no_terminal_model_text"
        if termination_reason in {"max_steps", "response_limit", "context_limit", "reset_failure"}:
            return None, f"termination_reason:{termination_reason}"
        if termination_reason == "generation_complete" and generation_finish_reason not in {"", "stop"}:
            return None, f"generation_finish_reason:{generation_finish_reason}"

        response = terminal_response
        judge = str(tags.get("judge", "")).casefold()
        if "format gates" in judge:
            gate_reason, response = apply_browsecomp_format_gates(terminal_response)
            if gate_reason is not None:
                return None, f"format:{gate_reason}"

        prompt = self._prompt_text(line)
        if not prompt:
            return None, None
        return (
            BROWSECOMP_CORRECTNESS_JUDGE_PROMPT.format(
                question=question_from_prompt(prompt),
                reference_answer=format_reference_answer(reference_answer),
                response=response,
            ),
            None,
        )

    def trace(self, record_id: str, offset: int = 0, limit: int = 50_000) -> dict[str, Any]:
        pointer, _, line = self._record_line(record_id)
        text = self._decoded_text(record_id, line)
        offset = max(0, offset)
        limit = max(1_000, min(limit, 100_000))
        return {
            "id": pointer.id,
            "offset": offset,
            "limit": limit,
            "total_chars": len(text),
            "content": text[offset : offset + limit],
            "has_more": offset + limit < len(text),
        }

    def turns(self, record_id: str, max_chars_per_turn: int = 8_000) -> dict[str, Any]:
        """Return the trajectory split into ordered, individually truncated segments.

        Tool observations dominate stored trajectories, so each segment is capped to
        keep the payload small; the full text stays behind ``trace``.
        """
        pointer, file_info, line = self._record_line(record_id)
        max_chars_per_turn = max(500, min(max_chars_per_turn, 40_000))
        payload = self._trajectory_segments(record_id, line, max_chars_per_turn)
        reference_answer = _ground_truth(extract_json_value(line, "ground_truth", ""))
        reference_matches = self._literal_matches(reference_answer, payload)
        return {
            "id": pointer.id,
            "source": file_info.source,
            "max_chars_per_turn": max_chars_per_turn,
            "total_segments": len(payload),
            "kind_counts": dict(Counter(item["kind"] for item in payload)),
            "segments": payload,
            "reference_matches": reference_matches,
        }

    def matches(self, record_id: str, query: str, max_chars_per_turn: int = 8_000) -> dict[str, Any]:
        """Search a complete training trajectory without returning every full turn."""
        term = query.strip()
        if not term:
            raise RolloutStoreError("Trajectory search query must not be empty")
        if len(term) > 256:
            raise RolloutStoreError("Trajectory search query must be at most 256 characters")
        pointer, file_info, line = self._record_line(record_id)
        max_chars_per_turn = max(500, min(max_chars_per_turn, 40_000))
        segments = self._trajectory_segments(record_id, line, max_chars_per_turn)
        return {"id": pointer.id, "source": file_info.source, **self._literal_matches(term, segments)}

    def _trajectory_segments(self, record_id: str, line: bytes, max_chars_per_turn: int) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        prompt = self._prompt_text(line)
        if prompt:
            segments.append({"role": "user", "kind": "prompt", "content": prompt})
        segments.extend(segment_trajectory(self._decoded_text(record_id, line)))

        # The terminal assistant prose is the final response. Earlier plain-text
        # assistant messages remain reasoning/text, matching EvaluationStore.
        terminal_index = next(
            (index for index in range(len(segments) - 1, -1, -1) if segments[index]["kind"] == "assistant_text"), None
        )
        if terminal_index is not None:
            segments[terminal_index] = {**segments[terminal_index], "kind": "final_output"}

        payload: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            content = segment["content"]
            kind = segment["kind"]
            match_category = "reasoning" if kind == "assistant_text" else kind
            if match_category not in TRAJECTORY_MATCH_CATEGORIES:
                match_category = None
            payload.append(
                {
                    "index": index,
                    "role": segment.get("role", "assistant"),
                    "kind": kind,
                    "tool_name": segment.get("tool_name"),
                    "content": content[:max_chars_per_turn],
                    "char_len": len(content),
                    "truncated": len(content) > max_chars_per_turn,
                    "match_category": match_category,
                    "_full_content": content,
                }
            )
        return payload

    @staticmethod
    def _literal_matches(term: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {category: 0 for category in TRAJECTORY_MATCH_CATEGORIES}
        matches: list[dict[str, Any]] = []
        pattern = re.compile(re.escape(term), re.IGNORECASE) if term else None
        for segment in segments:
            content = str(segment.pop("_full_content", segment.get("content") or ""))
            category = segment.get("match_category")
            segment_match_count = 0
            segment_matches = () if category not in counts or pattern is None else pattern.finditer(content)
            for match in segment_matches:
                segment_match_count += 1
                if len(matches) >= TRAJECTORY_MATCH_LIMIT:
                    continue
                excerpt_start = max(0, match.start() - 100)
                excerpt_end = min(len(content), match.end() + 100)
                matches.append(
                    {
                        "segment_index": segment["index"],
                        "segment_kind": segment["kind"],
                        "category": category,
                        "start": match.start(),
                        "end": match.end(),
                        "excerpt": content[excerpt_start:excerpt_end],
                        "excerpt_match_start": match.start() - excerpt_start,
                        "excerpt_match_end": match.end() - excerpt_start,
                        "in_preview": match.end() <= len(segment["content"]),
                    }
                )
            segment["reference_answer_match_count"] = segment_match_count
            if category in counts:
                counts[category] += segment_match_count
        total = sum(counts.values())
        return {
            "term": term,
            "total": total,
            "counts": counts,
            "matches": matches,
            "returned": len(matches),
            "truncated": total > len(matches),
            "limit": TRAJECTORY_MATCH_LIMIT,
        }

    def _prompt_text(self, line: bytes) -> str:
        """Prefer the stored raw prompt, falling back to decoding prompt tokens."""
        raw_prompt = extract_json_value(line, "raw_prompt")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            return raw_prompt.strip()
        try:
            decoded = self._decode_token_field(line, "prompt_tokens")
        except (RolloutStoreError, ValueError, json.JSONDecodeError):
            return ""
        return decoded.strip() if decoded else ""

    def _decoded_text(self, record_id: str, line: bytes) -> str:
        """Decode one response, reusing an LRU cache of at most four traces."""
        with self._lock:
            cached = self._trace_cache.get(record_id)
            if cached is not None:
                self._trace_cache.move_to_end(record_id)
                return cached
        decoded_response = extract_json_value(line, "decoded_response")
        if isinstance(decoded_response, str):
            text = decoded_response
        else:
            text = self._decode_token_field(line, "response_tokens")
            if text is None:
                raise RolloutStoreError("This rollout has no response tokens")
        with self._lock:
            self._trace_cache[record_id] = text
            self._trace_cache.move_to_end(record_id)
            while len(self._trace_cache) > 4:
                self._trace_cache.popitem(last=False)
        return text

    def _decode_token_field(self, line: bytes, key: str) -> str | None:
        token_start = _value_start(line, key)
        if token_start is None:
            return None
        token_end = _value_end(line, token_start)
        tokens = json.loads(line[token_start:token_end])
        if not tokens:
            return None
        return self._get_tokenizer().decode(tokens, skip_special_tokens=False)

    def prewarm_tokenizer(self) -> None:
        """Load the tokenizer ahead of the first decode, ignoring any failure.

        Loading it costs tens of seconds, which would otherwise be paid by whoever
        opens the first accepted rollout.
        """
        try:
            self._get_tokenizer()
        except RolloutStoreError:
            return

    def _get_tokenizer(self) -> Any:
        with self._lock:
            if self._tokenizer is not None:
                return self._tokenizer
        model_name = self.tokenizer_name
        if not model_name:
            for metadata in self._metadata.values():
                candidate = metadata.get("model_name")
                if candidate:
                    model_name = str(candidate)
                    break
        if not model_name:
            raise RolloutStoreError("No tokenizer configured; restart with --tokenizer MODEL")
        try:
            AutoTokenizer = import_module("transformers").AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception as error:
            raise RolloutStoreError(
                f"Could not load tokenizer {model_name!r} from the local cache: {error}"
            ) from error
        with self._lock:
            self._tokenizer = tokenizer
        return tokenizer

    def _load_step(self, run: str, source: str, step: int) -> StepData:
        cache_key = (run, source, step)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached
        candidates = [
            item
            for item in self._resolve_run(run)
            if item.source == source and item.first_step <= step <= item.last_step
        ]
        preferred_attempt = self._preferred_attempt(candidates)
        files = [item for item in candidates if item.run == preferred_attempt]
        records: list[dict[str, Any]] = []
        pointers: dict[str, RecordPointer] = {}
        verifier_policy = self._verifier_policy(run)
        for file_info in files:
            for pointer, line in self._lines_for_step(file_info, step):
                records.append(summarize_line(line, pointer, source, self.response_limit, verifier_policy))
                pointers[pointer.id] = pointer
        # Group shape needs the whole step, so it is applied after the scan.
        annotate_groups(records)
        category_counts = Counter()
        for record in records:
            category_counts.update(record["categories"])
        result = StepData(
            run=run,
            source=source,
            step=step,
            records=records,
            category_counts=category_counts,
            total_records=len(records),
        )
        with self._lock:
            self._cache[cache_key] = result
            self._pointer_by_id.update(pointers)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_steps:
                self._cache.popitem(last=False)
        return result

    def _lines_for_step(self, file_info: FileInfo, step: int) -> Iterator[tuple[RecordPointer, bytes]]:
        with file_info.path.open("rb") as handle:
            start = self._first_step_offset(handle, file_info, step)
            if start is None:
                return
            handle.seek(start)
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                current_step = self._step_from_line(line)
                if current_step != step:
                    break
                pointer_id = f"{file_info.id}:{offset}"
                pointer = RecordPointer(pointer_id, file_info.id, offset, len(line))
                yield pointer, line

    def _first_step_offset(self, handle: BinaryIO, file_info: FileInfo, target: int) -> int | None:
        if target < file_info.first_step or target > file_info.last_step:
            return None
        if target == file_info.first_step:
            return 0
        low = 0
        high = file_info.size
        while high - low > 65_536:
            middle = (low + high) // 2
            line_start = self._next_line_start(handle, middle, file_info.size)
            if line_start >= file_info.size:
                high = middle
                continue
            if line_start >= high:
                # The midpoint landed inside the same very large JSON line as the
                # current upper bound. Shrink by bytes so the search still converges.
                high = middle
                continue
            current_step = self._step_at(handle, line_start)
            if current_step < target:
                low = line_start
            else:
                high = line_start
        current = self._next_line_start(handle, low, file_info.size)
        while current < file_info.size:
            current_step = self._step_at(handle, current)
            if current_step == target:
                return current
            if current_step > target:
                return None
            current = self._next_line_start(handle, current, file_info.size)
        return None

    def _record_line(self, record_id: str) -> tuple[RecordPointer, FileInfo, bytes]:
        with self._lock:
            pointer = self._pointer_by_id.get(record_id)
        if pointer is None:
            try:
                file_id, offset_string = record_id.split(":", 1)
                offset = int(offset_string)
                file_info = self._file_by_id[file_id]
            except (ValueError, KeyError) as error:
                raise RolloutStoreError(f"Unknown rollout id: {record_id}") from error
            with file_info.path.open("rb") as handle:
                handle.seek(offset)
                line = handle.readline()
            pointer = RecordPointer(record_id, file_id, offset, len(line))
        file_info = self._file_by_id.get(pointer.file_id)
        if file_info is None:
            raise RolloutStoreError("The source rollout file is no longer available")
        with file_info.path.open("rb") as handle:
            handle.seek(pointer.offset)
            line = handle.readline()
        if not line:
            raise RolloutStoreError("The rollout record is no longer available")
        return pointer, file_info, line

    @staticmethod
    def _step_from_line(line: bytes) -> int:
        match = STEP_PREFIX.match(line)
        if match is None:
            raise ValueError("Rollout line does not begin with a step")
        return int(match.group(1))

    def _step_at(self, handle: BinaryIO, offset: int) -> int:
        handle.seek(offset)
        prefix = handle.read(96)
        return self._step_from_line(prefix)

    @staticmethod
    def _next_line_start(handle: BinaryIO, offset: int, size: int) -> int:
        if offset < 0:
            return 0
        handle.seek(offset)
        position = offset
        while position < size:
            chunk = handle.read(min(65_536, size - position))
            if not chunk:
                return size
            newline = chunk.find(b"\n")
            if newline >= 0:
                return position + newline + 1
            position += len(chunk)
        return size

    @staticmethod
    def _last_line_start(handle: BinaryIO, size: int) -> int:
        """Locate the final line, reading backwards in blocks that grow as needed.

        A stored record can span several megabytes, so scanning in small windows
        would cost dozens of round trips per shard on network storage.
        """
        if size <= 0:
            return 0
        end = size
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            end -= 1
        position = end
        window = 1 << 20
        while position > 0:
            start = max(0, position - window)
            handle.seek(start)
            chunk = handle.read(position - start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                return start + newline + 1
            position = start
            window = min(window * 2, 1 << 24)
        return 0
