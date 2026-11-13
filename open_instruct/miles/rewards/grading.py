"""Select grading inputs without changing the rollout or its training tokens.

This module owns reasoning boundaries for MILES final-answer grading. Task
parsers still own answer wrappers, mathematical syntax, programs and tool calls.
Legacy dispatch and direct verifier calls retain their existing extraction in
their own adapters; that normalization is not part of this selector.
"""

from dataclasses import dataclass
from typing import Literal

from open_instruct.miles.rewards import response_text

Scope = Literal["response", "thinking", "full"]
ReasoningMode = Literal["required", "none"]
SCOPES = ("response", "thinking", "full")
REASONING_MODES = ("required", "none")
# Also identifies the effective grading contract in persisted evidence. Version
# 2 sends selected judge inputs intact, including answer tags and whitespace.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ResponseParts:
    """One immutable view of a generated assistant response.

    ``raw`` retains transport terminators; ``full`` removes only the shared
    terminal markers. ``thinking`` excludes an optional initial opening tag.
    ``response`` is the delivered answer or action, not an extracted scalar.
    A missing response is None, distinct from a present but empty response.
    Character offsets refer to ``full``; token boundaries cannot be inferred
    from decoded text and are deliberately not fabricated here.
    """

    raw: str
    full: str
    thinking: str | None
    response: str | None
    response_start: int | None
    reasoning_mode: ReasoningMode
    prompt_opens_reasoning: bool
    prompt_closes_reasoning: bool
    closing_markers: int
    completion_status: str | None
    gate: str | None

    def text(self, scope: Scope) -> str:
        if scope == "response":
            return self.response or ""
        if scope == "thinking":
            return self.thinking or ""
        if scope == "full":
            return self.full
        raise ValueError(f"Unknown grading scope: {scope!r}")

    def summary(self) -> dict:
        """Compact provenance for records; do not duplicate generated text."""
        return {
            "schema_version": SCHEMA_VERSION,
            "reasoning_mode": self.reasoning_mode,
            "prompt_opens_reasoning": self.prompt_opens_reasoning,
            "prompt_closes_reasoning": self.prompt_closes_reasoning,
            "closing_markers": self.closing_markers,
            "response_start": self.response_start,
            "thinking_characters": None if self.thinking is None else len(self.thinking),
            "response_characters": None if self.response is None else len(self.response),
            "completion_status": self.completion_status,
            "gate": self.gate,
        }


def split_response(
    raw: str, *, reasoning_mode: ReasoningMode = "required", prompt=None, completion_status: str | None = None
) -> ResponseParts:
    """Split once using a declared reasoning convention, never guessed from output.

    Required reasoning supports an opener prefilled in the prompt and retains
    the existing last-closing-marker policy. Repeated markers are recorded.
    Non-thinking runs must opt into ``none``; reasoning tags are reserved control
    syntax in this mode. Literal/quoted tags remain a documented limitation.
    """
    if reasoning_mode not in REASONING_MODES:
        raise ValueError(f"reasoning_mode must be one of {REASONING_MODES}")
    full = response_text.strip_terminal_special_tokens(raw)
    opened = isinstance(prompt, str) and prompt.rstrip().endswith("<think>")
    closed = isinstance(prompt, str) and prompt.rstrip().endswith("</think>")
    count = full.count("</think>")
    thinking, response, start, gate = None, None, None, None
    if reasoning_mode == "none":
        if opened or "<think>" in full or count:
            gate = "unexpected_reasoning"
        else:
            response, start = full, 0
    else:
        prefix, separator, suffix = full.rpartition("</think>")
        if separator:
            thinking = prefix.removeprefix("<think>")
            response, start = suffix, len(prefix) + len(separator)
            if suffix.lstrip().startswith("<think>"):
                gate = "unfinished_reasoning"
        elif closed:
            # Some templates prefill a closed (often empty) thinking section.
            # Its tokens belong to the prompt; generated text is already final.
            thinking, response, start = "", full, 0
            if full.lstrip().startswith("<think>"):
                gate = "unfinished_reasoning"
        else:
            thinking = full.removeprefix("<think>")
    if gate is None and (response is None or not response.strip()):
        gate = "missing_final_answer"
    return ResponseParts(
        raw, full, thinking, response, start, reasoning_mode, opened, closed, count, completion_status, gate
    )


@dataclass(frozen=True)
class GradingPolicy:
    """Trusted registry policy, separate from dataset targets and weights."""

    scope: Scope = "response"
    require_response: bool = True

    @classmethod
    def from_spec(cls, spec: dict, *, default_scope: Scope = "response"):
        value = spec.get("grading", {})
        if not isinstance(value, dict) or set(value) - {"scope", "require_response"}:
            raise ValueError("grading accepts only scope and require_response")
        scope = value.get("scope", default_scope)
        required = value.get("require_response", True)
        if scope not in SCOPES or type(required) is not bool:
            raise ValueError("grading requires scope=response|thinking|full and boolean require_response")
        if not required and value.get("scope") not in ("thinking", "full"):
            raise ValueError("grading.require_response=false requires explicit scope=thinking|full")
        return cls(scope, required)

    def gate(self, parts: ResponseParts, *, zero_truncated: bool) -> str | None:
        if zero_truncated and parts.completion_status == "truncated":
            return "truncated"
        return parts.gate if self.require_response else None
