#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E501

"""Obtain rubric-based rewards from an OpenAI-compatible language-model judge.
This module builds the shared judge prompts, checks the request's context budget,
submits bounded requests and parses the returned verdict or numeric score. It
records response and failure evidence on each sample so training can distinguish
a low grade from an unavailable service or an unreadable judge reply.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import requests

from open_instruct import judge_utils, logger_utils
from open_instruct.answer_utils import extract_final_answer
from open_instruct.miles.infrastructure import infra_timeouts
from open_instruct.miles.rewards import abstention, judge_registry, judging, service

logger = logger_utils.setup_logger(__name__)

_PROMPTS = {
    "general-quality": (
        judge_utils.general_quality_template,
        "dc5690758431cfeb5d5eb308540eddb728a581ef2c341a73ca12fb4beaa6e6b1",
    ),
    "general-quality_ref": (
        judge_utils.general_quality_ref_template,
        "33907cb811254249db7fd78683ebb7c2cc194c9e04dc78472dde0663b0762d4e",
    ),
    "general-web_instruct_general_verifier": (
        judge_utils.web_instruct_general_verifier_template,
        "6d14cbfb0909d57f3f607d3dc92f9f35ee98219434f62b3ab7346c95dc29b098",
    ),
}
_EXECUTORS: dict[int, ThreadPoolExecutor] = {}


@dataclass(frozen=True, kw_only=True)
class GeneralJudgeConfig:
    """Validated settings for one OpenAI-compatible judge request."""

    api_url: str
    api_key: str
    model: str
    max_tokens: int
    max_context_length: int
    temperature: float
    timeout: float
    seed: int
    max_concurrent_calls: int
    check_context: bool = False


def general_judge_config(args: Any) -> GeneralJudgeConfig:
    """Resolve general-judge settings without requiring Open-Instruct or LiteLLM."""
    api_base = (
        getattr(args, "llm_judge_api_base", None)
        or os.environ.get("OI_MILES_JUDGE_API_BASE")
        or os.environ.get("HOSTED_VLLM_API_BASE")
    )
    if not api_base:
        raise RuntimeError("general-quality requires OI_MILES_JUDGE_API_BASE or HOSTED_VLLM_API_BASE")
    api_url = str(api_base).rstrip("/")
    if not api_url.endswith("/chat/completions"):
        api_url += "/chat/completions"
    model = getattr(args, "llm_judge_model", None) or os.environ.get("OI_MILES_JUDGE_MODEL", "Qwen/Qwen3-32B")
    model = str(model)
    if model.startswith("hosted_vllm/"):
        model = model.removeprefix("hosted_vllm/")
    max_tokens = int(service.number_setting(args, "llm_judge_max_tokens", "OI_MILES_JUDGE_MAX_TOKENS", 2048))
    max_context_length = int(
        service.number_setting(args, "llm_judge_max_context_length", "OI_MILES_JUDGE_MAX_CONTEXT_LENGTH", 32768)
    )
    temperature = float(service.number_setting(args, "llm_judge_temperature", "OI_MILES_JUDGE_TEMPERATURE", 1.0))
    timeout = float(
        service.number_setting(
            args, "llm_judge_timeout", "OI_MILES_JUDGE_TIMEOUT", float(judging.DEFAULT_REQUEST_TIMEOUT)
        )
    )
    seed = int(service.number_setting(args, "seed", "OI_MILES_JUDGE_SEED", 1))
    max_concurrent_calls = int(
        service.number_setting(args, "llm_judge_max_concurrent_calls", "OI_MILES_JUDGE_MAX_CONCURRENT_CALLS", 256)
    )
    if max_tokens < 1 or max_context_length < 1 or timeout <= 0 or max_concurrent_calls < 1:
        raise ValueError("general-judge token limits, timeout, and concurrency must be positive")
    return GeneralJudgeConfig(
        api_url=api_url,
        api_key=os.environ.get("OI_MILES_JUDGE_API_KEY", os.environ.get("HOSTED_VLLM_API_KEY", "EMPTY")),
        model=model,
        max_tokens=max_tokens,
        max_context_length=max_context_length,
        temperature=temperature,
        timeout=timeout,
        seed=seed,
        max_concurrent_calls=max_concurrent_calls,
        check_context=os.environ.get("OI_MILES_JUDGE_CHECK_CONTEXT", "false").lower() == "true",
    )


_MARKDOWN_SCORE_PATTERN = r"^\s*\*{0,2}SCORE\*{0,2}\s*:\*{0,2}\s*([0-9]+(?:\.[0-9]+)?)\s*$"
# Used for incomplete objects, where the keys cannot be checked after JSON decoding.
_JSON_SCORE_FIELD = r'"(?:S|\\u0053)(?:C|\\u0043)(?:O|\\u004[fF])(?:R|\\u0052)(?:E|\\u0045)"\s*:'


def _decode_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode a JSON object while rejecting duplicate SCORE keys."""
    if sum(1 for key, _ in pairs if key == "SCORE") > 1:
        raise JudgeResponseError("general judge returned duplicate SCORE keys")
    return dict(pairs)


def _loads_judge_json(text: str) -> Any:
    """Decode judge JSON, allowing unescaped control characters and stray backslashes."""
    try:
        return json.loads(text, strict=False, object_pairs_hook=_decode_json_object)
    except json.JSONDecodeError:
        escaped = re.sub(
            r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|\\',
            lambda match: match.group(0) if match.group(1) is not None else r"\\",
            text,
        )
        return json.loads(escaped, strict=False, object_pairs_hook=_decode_json_object)


def _contains_json_score(text: str) -> bool:
    """Find a competing top-level score among JSON objects embedded in a reply."""
    decoder = json.JSONDecoder(strict=False, object_pairs_hook=_decode_json_object)
    opening_delimiter = re.compile(r"[{\[]")
    offset = 0
    while (opening := opening_delimiter.search(text, offset)) is not None:
        start = opening.start()
        try:
            data, offset = decoder.raw_decode(text, start)
        except (json.JSONDecodeError, ValueError):
            # An incomplete object may still contain a competing grade.
            return re.search(_JSON_SCORE_FIELD, text[start:]) is not None
        if isinstance(data, dict) and "SCORE" in data:
            return True
    return False


def _normalize_score_value(score_value: Any) -> float:
    """Validate a raw 0..10 judge score and normalize it to [0.0, 1.0]."""
    if isinstance(score_value, bool):
        raise JudgeResponseError("general judge returned a boolean score")
    try:
        score = float(score_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise JudgeResponseError(f"general judge returned non-numeric SCORE {score_value!r}") from error
    if not math.isfinite(score) or not 0 <= score <= 10:
        raise JudgeResponseError(f"general judge returned out-of-range SCORE {score!r}")
    return score / 10.0


def _extract_json_score(data: Any) -> tuple[str, float]:
    """Validate and normalize a decoded judge JSON object."""
    if not isinstance(data, dict) or "SCORE" not in data:
        raise JudgeResponseError("general judge JSON response must be an object with SCORE")
    return str(data.get("REASONING", "")), _normalize_score_value(data["SCORE"])


def _repair_json_separators(text: str) -> str:
    text = re.sub(r'，(?=\s*")', ",", text)
    return re.sub(r",\s*\}\s*$", "}", text)


def _has_unclosed_inner_container(text: str) -> bool:
    """Detect JSON containers that could contain the proposed score."""
    raw_brace_depth = 0
    for brace in re.findall(r"[{}]", text):
        if brace == "{":
            raw_brace_depth += 1
        elif raw_brace_depth:
            raw_brace_depth -= 1
    if raw_brace_depth:
        return True

    # A brace in prose can precede an unmatched quote without opening a JSON object.
    object_start = re.compile(r'\{\s*(?:"(?:[^"\\]|\\.)*"\s*:|\})', flags=re.DOTALL)
    closing_delimiters = {"{": "}", "[": "]"}
    expected_closers = []
    for match in re.finditer(r'"(?:[^"\\]|\\.)*"|[{}\[\]]', text, flags=re.DOTALL):
        token = match.group()
        if token == "{" and object_start.match(text, match.start()) is None:
            continue
        if token in closing_delimiters:
            expected_closers.append(closing_delimiters[token])
        elif expected_closers and token == expected_closers[-1]:
            expected_closers.pop()
    return bool(expected_closers)


def _recover_final_json_score(text: str) -> float | None:
    """Recover an unambiguous top-level final SCORE field from malformed judge JSON."""
    brace_idx = text.find("{")
    if brace_idx == -1:
        return None
    candidate = text[brace_idx:].strip()

    # 1. Trailing comma right before '}' — decode structurally so duplicate-SCORE and type checks run.
    without_trailing_comma = re.sub(r",\s*\}\s*$", "}", candidate)
    if without_trailing_comma != candidate:
        try:
            data = _loads_judge_json(without_trailing_comma)
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            return _extract_json_score(data)[1]

    # 2. Separate final {"SCORE": ...} block after an earlier JSON object (or competing objects).
    try:
        first_obj, end_idx = json.JSONDecoder(strict=False, object_pairs_hook=_decode_json_object).raw_decode(
            candidate
        )
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        trailing = candidate[end_idx:].strip()
        if isinstance(first_obj, dict) and trailing:
            if "SCORE" in first_obj and _contains_json_score(trailing):
                raise JudgeResponseError("general judge response contains competing JSON objects")
            if "SCORE" not in first_obj and trailing.startswith("{"):
                trailing_repaired = re.sub(r",\s*\}\s*$", "}", trailing)
                try:
                    final_obj = _loads_judge_json(trailing_repaired)
                except (json.JSONDecodeError, ValueError):
                    return None
                return _extract_json_score(final_obj)[1]

    # 3. Single top-level object whose leading string field (e.g. REASONING) has syntax errors
    #    (unescaped quotes) or is followed by a missing/full-width comma.
    opening = re.match(r'^\{\s*"(?P<key>[^"\n]+)"\s*:\s*"', candidate)
    if opening is None or re.fullmatch(_JSON_SCORE_FIELD, f'"{opening.group("key")}":'):
        return None

    tail_start = opening.end()
    tail = candidate[tail_start:]
    locked_split: tuple[int, int] | None = None
    try:
        _, str_end = json.JSONDecoder(strict=False).raw_decode(candidate, tail_start - 1)
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        after_str = candidate[str_end:]
        sep_match = re.match(r'^\s*(?:[,，]\s*|\s+)(?="[^"\n]+"\s*:)', after_str)
        if sep_match is not None:
            locked_split = (str_end - tail_start, str_end - tail_start + sep_match.end())

    split_candidates = (
        [locked_split]
        if locked_split is not None
        else [
            (match.start(), match.end())
            for match in re.finditer(r'(?:"\s*[,，]\s*|[,，]\s*\n\s*|"\s+)(?="[^"\n]+"\s*:)', tail)
        ]
    )

    for sep_start, key_start in split_candidates:
        before_split = tail[:sep_start]
        # Some separators include the reasoning's closing quote; keep it when scanning strings.
        prefix_end = sep_start + (tail[sep_start] == '"')
        if _has_unclosed_inner_container(tail[:prefix_end]):
            continue
        if re.search(r'(?:"\s*[,，]\s*|[,，]\s*\n\s*|"\s+)' + _JSON_SCORE_FIELD, before_split):
            raise JudgeResponseError("general judge returned duplicate SCORE keys")
        suffix = _repair_json_separators(tail[key_start:])
        repaired = '{"REASONING": "", ' + suffix
        try:
            data = _loads_judge_json(repaired)
        except (json.JSONDecodeError, ValueError):
            continue
        return _extract_json_score(data)[1]

    return None


def _parse_json_candidate(candidate: str, fallback_reasoning: str) -> tuple[str, float] | None:
    """Decode a JSON judge verdict, falling back to an anchored final SCORE if JSON is malformed."""
    try:
        data = _loads_judge_json(candidate)
    except json.JSONDecodeError:
        recovered = _recover_final_json_score(candidate)
        return (fallback_reasoning, recovered) if recovered is not None else None
    except ValueError as error:
        raise JudgeResponseError(f"general judge returned invalid JSON value: {error}") from error
    return _extract_json_score(data)


def _strip_outer_fence(text: str) -> str | None:
    """If the entire text is wrapped in a single code fence, return its inner body."""
    match = re.match(r"^(`{3,})(?:json(?=\s|\{|$))?", text, flags=re.IGNORECASE)
    if match is None:
        return None
    ticks = match.group(1)
    if not text.endswith(ticks) or len(text) < match.end() + len(ticks):
        return None
    return text[match.end() : -len(ticks)].strip()


def _scan_fenced_blocks(text: str) -> tuple[list[tuple[str, str]], str, list[str]]:
    """Split text into closed (lang, body) fenced blocks, non-fenced text, and unclosed fence langs."""
    blocks: list[tuple[str, str]] = []
    outside_lines: list[str] = []
    unclosed_langs: list[str] = []
    in_fence = False
    fence_len = 0
    fence_lang = ""
    fence_lines: list[str] = []

    for line in text.split("\n"):
        if not in_fence:
            opening = re.match(r"^\s*(`{3,})(.*)$", line)
            if opening is not None:
                ticks, rest = opening.group(1), opening.group(2).strip()
                if rest.endswith(ticks) and len(rest) >= len(ticks):
                    body = rest[: -len(ticks)]
                    if body[:4].lower() == "json" and (len(body) == 4 or body[4].isspace() or body[4] == "{"):
                        blocks.append(("json", body[4:].strip()))
                    else:
                        blocks.append(("", body.strip()))
                    continue
                in_fence = True
                fence_len = len(ticks)
                fence_lang = rest.split(None, 1)[0].lower() if rest else ""
                fence_lines = []
                continue
            outside_lines.append(line)
        else:
            if re.match(rf"^\s*`{{{fence_len},}}\s*$", line) is not None:
                blocks.append((fence_lang, "\n".join(fence_lines).strip()))
                in_fence = False
                fence_len = 0
                fence_lang = ""
                fence_lines = []
            else:
                fence_lines.append(line)

    if in_fence:
        body = "\n".join(fence_lines).strip()
        unclosed_langs.append("json" if fence_lang == "json" or body.startswith("{") else fence_lang)
    return blocks, "\n".join(outside_lines), unclosed_langs


def parse_judge_response(content: str) -> tuple[str, float]:
    """Parse and normalize Open-Instruct's JSON-or-SCORE judge response.

    Tuned for the default ``Qwen/Qwen3-32B`` judge under the ``general_quality`` /
    ``general_quality_ref`` JSON templates: strips ``<think>``/``<answer>`` wrappers,
    decodes single- or multi-line JSON directly (with ``strict=False`` and stray-backslash
    recovery), recovers an unambiguous final ``"SCORE"`` field from malformed JSON (such as
    unescaped quotes in ``REASONING``, missing or full-width commas, trailing commas, or a
    separate final ``{"SCORE": ...}`` block), and falls back to a standalone Markdown
    ``SCORE:`` line for non-JSON replies. Ambiguous or competing scores, duplicate
    ``SCORE`` keys, and invalid/out-of-range values are rejected so ``general_judge_score``
    can retry or exclude the sample. Re-check these assumptions when changing the judge
    model or rubric prompt.
    """
    try:
        return _parse_judge_response(content)
    except RecursionError as error:
        raise JudgeResponseError("general judge returned JSON nested too deeply") from error


def _parse_judge_response(content: str) -> tuple[str, float]:
    content = re.sub(r"<think>\s*.*?\s*</think>\s*", "", content, flags=re.DOTALL)
    content = content.replace("<answer>", "").replace("</answer>", "")
    cleaned = content.replace("\r\n", "\n").strip()

    outer_body = _strip_outer_fence(cleaned)
    direct_candidate = outer_body if outer_body is not None else cleaned
    try:
        data = _loads_judge_json(direct_candidate)
    except json.JSONDecodeError:
        pass
    except ValueError as error:
        raise JudgeResponseError(f"general judge returned invalid JSON value: {error}") from error
    else:
        return _extract_json_score(data)

    blocks, outside_text, unclosed_langs = _scan_fenced_blocks(cleaned)
    if "json" in unclosed_langs:
        raise JudgeResponseError("general judge returned unclosed JSON fence")
    json_blocks = [body for lang, body in blocks if lang == "json" or body.startswith("{")]
    if json_blocks:
        if _contains_json_score(outside_text):
            raise JudgeResponseError("general judge response contains competing JSON objects")
        if any(_contains_json_score(block) for block in json_blocks[:-1]):
            raise JudgeResponseError("general judge returned multiple JSON blocks with SCORE")
        parsed = _parse_json_candidate(json_blocks[-1], cleaned)
        if parsed is None:
            raise JudgeResponseError("general judge returned malformed JSON")
        reasoning, normalized_score = parsed
        outside_scores = re.findall(_MARKDOWN_SCORE_PATTERN, outside_text, flags=re.MULTILINE)
        if any(not math.isfinite(float(s)) or float(s) / 10.0 != normalized_score for s in outside_scores):
            raise JudgeResponseError("general judge response contains conflicting JSON and Markdown scores")
        return reasoning, normalized_score

    recovered = _recover_final_json_score(direct_candidate)
    if recovered is not None:
        preamble = direct_candidate[: direct_candidate.find("{")] if "{" in direct_candidate else ""
        preamble_scores = re.findall(_MARKDOWN_SCORE_PATTERN, preamble, flags=re.MULTILINE)
        if any(not math.isfinite(float(s)) or float(s) / 10.0 != recovered for s in preamble_scores):
            raise JudgeResponseError("general judge response contains conflicting JSON and Markdown scores")
        return cleaned, recovered
    if (
        direct_candidate.startswith("{")
        or _contains_json_score(outside_text)
        or re.search(r'\{[\s\S]*?"REASONING"\s*:', outside_text)
    ):
        raise JudgeResponseError("general judge returned malformed JSON")

    # Accept a single explicit Markdown score line for non-JSON replies, strictly outside code fences
    # (or inside a sole unlabeled fence containing only the score line).
    if not outside_text.strip() and not unclosed_langs and len(blocks) == 1 and blocks[0][0] == "":
        sole_score = re.fullmatch(_MARKDOWN_SCORE_PATTERN, blocks[0][1])
        scores = [sole_score.group(1)] if sole_score is not None else []
    else:
        scores = re.findall(_MARKDOWN_SCORE_PATTERN, outside_text, flags=re.MULTILINE)
    if len(scores) != 1:
        raise JudgeResponseError("general judge response has no unambiguous parseable SCORE")
    return cleaned, _normalize_score_value(scores[0])


def parse_web_instruct_judge_response(content: str) -> tuple[str, float]:
    """Parse Open-Instruct's binary WebInstruct judge response."""
    cleaned = re.sub(r"<think>\s*.*?\s*</think>\s*", "", content, flags=re.DOTALL)
    cleaned = cleaned.replace("<answer>", "").replace("</answer>", "")
    normalized = cleaned.lower()
    decisions = set(re.findall(r"final decision:\s*(yes|no)\b", normalized))
    if len(decisions) != 1:
        raise JudgeResponseError("binary judge response must contain one unambiguous Final Decision")
    return cleaned, float(decisions == {"yes"})


def build_judge_prompt(
    name: str, *, query: str, prediction: str, target: Any, extract_answer: bool = True
) -> tuple[str, str]:
    """Build an oracle-validated general-quality judge prompt.

    ``extract_answer`` preserves the historical direct-call/flag-off contract.
    Shared grading views must pass False: a quality judge should see the chosen
    text intact, rather than implicitly accepting an answer-wrapper convention.
    """
    try:
        template, expected_sha256 = _PROMPTS[name]
    except KeyError as error:
        raise ValueError(f"unknown general judge verifier {name!r}") from error
    actual_sha256 = hashlib.sha256(template.encode()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"general judge prompt drift for {name}: {actual_sha256} != {expected_sha256}")
    output = extract_final_answer(prediction) if extract_answer else prediction
    return template.format(input=query, output=output, label=target), actual_sha256


def _get_session() -> Any:
    return service.http_session(pool_size=256)


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    executor = _EXECUTORS.get(max_workers)
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="miles-general-judge")
        _EXECUTORS[max_workers] = executor
    return executor


def count_prompt_tokens(config: GeneralJudgeConfig, prompt: str) -> int:
    """Count the complete chat request with the serving tokenizer/template."""
    response = infra_timeouts.request(
        _get_session(),
        "post",
        config.api_url.removesuffix("/v1/chat/completions") + "/tokenize",
        json={"model": config.model, "messages": [{"role": "user", "content": prompt}], "add_generation_prompt": True},
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=config.timeout,
    )
    response.raise_for_status()
    count = response.json()["count"]
    if type(count) is not int or count < 1:
        raise RuntimeError("judge tokenizer returned an invalid count")
    return count


class JudgeContextOverflow(RuntimeError):
    """The unchanged request cannot fit; retrying cannot recover."""


class JudgeResponseError(RuntimeError):
    """Keep bounded response evidence without accepting an invalid grade."""

    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _request(config: GeneralJudgeConfig, prompt: str) -> str:
    try:
        if config.check_context:
            count = count_prompt_tokens(config, prompt)
            if count + config.max_tokens > config.max_context_length:
                raise JudgeContextOverflow(
                    f"judge context overflow: {count} prompt + {config.max_tokens} output > "
                    f"{config.max_context_length}; request was not truncated"
                )
        response = infra_timeouts.request(
            _get_session(),
            "post",
            config.api_url,
            json={
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "seed": config.seed,
            },
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            timeout=config.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        if choice.get("finish_reason") != "stop":
            raw = choice.get("message", {}).get("content")
            raise JudgeResponseError(
                "judge response did not complete normally",
                {
                    "finish_reason": choice.get("finish_reason"),
                    "max_tokens": config.max_tokens,
                    "raw_reply": raw[:65536] if isinstance(raw, str) else None,
                    "raw_reply_clipped": isinstance(raw, str) and len(raw) > 65536,
                    "usage": {
                        k: v
                        for k, v in (payload.get("usage") or {}).items()
                        if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                    },
                },
            )
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("response content is not a string")
        return content
    except (JudgeResponseError, JudgeContextOverflow):
        raise
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        raise JudgeResponseError(
            f"general judge request failed (transport or response): {type(error).__name__}: {error}"
        ) from error
    except Exception as error:
        raise RuntimeError(f"general judge request failed for {config.api_url}: {error}") from error


async def general_judge_score(
    args: Any, sample: Any, *, name: str, target: Any, prediction: str | None = None, extract_answer: bool = True
) -> float:
    """Score one response with the configured OpenAI-compatible general judge."""
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("general-quality samples require dictionary metadata")
    query = metadata.get("judge_query")
    if not isinstance(query, str) or not query:
        raise ValueError("general-quality samples require metadata.judge_query")
    # Keep the original sample for training/records while grading the selected answer.
    prediction = str(sample.response) if prediction is None else prediction
    prompt_options = {} if extract_answer else {"extract_answer": False}
    resolved = judge_registry.resolve_request(args, name, query, prediction, target, **prompt_options)
    binding = None
    rubric_sha256 = None
    if resolved:
        config, prompt, prompt_sha256, parser, binding, rubric_sha256 = resolved
    else:
        prompt, prompt_sha256 = build_judge_prompt(
            name, query=query, prediction=prediction, target=target, **prompt_options
        )
        config = general_judge_config(args)
        parser = "yes-no" if name == "general-web_instruct_general_verifier" else "score-1-to-10"
    policy = getattr(args, "llm_judge_failure_policy", None) or os.environ.get("OI_MILES_JUDGE_FAILURE_POLICY", "zero")
    if policy not in {"zero", "raise"}:
        raise ValueError("OI_MILES_JUDGE_FAILURE_POLICY must be 'zero' or 'raise'")
    # Explicit abstention makes no judge request, so it is deliberately excluded
    # from general_judge request/error metrics. rubric_profile links both paths.
    if parser == abstention.KIND and abstention.is_abstention(prediction):
        metadata.setdefault("verifier_diagnostics", {})[name] = {
            "kind": abstention.KIND,
            "status": "ok",
            "verdict": "abstain",
            "score": 0.5,
            "judge_called": False,
            "binding": binding,
            "rubric_sha256": rubric_sha256,
            "rubric_profile": abstention.PROFILE,
        }
        return 0.5
    started = time.monotonic()
    last_error: RuntimeError | None = None
    failed_attempts = []
    for attempt in range(3):
        content = None
        try:
            executor = _get_executor(config.max_concurrent_calls)
            content = await asyncio.get_running_loop().run_in_executor(executor, _request, config, prompt)
            if parser == "yes-no":
                reasoning, score = parse_web_instruct_judge_response(content)
            elif parser == abstention.KIND:
                try:
                    reasoning, score = abstention.grade(content)
                except ValueError as error:
                    raise JudgeResponseError(str(error)) from error
            else:
                reasoning, score = parse_judge_response(content)
            break
        except RuntimeError as error:
            last_error = error
            evidence = {"attempt": attempt + 1, "error": str(error), **getattr(error, "diagnostics", {})}
            if isinstance(content, str):
                evidence["raw_reply"] = content[:65536]
            failed_attempts.append(evidence)
            metadata.setdefault("verifier_diagnostics", {})[name] = {
                "binding": binding,
                "rubric_sha256": rubric_sha256,
                "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "model": config.model,
                "seed": config.seed,
                "temperature": config.temperature,
                "elapsed_seconds": time.monotonic() - started,
                "verdict": "failed",
                "kind": "general_judge",
                **({"rubric_profile": abstention.PROFILE, "judge_called": True} if parser == abstention.KIND else {}),
                "failure_policy": policy,
                "attempts": list(failed_attempts),
            }
            if attempt == 2 or isinstance(error, JudgeContextOverflow):
                diagnostics = metadata["verifier_diagnostics"][name]
                if policy == "zero" and isinstance(error, JudgeResponseError):
                    diagnostics.update(status="judge_error", fallback_reward=0.0)
                    logger.warning(
                        "General judge failed; assigning zero reward and continuing: %s", json.dumps(diagnostics)
                    )
                    return 0.0
                logger.error("General judge failed: %s", json.dumps(diagnostics))
                raise
            await asyncio.sleep(2**attempt)
    else:
        assert last_error is not None
        raise last_error
    if not math.isfinite(score):
        raise RuntimeError(f"general judge returned non-finite score {score!r}")
    metadata.setdefault("verifier_diagnostics", {})[name] = {
        "kind": "general_judge",
        "status": "ok",
        "binding": binding,
        "rubric_sha256": rubric_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "max_context_length": config.max_context_length,
        "context_checked": config.check_context,
        "max_concurrent_calls": config.max_concurrent_calls,
        "max_tokens": config.max_tokens,
        "model": config.model,
        "prompt_sha256": prompt_sha256,
        "reasoning": reasoning,
        "failed_attempts": failed_attempts,
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw_reply": content if binding is not None else None,
        "seed": config.seed,
        "temperature": config.temperature,
    }
    if parser == abstention.KIND:
        metadata["verifier_diagnostics"][name].update(
            rubric_profile=abstention.PROFILE, verdict=reasoning, judge_called=True, score=score
        )
    return score
