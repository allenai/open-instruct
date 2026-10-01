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

"""Judge rewards ported from the Megatron implementation."""

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
from open_instruct.miles.rewards import judge_registry, service

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
    timeout = float(service.number_setting(args, "llm_judge_timeout", "OI_MILES_JUDGE_TIMEOUT", 600.0))
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


def parse_judge_response(content: str) -> tuple[str, float]:
    """Parse and normalize Open-Instruct's JSON-or-SCORE judge response."""
    content = re.sub(r"<think>\s*.*?\s*</think>\s*", "", content, flags=re.DOTALL)
    content = content.replace("<answer>", "").replace("</answer>", "")
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.replace("\r\n", "\n").replace("\n", "\\n")
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned).strip()
    try:
        data = json.loads(cleaned)
        reasoning = str(data.get("REASONING", ""))
        score_value = data["SCORE"]
        if isinstance(score_value, bool):
            raise JudgeResponseError("general judge returned a boolean score")
        score = float(score_value)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError):
        match = re.search(r'"SCORE"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', cleaned)
        if match is None:
            # Accept a single explicit Markdown score line, not numbers in prose.
            scores = re.findall(
                r"^\s*\*{0,2}SCORE\*{0,2}\s*:\*{0,2}\s*([0-9]+(?:\.[0-9]+)?)\s*$", content, flags=re.MULTILINE
            )
            if len(scores) != 1:
                raise JudgeResponseError("general judge response has no unambiguous parseable SCORE") from None
            score = float(scores[0])
        else:
            score = float(match.group(1))
        reasoning = cleaned
    if not math.isfinite(score) or not 0 <= score <= 10:
        raise JudgeResponseError(f"general judge returned out-of-range SCORE {score!r}")
    return reasoning, score / 10.0


def parse_web_instruct_judge_response(content: str) -> tuple[str, float]:
    """Parse Open-Instruct's binary WebInstruct judge response."""
    cleaned = re.sub(r"<think>\s*.*?\s*</think>\s*", "", content, flags=re.DOTALL)
    cleaned = cleaned.replace("<answer>", "").replace("</answer>", "")
    normalized = cleaned.lower()
    decisions = set(re.findall(r"final decision:\s*(yes|no)\b", normalized))
    if len(decisions) != 1:
        raise JudgeResponseError("binary judge response must contain one unambiguous Final Decision")
    return cleaned, float(decisions == {"yes"})


def build_judge_prompt(name: str, *, query: str, prediction: str, target: Any) -> tuple[str, str]:
    """Build an oracle-validated general-quality judge prompt."""
    try:
        template, expected_sha256 = _PROMPTS[name]
    except KeyError as error:
        raise ValueError(f"unknown general judge verifier {name!r}") from error
    actual_sha256 = hashlib.sha256(template.encode()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"general judge prompt drift for {name}: {actual_sha256} != {expected_sha256}")
    return template.format(input=query, output=extract_final_answer(prediction), label=target), actual_sha256


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


async def general_judge_score(args: Any, sample: Any, *, name: str, target: Any) -> float:
    """Score one response with the configured OpenAI-compatible general judge."""
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("general-quality samples require dictionary metadata")
    query = metadata.get("judge_query")
    if not isinstance(query, str) or not query:
        raise ValueError("general-quality samples require metadata.judge_query")
    resolved = judge_registry.resolve_request(args, name, query, str(sample.response), target)
    binding = None
    rubric_sha256 = None
    if resolved:
        config, prompt, prompt_sha256, parser, binding, rubric_sha256 = resolved
    else:
        prompt, prompt_sha256 = build_judge_prompt(name, query=query, prediction=str(sample.response), target=target)
        config = general_judge_config(args)
        parser = "yes-no" if name == "general-web_instruct_general_verifier" else "score-1-to-10"
    policy = getattr(args, "llm_judge_failure_policy", None) or os.environ.get("OI_MILES_JUDGE_FAILURE_POLICY", "zero")
    if policy not in {"zero", "raise"}:
        raise ValueError("OI_MILES_JUDGE_FAILURE_POLICY must be 'zero' or 'raise'")
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
    return score
