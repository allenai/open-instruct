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

"""Strict judge rewards ported from olmo-miles afbdd6f (rl/general_judge.py)."""

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

from open_instruct import logger_utils
from open_instruct.miles import judge_registry

logger = logger_utils.setup_logger(__name__)

_GENERAL_QUALITY_TEMPLATE = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.

[Conversation History]
{input}

[AI Answer]
{output}

[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""

_GENERAL_QUALITY_REF_TEMPLATE = """
### Task Description
Please act as an impartial judge and evaluate the quality of the answer provided by an
AI assistant to the conversation history leading up to the answer displayed below.
Judge whether the provided answer is good by comparing it to the reference answer.

Notes:
- Besides comparing to the reference answer, your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable and can get a full score.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.

[Conversation History]
{input}

[AI Answer]
{output}

[Reference Gold Answer]
{label}

[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""

_WEB_INSTRUCT_GENERAL_VERIFIER_TEMPLATE = """User: ### Question: {input}


### Ground Truth Answer: {label}


### Student Answer: {output}


For the above question, please verify if the student's answer is equivalent to the ground truth answer.
Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.
If the student's answer is correct, output "Final Decision: Yes". If the student's answer is incorrect, output Final Decision: No. Assistant:"""

_PROMPTS = {
    "general-quality": (_GENERAL_QUALITY_TEMPLATE, "dc5690758431cfeb5d5eb308540eddb728a581ef2c341a73ca12fb4beaa6e6b1"),
    "general-quality_ref": (
        _GENERAL_QUALITY_REF_TEMPLATE,
        "33907cb811254249db7fd78683ebb7c2cc194c9e04dc78472dde0663b0762d4e",
    ),
    "general-web_instruct_general_verifier": (
        _WEB_INSTRUCT_GENERAL_VERIFIER_TEMPLATE,
        "6d14cbfb0909d57f3f607d3dc92f9f35ee98219434f62b3ab7346c95dc29b098",
    ),
}
_SESSION: Any = None
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


def _number_setting(args: Any, attribute: str, environment: str, default: int | float) -> int | float:
    value = getattr(args, attribute, None)
    if value is None:
        value = os.environ.get(environment, default)
    expected_type = int if isinstance(default, int) else float
    try:
        parsed = expected_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{environment} must be a finite number") from error
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise ValueError(f"{environment} must be a finite number")
    return parsed


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
    max_tokens = int(_number_setting(args, "llm_judge_max_tokens", "OI_MILES_JUDGE_MAX_TOKENS", 2048))
    max_context_length = int(
        _number_setting(args, "llm_judge_max_context_length", "OI_MILES_JUDGE_MAX_CONTEXT_LENGTH", 32768)
    )
    temperature = float(_number_setting(args, "llm_judge_temperature", "OI_MILES_JUDGE_TEMPERATURE", 1.0))
    timeout = float(_number_setting(args, "llm_judge_timeout", "OI_MILES_JUDGE_TIMEOUT", 600.0))
    seed = int(_number_setting(args, "seed", "OI_MILES_JUDGE_SEED", 1))
    max_concurrent_calls = int(
        _number_setting(args, "llm_judge_max_concurrent_calls", "OI_MILES_JUDGE_MAX_CONCURRENT_CALLS", 256)
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


def extract_final_answer(prediction: str) -> str:
    """Extract the answer surface using Open-Instruct's precedence rules."""
    answer_match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    think_match = re.search(r"</think>(.*)", prediction, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    cleaned = re.sub(r"<\|assistant\|>", "", prediction)
    return cleaned.strip() if cleaned != prediction else prediction


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
            raise RuntimeError("general judge returned a boolean score")
        score = float(score_value)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError):
        match = re.search(r'"SCORE"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', cleaned)
        if match is None:
            raise RuntimeError("general judge response has no parseable SCORE") from None
        reasoning = cleaned
        score = float(match.group(1))
    if not math.isfinite(score) or not 0 <= score <= 10:
        raise RuntimeError(f"general judge returned out-of-range SCORE {score!r}")
    return reasoning, score / 10.0


def parse_web_instruct_judge_response(content: str) -> tuple[str, float]:
    """Parse Open-Instruct's binary WebInstruct judge response."""
    cleaned = re.sub(r"<think>\s*.*?\s*</think>\s*", "", content, flags=re.DOTALL)
    cleaned = cleaned.replace("<answer>", "").replace("</answer>", "")
    normalized = cleaned.lower()
    decisions = set(re.findall(r"final decision:\s*(yes|no)\b", normalized))
    if len(decisions) != 1:
        raise RuntimeError("binary judge response must contain one unambiguous Final Decision")
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
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _SESSION = session
    return session


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    executor = _EXECUTORS.get(max_workers)
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="olmo-miles-general-judge")
        _EXECUTORS[max_workers] = executor
    return executor


def count_prompt_tokens(config: GeneralJudgeConfig, prompt: str) -> int:
    """Count the complete chat request with the serving tokenizer/template."""
    response = _get_session().post(
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
                raise RuntimeError(
                    f"judge context overflow: {count} prompt + {config.max_tokens} output > "
                    f"{config.max_context_length}; request was not truncated"
                )
        response = _get_session().post(
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
    except JudgeResponseError:
        raise
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
            if isinstance(content, str) and not isinstance(error, JudgeResponseError):
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
                "attempts": list(failed_attempts),
            }
            if attempt == 2:
                logger.error("General judge failed: %s", json.dumps(metadata["verifier_diagnostics"][name]))
                raise
            await asyncio.sleep(2**attempt)
    else:
        assert last_error is not None
        raise last_error
    if not math.isfinite(score):
        raise RuntimeError(f"general judge returned non-finite score {score!r}")
    metadata.setdefault("verifier_diagnostics", {})[name] = {
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
