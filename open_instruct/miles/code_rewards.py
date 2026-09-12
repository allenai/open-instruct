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

"""Open-Instruct-compatible external code-execution rewards."""

from __future__ import annotations

import asyncio
import math
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import requests

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?(.*?)```", re.DOTALL)
_SESSION: Any = None


@dataclass(frozen=True, kw_only=True)
class CodeVerifierConfig:
    """Validated execution-service settings for one code verifier call."""

    api_url: str
    max_execution_time: float
    pass_rate_reward_threshold: float
    apply_perf_penalty: bool


def extract_python_code(model_output: str) -> str:
    """Return the last fenced Python block, or the complete response."""
    matches = _CODE_BLOCK_PATTERN.findall(model_output)
    return matches[-1].strip() if matches else model_output


def _float_setting(args: Any, attribute: str, environment: str, default: float) -> float:
    value = getattr(args, attribute, None)
    if value is None:
        value = os.environ.get(environment, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{environment} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{environment} must be a finite number")
    return parsed


def _bool_setting(args: Any, attribute: str, environment: str, default: bool) -> bool:
    value = getattr(args, attribute, None)
    if value is None:
        value = os.environ.get(environment, str(default))
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{environment} must be a boolean")


def code_verifier_config(args: Any, *, stdio: bool = False) -> CodeVerifierConfig:
    """Resolve code verifier settings from compatible args and environment."""
    url = getattr(args, "code_api_url", None) or os.environ.get("OI_MILES_CODE_API_URL")
    if not url:
        base = os.environ.get("CODE_API_URL", "http://localhost:1234").rstrip("/")
        url = f"{base}/test_program"
    if stdio:
        if not str(url).endswith("/test_program"):
            raise ValueError("code_stdio requires a code API URL ending in /test_program")
        url = str(url)[: -len("/test_program")] + "/test_program_stdio"
    max_execution_time = _float_setting(args, "code_max_execution_time", "OI_MILES_CODE_MAX_EXECUTION_TIME", 1.0)
    threshold = _float_setting(
        args, "code_pass_rate_reward_threshold", "OI_MILES_CODE_PASS_RATE_REWARD_THRESHOLD", 0.0
    )
    if max_execution_time <= 0:
        raise ValueError("OI_MILES_CODE_MAX_EXECUTION_TIME must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("OI_MILES_CODE_PASS_RATE_REWARD_THRESHOLD must be between zero and one")
    return CodeVerifierConfig(
        api_url=str(url),
        max_execution_time=max_execution_time,
        pass_rate_reward_threshold=threshold,
        apply_perf_penalty=_bool_setting(args, "code_apply_perf_penalty", "OI_MILES_CODE_APPLY_PERF_PENALTY", False),
    )


# The code service is an external API gateway that returns transient 5xx errors
# under load. urllib3 retries only idempotent methods by default, so the scoring
# POSTs must be allowed explicitly; the backoff grows 1, 2, 4, ... seconds and is
# capped by urllib3, about four minutes in total before the verifier gives up.
RETRY = requests.adapters.Retry(
    total=8,
    backoff_factor=1.0,
    status_forcelist=[502, 503, 504],
    allowed_methods=frozenset({"GET", "POST"}),
    raise_on_status=False,
)


def _get_session() -> Any:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=RETRY)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _SESSION = session
    return session


def _score_response(result: Any, *, config: CodeVerifierConfig) -> float:
    if not isinstance(result, dict):
        raise RuntimeError("code service response must be an object")
    passes = result.get("results")
    if not isinstance(passes, list):
        raise RuntimeError("code service response requires a results list")
    if any(isinstance(value, bool) is False and not isinstance(value, int | float) for value in passes):
        raise RuntimeError("code service results must be numeric")
    pass_rate = sum(float(value) for value in passes) / len(passes) if passes else 0.0
    score = 0.0 if pass_rate < config.pass_rate_reward_threshold else pass_rate
    if not config.apply_perf_penalty or score <= 0:
        return score
    runtimes = result.get("runtimes")
    if not isinstance(runtimes, list) or len(runtimes) != len(passes):
        raise RuntimeError("performance-penalized code rewards require aligned runtimes")
    penalized = [
        float(passed) * (config.max_execution_time - float(runtime)) / config.max_execution_time
        for passed, runtime in zip(passes, runtimes, strict=True)
    ]
    return sum(penalized) / len(penalized) if penalized else 0.0


async def code_score(args: Any, prediction: str, target: Any, *, stdio: bool = False) -> float:
    """Execute one completion against its tests and return its pass-rate reward."""
    config = code_verifier_config(args, stdio=stdio)
    payload = {
        "program": extract_python_code(prediction),
        "tests": target,
        "max_execution_time": config.max_execution_time,
    }
    timeout = max(30.0, min(300.0, config.max_execution_time * 10))

    def request() -> Any:
        try:
            response = _get_session().post(
                config.api_url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            status = getattr(error.response, "status_code", None)
            if status is not None and (400 <= status < 500 and status != 429 or status == 500):
                # A client error is a property of this sample (an oversized test
                # payload, for instance), and the code service answers 500 when
                # executing a program raises inside its harness; neither is a
                # service outage. Score the sample zero, as the standard
                # open-instruct verifier does, and keep training. Gateway errors
                # (502-504) and 429 are retried and then fail the run.
                logger.warning(
                    "code verifier rejected a sample with HTTP %s (program %d chars, %d tests); scoring it zero",
                    status,
                    len(payload["program"]),
                    len(target) if isinstance(target, list) else -1,
                )
                return {"results": []}
            raise RuntimeError(f"code verifier request failed for {config.api_url}: {error}") from error
        except Exception as error:
            raise RuntimeError(f"code verifier request failed for {config.api_url}: {error}") from error

    result = await asyncio.to_thread(request)
    score = _score_response(result, config=config)
    if not math.isfinite(score):
        raise RuntimeError(f"code verifier returned non-finite score {score!r}")
    return score


@dataclass
class ServiceConfig:
    api_url: str
    stdio: bool = False
    max_execution_time: float = 1.0
    pass_rate_reward_threshold: float = 0.99


class CodeVerifier:
    get_config_class = staticmethod(lambda: ServiceConfig)

    def __init__(self, verifier_config):
        self.config = verifier_config

    async def async_call(self, tokens, prediction, label, **kwargs):
        args = SimpleNamespace(
            code_api_url=self.config.api_url,
            code_max_execution_time=self.config.max_execution_time,
            code_pass_rate_reward_threshold=self.config.pass_rate_reward_threshold,
        )
        return SimpleNamespace(score=await code_score(args, prediction, label, stdio=self.config.stdio), cost=0.0)
