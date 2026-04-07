"""BounceSim (ballsim) verifier (HTTP API)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from open_instruct import logger_utils
from open_instruct.ground_truth_registry import register_verifier, register_verifier_config
from open_instruct.ground_truth_utils import VerificationResult, VerifierConfig, VerifierFunction

logger = logger_utils.setup_logger(__name__)


def _default_ballsim_api_url() -> str:
    raw = os.environ.get("BALLSIM_API_URL", "http://localhost:2345").rstrip("/")
    if raw.endswith("/test_program"):
        return raw
    return f"{raw}/test_program"


def _default_ballsim_max_execution_time() -> float:
    return float(os.environ.get("BALLSIM_MAX_EXECUTION_TIME", "1.0"))


def _default_ballsim_scoring_mode() -> str:
    return os.environ.get("BALLSIM_SCORING_MODE", "all_pass")


@register_verifier_config
@dataclass
class BallsimVerifierConfig(VerifierConfig):
    ballsim_api_url: str = field(default_factory=_default_ballsim_api_url)
    ballsim_max_execution_time: float = field(default_factory=_default_ballsim_max_execution_time)
    ballsim_scoring_mode: str = field(default_factory=_default_ballsim_scoring_mode)

    @classmethod
    def from_args(cls, *arg_sources) -> BallsimVerifierConfig:
        verifier_fields = {f.name for f in dataclasses.fields(cls)}
        matching_kwargs: dict[str, Any] = {}
        for source in arg_sources:
            if source is None:
                continue
            for field_name in verifier_fields:
                if hasattr(source, field_name):
                    matching_kwargs[field_name] = getattr(source, field_name)
        for f in dataclasses.fields(cls):
            if f.name not in matching_kwargs:
                if f.default_factory is not dataclasses.MISSING:
                    matching_kwargs[f.name] = f.default_factory()
                elif f.default is not dataclasses.MISSING:
                    matching_kwargs[f.name] = f.default
        return cls(**matching_kwargs)


@register_verifier
class BallsimVerifier(VerifierFunction):
    """
    Verifier that executes Python code against BounceSim test cases using an external API.
    """

    _session_pool = None

    def __init__(self, verifier_config: BallsimVerifierConfig) -> None:
        super().__init__("ballsim", verifier_config=verifier_config, weight=1.0)

    @classmethod
    def _get_session(cls):
        if cls._session_pool is None:
            cls._session_pool = requests.Session()
            retry_config = requests.adapters.Retry(
                total=3,
                connect=3,
                read=3,
                status=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=None,
            )
            adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry_config)
            cls._session_pool.mount("http://", adapter)
            cls._session_pool.mount("https://", adapter)
        return cls._session_pool

    def extract_python_code(self, model_output: str) -> str:
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, model_output, re.DOTALL)
        if not matches:
            return model_output
        return matches[-1].strip()

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        python_code = self.extract_python_code(prediction)

        if isinstance(label, str):
            try:
                tests = json.loads(label)
            except json.JSONDecodeError:
                logger.warning("Failed to parse BounceSim tests as JSON; got string label")
                return VerificationResult(score=0.0)
        else:
            tests = label

        payload = {
            "program": python_code,
            "tests": tests,
            "max_execution_time": self.verifier_config.ballsim_max_execution_time,
        }

        try:
            session = self._get_session()
            http_timeout = max(30, min(300, self.verifier_config.ballsim_max_execution_time * 10))

            def make_request():
                response = session.post(
                    self.verifier_config.ballsim_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=http_timeout,
                )
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(make_request)
            passes = result["results"]
            pass_rate = sum(passes) / len(passes) if passes else 0.0

            if self.verifier_config.ballsim_scoring_mode == "pass_rate":
                score = pass_rate
            else:
                score = 1.0 if pass_rate == 1.0 else 0.0
            return VerificationResult(score=score)
        except Exception as e:
            logger.warning(f"Error verifying ballsim code sample: {e}")
            return VerificationResult(score=0.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous __call__ method from within an async context. Use async_call instead."
                )
            else:
                return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
        except RuntimeError as e:
            if "cannot schedule new futures after interpreter shutdown" in str(e):
                logger.warning("Skipping ballsim verification due to interpreter shutdown")
                return VerificationResult(score=0.0, reasoning="Verification skipped due to shutdown")
            try:
                return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
            except Exception as nested_e:
                logger.warning(f"Error verifying ballsim sample during shutdown: {nested_e}")
                return VerificationResult(score=0.0, reasoning=f"Verification failed: {nested_e}")
        except Exception as e:
            logger.warning(f"Error verifying ballsim sample: {e}")
            return VerificationResult(score=0.0, reasoning=f"Verification failed: {e}")

    @classmethod
    def get_config_class(cls) -> type:
        return BallsimVerifierConfig
