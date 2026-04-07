"""Manufactoria DSL verifier (HTTP API)."""

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


def _default_manufactoria_api_url() -> str:
    raw = os.environ.get("MANUFACTORIA_API_URL", "http://localhost:1235").rstrip("/")
    if raw.endswith("/test_solution"):
        return raw
    return f"{raw}/test_solution"


def _default_manufactoria_max_execution_time() -> float:
    return float(os.environ.get("MANUFACTORIA_MAX_EXECUTION_TIME", "1.0"))


def _default_manufactoria_scoring_mode() -> str:
    return os.environ.get("MANUFACTORIA_SCORING_MODE", "all_pass")


@register_verifier_config
@dataclass
class ManufactoriaVerifierConfig(VerifierConfig):
    manufactoria_api_url: str = field(default_factory=_default_manufactoria_api_url)
    manufactoria_max_execution_time: float = field(default_factory=_default_manufactoria_max_execution_time)
    manufactoria_scoring_mode: str = field(default_factory=_default_manufactoria_scoring_mode)

    @classmethod
    def from_args(cls, *arg_sources) -> ManufactoriaVerifierConfig:
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
class ManufactoriaVerifier(VerifierFunction):
    """
    Verifier that executes Manufactoria DSL code against test cases using an external API.
    """

    _session_pool = None

    def __init__(self, verifier_config: ManufactoriaVerifierConfig) -> None:
        super().__init__("manufactoria", verifier_config=verifier_config, weight=1.0)

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

    def extract_manufactoria_code(self, model_output: str) -> str:
        pattern = r"```(?:manufactoria)?(.*?)```"
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
        if isinstance(label, str):
            try:
                test_cases = json.loads(label)
            except json.JSONDecodeError:
                logger.warning("Failed to parse Manufactoria tests as JSON; got string label")
                return VerificationResult(score=0.0)
        else:
            test_cases = label

        if not isinstance(test_cases, list) or not test_cases:
            logger.warning("Manufactoria verifier expected a non-empty test case list")
            return VerificationResult(score=0.0)

        payload = {
            "dsl": self.extract_manufactoria_code(prediction),
            "test_cases": test_cases,
            "max_execution_time": self.verifier_config.manufactoria_max_execution_time,
        }

        try:
            session = self._get_session()
            http_timeout = max(30, min(300, self.verifier_config.manufactoria_max_execution_time * 10))

            def make_request():
                response = session.post(
                    self.verifier_config.manufactoria_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=http_timeout,
                )
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(make_request)

            if "all_passed" in result:
                all_pass_score = 1.0 if result["all_passed"] else 0.0
                raw_results = result.get("results", [])
                if isinstance(raw_results, list) and raw_results:
                    passes = [bool(test_result.get("passed", False)) for test_result in raw_results]
                    pass_rate_score = sum(passes) / len(passes)
                else:
                    pass_rate_score = all_pass_score
            elif "results" in result and isinstance(result["results"], list) and result["results"]:
                raw_results = result["results"]
                if isinstance(raw_results[0], dict):
                    passes = [bool(test_result.get("passed", False)) for test_result in raw_results]
                else:
                    passes = [bool(value) for value in raw_results]
                pass_rate_score = sum(passes) / len(passes)
                all_pass_score = 1.0 if pass_rate_score == 1.0 else 0.0
            else:
                logger.warning(f"Unexpected Manufactoria API response format: {result}")
                return VerificationResult(score=0.0)

            if self.verifier_config.manufactoria_scoring_mode == "pass_rate":
                score = pass_rate_score
            else:
                score = all_pass_score
            return VerificationResult(score=score)
        except Exception as e:
            logger.warning(f"Error verifying Manufactoria code sample: {e}")
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
                logger.warning("Skipping Manufactoria verification due to interpreter shutdown")
                return VerificationResult(score=0.0, reasoning="Verification skipped due to shutdown")
            try:
                return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
            except Exception as nested_e:
                logger.warning(f"Error verifying Manufactoria sample during shutdown: {nested_e}")
                return VerificationResult(score=0.0, reasoning=f"Verification failed: {nested_e}")
        except Exception as e:
            logger.warning(f"Error verifying Manufactoria sample: {e}")
            return VerificationResult(score=0.0, reasoning=f"Verification failed: {e}")

    @classmethod
    def get_config_class(cls) -> type:
        return ManufactoriaVerifierConfig
