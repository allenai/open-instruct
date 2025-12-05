from dataclasses import dataclass
from typing import Any


@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 256
    n: int = 1
    stop: list[str] | None = None
    seed: int | None = None
    logprobs: int | None = 1


@dataclass
class TokenStatistics:
    """Container for token statistics from inference."""

    num_prompt_tokens: int
    num_response_tokens: int
    generation_time: float
    earliest_start_time: float | None = None


@dataclass
class RequestInfo:
    """Container for tool usage information used in queue payloads."""

    num_calls: list[int]
    timeouts: list[int]
    tool_errors: list[str]
    tool_outputs: list[str]
    tool_runtimes: list[float]
    tool_calleds: list[bool]


@dataclass
class GenerationResult:
    """Container for generation results returned via Ray queues."""

    responses: list[list[int]]
    finish_reasons: list[str]
    masks: list[list[int]]
    request_info: RequestInfo
    dataset_index: int | None
    prompt_id: str | None
    token_statistics: TokenStatistics | None = None
    start_time: float | None = None
    logprobs: list[list[float]] | None = None
    reward_scores: list[float] | None = None
    reward_metrics: dict[str, Any] | None = None


@dataclass
class PromptRequest:
    """Container for prompt requests sent via Ray queues.

    Note: We intentionally type `generation_config` as `Any` to avoid importing
    heavy dependencies (e.g., vLLM) at import time in deserializers like Ray's
    `_QueueActor`.
    """

    prompt: list[int]
    generation_config: Any
    dataset_index: int
    prompt_id: str
    is_eval: bool = False
