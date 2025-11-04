from dataclasses import dataclass
from typing import Any


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
    dataset_index: int | None = None
    epoch_number: int | None = None
    token_statistics: TokenStatistics | None = None
    start_time: float | None = None
    logprobs: list[list[float]] | None = None

    def good_outputs(self) -> list[bool]:
        return [
            len(self.request_info.tool_outputs[i]) > 0
            and self.request_info.tool_calleds[i]
            and not self.request_info.timeouts[i]
            and not self.request_info.tool_errors[i]
            for i in range(len(self.request_info.tool_outputs))
        ]


@dataclass
class PromptRequest:
    """Container for prompt requests sent via Ray queues.

    Note: We intentionally type `generation_config` as `Any` to avoid importing
    heavy dependencies (e.g., vLLM) at import time in deserializers like Ray's
    `_QueueActor`.
    """

    prompt: list[int]
    generation_config: Any
    epoch_number: int | None = None
    training_step: int | None = None
    dataset_index: int | None = None
    is_eval: bool = False
    start_time: float | None = None
