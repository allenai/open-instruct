from dataclasses import dataclass
from typing import List, Optional

import vllm


@dataclass
class TokenStatistics:
    """Container for token statistics from inference."""

    num_prompt_tokens: int
    num_response_tokens: int
    generation_time: float
    earliest_start_time: Optional[float] = None


@dataclass
class RequestInfo:
    """Container for tool usage information used in queue payloads."""

    num_calls: List[int]
    timeouts: List[int]
    tool_errors: List[str]
    tool_outputs: List[str]
    tool_runtimes: List[float]
    tool_calleds: List[bool]


@dataclass
class GenerationResult:
    """Container for generation results returned via Ray queues."""

    responses: List[List[int]]
    finish_reasons: List[str]
    masks: List[List[int]]
    request_info: RequestInfo
    dataset_index: Optional[int] = None
    epoch_number: Optional[int] = None
    token_statistics: Optional[TokenStatistics] = None
    start_time: Optional[float] = None
    logprobs: Optional[List[List[float]]] = None


@dataclass
class PromptRequest:
    """Container for prompt requests sent via Ray queues."""

    prompt: List[int]
    generation_config: vllm.SamplingParams
    epoch_number: int
    training_step: int
    dataset_index: int
    is_eval: bool = False
    start_time: Optional[float] = None
