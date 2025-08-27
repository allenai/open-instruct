from dataclasses import dataclass
from typing import Any, List, Optional


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
    training_step: Optional[int] = None
    start_time: Optional[float] = None


@dataclass
class PromptRequest:
    """Container for prompt requests sent via Ray queues.

    Note: We intentionally type `generation_config` as `Any` to avoid importing
    heavy dependencies (e.g., vLLM) at import time in deserializers like Ray's
    `_QueueActor`.
    """

    prompt: List[int]
    generation_config: Any
    training_step: Optional[int] = None
    dataset_index: Optional[int] = None
    is_eval: bool = False
    start_time: Optional[float] = None
