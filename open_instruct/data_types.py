import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch


class ShutdownSentinel:
    """Sentinel value to signal thread shutdown via queue."""


@dataclass
class TokenStatistics:
    """Container for token statistics from inference."""

    num_prompt_tokens: int
    num_response_tokens: int
    generation_time: float
    earliest_start_time: float | None = None


@dataclass
class ToolCallStats:
    """Statistics for a single tool call."""

    tool_name: str
    success: bool
    runtime: float


@dataclass
class RequestInfo:
    """Container for tool usage information used in queue payloads."""

    num_calls: list[int]
    timeouts: list[int]
    tool_errors: list[str]
    tool_outputs: list[str]
    tool_runtimes: list[float]
    tool_calleds: list[bool]
    tool_call_stats: list[list[ToolCallStats]] = field(default_factory=list)
    rollout_states: list[dict] = field(default_factory=list)
    """Per-sample rollout state dicts (rewards, step_count, done, info) — always present."""


@dataclass
class GenerationResult:
    """Container for generation results returned via Ray queues."""

    responses: list[list[int]]
    finish_reasons: list[str]
    masks: list[list[int]]
    request_info: RequestInfo
    index: int | None
    prompt_id: str | None
    token_statistics: TokenStatistics | None = None
    start_time: float | None = None
    logprobs: list[list[float]] | None = None
    reward_scores: list[float] | None = None
    reward_metrics: dict[str, Any] | None = None
    model_step: int | None = None


@dataclass
class EnvConfigEntry:
    """Entry for a single environment configuration."""

    env_name: str
    is_text_env: bool
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """Wrapper for environment configuration + related metadata."""

    max_steps: int = 100
    env_configs: dict[str, EnvConfigEntry] = field(default_factory=dict)
    """Mapping from env_name to its configuration entry."""


@dataclass
class PromptRequest:
    """Container for prompt requests sent via Ray queues.

    Note: We intentionally type `generation_config` as `Any` to avoid importing
    heavy dependencies (e.g., vLLM) at import time in deserializers like Ray's
    `_QueueActor`.
    """

    prompt: list[int]
    generation_config: Any
    index: int
    prompt_id: str
    is_eval: bool = False
    active_tools: list[str] | None = None
    """List of tool names that are active for this sample. If None, all tools are active."""
    env_config: EnvConfig = field(default_factory=EnvConfig)
    ground_truth: Any = None
    """Optional ground truth override (e.g. from evolving rubrics). When set, the vLLM
    engine uses this instead of looking up the ground truth from the dataset."""


@dataclass
class CollatedBatchData:
    """Container for collated batch data passed to training workers."""

    query_responses: list[torch.Tensor]
    attention_masks: list[torch.Tensor]
    position_ids: list[torch.Tensor]
    advantages: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    vllm_logprobs: list[torch.Tensor]
    # Optional value-model plumbing. When use_value_model=False these remain None and the
    # trainer uses the GRPO group-relative advantages above.
    rewards: list[torch.Tensor] | None = None
    """Per-token reward tensor aligned with query_responses (same shape). Typically sparse - a
    single nonzero entry at the terminal response token for each sub-sequence."""
    dones: list[torch.Tensor] | None = None
    """Per-token done mask (float) aligned with query_responses (same shape). 1 at the terminal
    token of each sub-sequence, 0 elsewhere."""
    ground_truths: list[list[str]] | None = None
    """Per-pack list of per-sub-sequence ground-truth answer strings (used for value-model
    conditioning; one entry per packed sub-sequence)."""
    sibling_rollouts: list[list[list[dict]]] | None = None
    """Per-pack list of per-sub-sequence list of sibling rollouts
    (each sibling is a dict: ``{"text": str, "is_correct": bool}``). Used by the
    rollout_context / correct_demo / lm_yesno_siblings value conditioning templates."""
    segment_boundaries: list[torch.Tensor] | None = None
    """Per-token boolean mask (1 where a SAE segment starts, 0 elsewhere). None when SAE is off."""

    _TENSOR_LIST_FIELDS = (
        "query_responses",
        "attention_masks",
        "position_ids",
        "advantages",
        "response_masks",
        "vllm_logprobs",
        "rewards",
        "dones",
        "segment_boundaries",
    )

    def __getitem__(self, idx: int | slice) -> "CollatedBatchData":
        return CollatedBatchData(
            **{
                f.name: (getattr(self, f.name)[idx] if getattr(self, f.name) is not None else None)
                for f in dataclasses.fields(self)
            }
        )

    def __len__(self) -> int:
        return len(self.query_responses)

    def to(self, device: torch.device, non_blocking: bool = True) -> "CollatedBatchData":
        replacements: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name in self._TENSOR_LIST_FIELDS:
                replacements[f.name] = [t.to(device, non_blocking=non_blocking) for t in value]
        return dataclasses.replace(self, **replacements)
