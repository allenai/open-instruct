"""Harbor integration for GRPO training.

Provides an alternative to ``process_request`` that delegates rollout
generation to Harbor's agent framework (e.g. Terminus-2).  Harbor handles
tool-calling, sandbox management, and multi-turn interaction; this module
converts the resulting trajectory data back into ``CompletionOutput`` objects
that the rest of the pipeline consumes.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import TYPE_CHECKING, Any

import ray

from open_instruct import logger_utils
from open_instruct.data_types import EnvConfig
from open_instruct.vllm_utils import CompletionOutput, RequestOutput, SamplingConfig, split_request_id

if TYPE_CHECKING:
    from open_instruct.vllm_utils import LLMRayActor

logger = logger_utils.setup_logger(__name__)

# Harbor imports — optional dependency.  Functions that need Harbor call
# ``_ensure_harbor()`` at the top to raise a clear error if missing.
try:
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    _harbor_available = True
except ImportError:
    _harbor_available = False


def _ensure_harbor() -> None:
    if not _harbor_available:
        raise ImportError(
            "Harbor is required for use_harbor=True but is not installed. Install it with: pip install harbor"
        )


def make_harbor_trial_config(
    actor: LLMRayActor,
    task_path: str,
    sampling_params: SamplingConfig,
    *,
    agent_name: str = "terminus-2",
    agent_kwargs: dict[str, Any] | None = None,
    environment: str = "docker",
) -> Any:
    """Build a Harbor ``TrialConfig`` from the task path and actor state.

    Returns a ``harbor.models.trial.config.TrialConfig`` instance.
    """
    _ensure_harbor()

    env_type = EnvironmentType.DAYTONA if environment == "daytona" else EnvironmentType.DOCKER
    node_ip = ray.util.get_node_ip_address()
    max_model_len = getattr(actor.llm_engine.model_config, "max_model_len", 32768)
    merged_kwargs: dict[str, Any] = {
        "api_base": f"http://{node_ip}:{actor.server_port}/v1",
        "collect_rollout_details": True,
        "linear_history": True,
        "temperature": sampling_params.temperature,
        "model_info": {
            "max_output_tokens": 32768,
            "max_input_tokens": max_model_len + 32768,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    }
    if agent_kwargs:
        merged_kwargs.update(agent_kwargs)

    is_local = task_path.startswith("/") or task_path.startswith(".")
    task_cfg = TaskConfig(path=task_path) if is_local else TaskConfig(name=task_path)

    return TrialConfig(
        task=task_cfg,
        agent=AgentConfig(
            name=agent_name, model_name=f"hosted_vllm/{actor.model_name.split('/')[-1]}", kwargs=merged_kwargs
        ),
        environment=EnvironmentConfig(type=env_type),
    )


def harbor_output_to_completions(
    trial_result: Any, tokenizer: Any, max_seq_len: int
) -> tuple[list[CompletionOutput], float]:
    """Convert Harbor trial output into ``CompletionOutput`` objects.

    With ``linear_history=True`` the agent context may contain multiple
    rollout details (one per linear-history segment, split at summarization
    boundaries).  Each segment becomes a separate ``CompletionOutput``.

    Returns ``(completions, reward)`` where *reward* is the scalar reward
    from Harbor's verifier.
    """
    reward = 0.0
    if trial_result.verifier_result and trial_result.verifier_result.rewards:
        reward = trial_result.verifier_result.rewards.get("reward", 0.0)

    agent_ctx = trial_result.agent_result
    if agent_ctx is None or not agent_ctx.rollout_details:
        logger.warning("Harbor trial returned no rollout_details; producing empty completion")
        eos_id = tokenizer.eos_token_id
        return [
            CompletionOutput(
                index=0,
                token_ids=[eos_id],
                logprobs=[float("nan")],
                finish_reason="stop",
                mask=[1],
                rollout_state={"harbor_reward": reward},
            )
        ], reward

    completions: list[CompletionOutput] = []
    for seg_idx, segment in enumerate(agent_ctx.rollout_details):
        prompt_token_ids_per_turn: list[list[int]] = segment.get("prompt_token_ids", [])
        completion_token_ids_per_turn: list[list[int]] = segment.get("completion_token_ids", [])
        logprobs_per_turn: list[list[float]] = segment.get("logprobs", [])

        token_ids: list[int] = []
        logprobs: list[float] = []
        mask: list[int] = []

        for turn_idx in range(len(completion_token_ids_per_turn)):
            turn_completion_ids = completion_token_ids_per_turn[turn_idx]
            turn_logprobs = logprobs_per_turn[turn_idx] if turn_idx < len(logprobs_per_turn) else []

            if turn_idx > 0 and turn_idx < len(prompt_token_ids_per_turn):
                prev_prompt_len = (
                    len(prompt_token_ids_per_turn[turn_idx - 1])
                    if turn_idx - 1 < len(prompt_token_ids_per_turn)
                    else 0
                )
                current_prompt = prompt_token_ids_per_turn[turn_idx]
                # Tool/user tokens: new tokens in the current prompt beyond
                # the previous prompt + previous completion.
                tool_start = prev_prompt_len + len(completion_token_ids_per_turn[turn_idx - 1])
                if tool_start < len(current_prompt):
                    tool_tokens = current_prompt[tool_start:]
                    token_ids.extend(tool_tokens)
                    logprobs.extend([0.0] * len(tool_tokens))
                    mask.extend([0] * len(tool_tokens))

            token_ids.extend(turn_completion_ids)
            if len(turn_logprobs) == len(turn_completion_ids):
                logprobs.extend(turn_logprobs)
            else:
                logprobs.extend(turn_logprobs[: len(turn_completion_ids)])
                logprobs.extend([0.0] * max(0, len(turn_completion_ids) - len(turn_logprobs)))
            mask.extend([1] * len(turn_completion_ids))

        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
            logprobs = logprobs[:max_seq_len]
            mask = mask[:max_seq_len]

        if len(token_ids) == 0:
            eos_id = tokenizer.eos_token_id
            token_ids = [eos_id]
            logprobs = [float("nan")]
            mask = [1]

        is_last_segment = seg_idx == len(agent_ctx.rollout_details) - 1
        finish_reason = "stop" if is_last_segment else "summarized"

        completions.append(
            CompletionOutput(
                index=seg_idx,
                token_ids=token_ids,
                logprobs=logprobs,
                finish_reason=finish_reason,
                mask=mask,
                rollout_state={"harbor_reward": reward, "step_count": len(completion_token_ids_per_turn)},
            )
        )

    return completions, reward


async def process_request_harbor(actor: LLMRayActor, sub_request_id: str, sampling_params: SamplingConfig) -> None:
    """Harbor replacement for ``process_request``.

    Creates a Harbor Trial, runs it, converts the output into
    ``CompletionOutput`` objects, and places the result on the actor's
    completion queue.
    """
    _ensure_harbor()

    base_request_id = split_request_id(sub_request_id)["base_id"]
    request_metadata = actor.request_metadata[base_request_id]
    env_config: EnvConfig = request_metadata.get("env_config", EnvConfig())
    task_path = env_config.harbor_task_path

    if task_path is None:
        raise ValueError(
            f"use_harbor=True but no harbor_task_path in env_config for request {sub_request_id}. "
            "Ensure the dataset provides harbor_task_path in env_config."
        )

    harbor_config = getattr(actor, "_harbor_config", {})
    trial_config = make_harbor_trial_config(
        actor,
        task_path,
        sampling_params,
        agent_name=harbor_config.get("agent_name", "terminus-2"),
        agent_kwargs=harbor_config.get("agent_kwargs"),
        environment=harbor_config.get("environment", "docker"),
    )

    logger.info(
        f"Harbor trial starting for {sub_request_id}: task={task_path}, "
        f"base_url={trial_config.agent.kwargs.get('base_url')}, model={trial_config.agent.model_name}"
    )

    start_time = time.perf_counter()
    try:
        trial = await Trial.create(config=trial_config)
        logger.info(f"Harbor trial created for {sub_request_id}, running agent...")
        trial_result = await trial.run()
    except Exception:
        elapsed = time.perf_counter() - start_time
        logger.exception(f"Harbor trial failed for {sub_request_id} after {elapsed:.1f}s")
        raise
    finally:
        asyncio.get_event_loop().run_in_executor(
            None, lambda: subprocess.run(["docker", "network", "prune", "-f"], capture_output=True)
        )
    generation_time = time.perf_counter() - start_time

    completions, reward = harbor_output_to_completions(
        trial_result, actor.llm_engine.tokenizer, sampling_params.max_tokens
    )

    logger.info(
        f"Harbor trial completed for {sub_request_id}: {len(completions)} segment(s), "
        f"reward={reward:.3f}, time={generation_time:.1f}s"
    )

    request_output = RequestOutput(
        request_id=sub_request_id, prompt_token_ids=request_metadata["prompt_token_ids"], outputs=completions
    )

    actor.active_tasks.pop(sub_request_id, None)
    actor.completion_queue.put(
        {
            "base_request_id": base_request_id,
            "expected_n": request_metadata["original_sampling_params"].n,
            "request_output": request_output,
            "use_tools": True,
        }
    )
