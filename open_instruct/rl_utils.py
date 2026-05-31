import contextlib
import datetime
import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np
import torch

from open_instruct import data_types, logger_utils, model_utils, utils

T = TypeVar("T")
logger = logger_utils.setup_logger(__name__)

_rollout_executor = ThreadPoolExecutor(max_workers=2)
_rollout_save_lock = threading.Lock()
ROLLOUT_SHARD_SIZE = 10000


def _json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def _log_rollout_save_failure(future: Future) -> None:
    try:
        future.result()
    except Exception:
        logger.exception("Failed to save rollout traces")


@dataclass
class RolloutMetadata:
    run_name: str
    git_commit: str
    model_name: str
    timestamp: str


@dataclass
class RolloutRecord:
    step: int
    sample_idx: int
    prompt_idx: int
    prompt_tokens: list[int]
    response_tokens: list[int]
    reward: float
    advantage: float
    finish_reason: str
    dataset: str
    ground_truth: list[int] | None = None
    request_info: dict | None = None
    logprobs: list[float] | None = None


@dataclass
class FilteredRolloutRecord:
    step: int
    filter_reason: str
    sample_idx: int
    prompt_idx: int
    prompt_id: str | None
    dataset_index: int | None
    model_step: int | None
    prompt_tokens: list[int]
    raw_prompt: str | None
    response_tokens: list[int]
    decoded_response: str | None
    reward: float
    finish_reason: str
    dataset: str
    ground_truth: Any = None
    active_tools: list[str] | None = None
    request_info: dict | None = None
    logprobs: list[float] | None = None
    reward_metrics: dict[str, Any] | None = None


def save_rollout_metadata(save_path: str, run_name: str, model_name: str | None) -> None:
    """Save metadata about the rollout collection to disk.

    Creates a JSONL file containing run information including git commit,
    model name, and timestamp for traceability.

    Args:
        save_path: Directory to save metadata file.
        run_name: Experiment run name.
        model_name: Name/path of the model being trained.
    """
    metadata = RolloutMetadata(
        run_name=run_name,
        git_commit=utils.get_git_commit(),
        model_name=model_name or "unknown",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    metadata_path = os.path.join(save_path, f"{run_name}_metadata.jsonl")
    os.makedirs(save_path, exist_ok=True)
    with open(metadata_path, "w") as f:
        f.write(json.dumps(asdict(metadata), default=_json_default) + "\n")
    logger.info(f"Saved rollout metadata to {metadata_path}")


def _get_request_info_for_sample(request_info: data_types.RequestInfo | None, i: int) -> dict | None:
    """Extract per-sample request info from batch-level RequestInfo."""
    if not request_info:
        return None
    return {
        "num_calls": request_info.num_calls[i] if i < len(request_info.num_calls) else 0,
        "timeouts": request_info.timeouts[i] if i < len(request_info.timeouts) else 0,
        "tool_errors": request_info.tool_errors[i] if i < len(request_info.tool_errors) else "",
        "tool_outputs": request_info.tool_outputs[i] if i < len(request_info.tool_outputs) else "",
        "tool_runtimes": request_info.tool_runtimes[i] if i < len(request_info.tool_runtimes) else 0.0,
        "tool_calleds": request_info.tool_calleds[i] if i < len(request_info.tool_calleds) else False,
        "tool_call_stats": (
            [asdict(s) for s in request_info.tool_call_stats[i]] if i < len(request_info.tool_call_stats) else []
        ),
        "rollout_state": request_info.rollout_states[i] if i < len(request_info.rollout_states) else {},
    }


def _save_rollouts(
    save_path: str,
    run_name: str,
    step: int,
    batch: model_utils.Batch,
    result: data_types.GenerationResult,
    advantages: np.ndarray,
    num_samples_per_prompt: int,
    shard_idx: int,
) -> None:
    shard_filename = f"{run_name}_rollouts_{shard_idx:06d}.jsonl"
    filepath = os.path.join(save_path, shard_filename)
    os.makedirs(save_path, exist_ok=True)

    assert batch.scores is not None, "batch.scores must not be None when saving rollouts"

    records = []
    for i in range(len(batch.queries)):
        records.append(
            asdict(
                RolloutRecord(
                    step=step,
                    sample_idx=i,
                    prompt_idx=i // num_samples_per_prompt,
                    prompt_tokens=batch.queries[i],
                    response_tokens=result.responses[i],
                    reward=float(batch.scores[i]),
                    advantage=float(advantages[i]),
                    finish_reason=result.finish_reasons[i],
                    dataset=batch.datasets[i],
                    ground_truth=batch.ground_truths[i],
                    request_info=_get_request_info_for_sample(result.request_info, i),
                    logprobs=result.logprobs[i] if result.logprobs else None,
                )
            )
        )

    with _rollout_save_lock, open(filepath, "a") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default) + "\n")
    logger.info(f"Saved {len(records)} rollouts to {filepath}")


def _save_filtered_rollouts(
    save_path: str,
    run_name: str,
    step: int,
    filter_reason: str,
    batch: model_utils.Batch,
    result: data_types.GenerationResult,
    num_samples_per_prompt: int,
    shard_idx: int,
) -> None:
    shard_filename = f"{run_name}_filtered_rollouts_{shard_idx:06d}.jsonl"
    filepath = os.path.join(save_path, shard_filename)
    os.makedirs(save_path, exist_ok=True)

    assert batch.scores is not None, "batch.scores must not be None when saving filtered rollouts"

    records = []
    for i in range(len(batch.queries)):
        model_step = (
            result.model_steps[i]
            if result.model_steps is not None and i < len(result.model_steps)
            else result.model_step
        )
        records.append(
            asdict(
                FilteredRolloutRecord(
                    step=step,
                    filter_reason=filter_reason,
                    sample_idx=i,
                    prompt_idx=i // num_samples_per_prompt,
                    prompt_id=result.prompt_id,
                    dataset_index=batch.indices[i] if batch.indices is not None else None,
                    model_step=model_step,
                    prompt_tokens=batch.queries[i],
                    raw_prompt=batch.raw_queries[i] if batch.raw_queries is not None else None,
                    response_tokens=result.responses[i],
                    decoded_response=batch.decoded_responses[i] if batch.decoded_responses is not None else None,
                    reward=float(batch.scores[i]),
                    finish_reason=result.finish_reasons[i],
                    dataset=batch.datasets[i],
                    ground_truth=batch.ground_truths[i],
                    active_tools=batch.active_tools[i] if batch.active_tools is not None else None,
                    request_info=_get_request_info_for_sample(result.request_info, i),
                    logprobs=result.logprobs[i] if result.logprobs else None,
                    reward_metrics=result.reward_metrics,
                )
            )
        )

    with _rollout_save_lock, open(filepath, "a") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default) + "\n")
    logger.info(f"Saved {len(records)} filtered rollouts to {filepath}")


def save_filtered_rollouts_to_disk(
    save_path: str,
    run_name: str,
    step: int,
    filter_reason: str,
    batch: model_utils.Batch,
    result: data_types.GenerationResult,
    num_samples_per_prompt: int,
    total_samples_written: int,
) -> None:
    """Asynchronously save filtered rollout records to disk for debugging."""
    shard_idx = total_samples_written // ROLLOUT_SHARD_SIZE
    future = _rollout_executor.submit(
        _save_filtered_rollouts,
        save_path,
        run_name,
        step,
        filter_reason,
        batch,
        result,
        num_samples_per_prompt,
        shard_idx,
    )
    future.add_done_callback(_log_rollout_save_failure)


def _to_cpu_tensor_list(tensors):
    if tensors is None:
        return None
    return [t.detach().cpu() for t in tensors]


def _flatten_tensor(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> list:
    tensor = tensor.detach().cpu()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.reshape(-1).tolist()


def _prompt_boundaries_from_masks(
    prompt_mask: torch.Tensor, attention_mask: torch.Tensor, rollout_sample_ids: torch.Tensor | None = None
) -> list[dict]:
    """Return packed prompt/sequence spans in shifted logprob-token coordinates."""
    prompt_mask_cpu = torch.atleast_2d(prompt_mask.detach().bool().cpu())
    attention_mask_cpu = torch.atleast_2d(attention_mask.detach().long().cpu())
    rollout_sample_ids_cpu = (
        torch.atleast_2d(rollout_sample_ids.detach().long().cpu()) if rollout_sample_ids is not None else None
    )
    boundaries = []
    for row_idx in range(attention_mask_cpu.shape[0]):
        row_attention = attention_mask_cpu[row_idx]
        row_prompt = prompt_mask_cpu[row_idx]
        row_rollout_sample_ids = rollout_sample_ids_cpu[row_idx] if rollout_sample_ids_cpu is not None else None
        token_idx = 0
        while token_idx < row_attention.numel():
            sequence_id = int(row_attention[token_idx].item())
            if sequence_id <= 0:
                token_idx += 1
                continue

            sequence_start = token_idx
            while token_idx < row_attention.numel() and int(row_attention[token_idx].item()) == sequence_id:
                token_idx += 1
            sequence_end = token_idx

            prompt_positions = torch.nonzero(row_prompt[sequence_start:sequence_end], as_tuple=False).flatten()
            if prompt_positions.numel() > 0:
                prompt_start = sequence_start + int(prompt_positions[0].item())
                prompt_end = sequence_start + int(prompt_positions[-1].item()) + 1
            else:
                prompt_start = None
                prompt_end = sequence_start

            boundary = {
                "row": row_idx,
                "sequence_id": sequence_id,
                "sequence_start": sequence_start,
                "prompt_start": prompt_start,
                "prompt_end": prompt_end,
                "sequence_end": sequence_end,
            }
            if row_rollout_sample_ids is not None:
                valid_sample_ids = row_rollout_sample_ids[sequence_start:sequence_end]
                valid_sample_ids = valid_sample_ids[valid_sample_ids >= 0]
                if valid_sample_ids.numel() > 0:
                    boundary["rollout_sample_idx"] = int(valid_sample_ids[0].item())
            boundaries.append(boundary)
    return boundaries


def _save_trainer_logprobs(
    save_path: str,
    run_name: str,
    step: int,
    trainer_logprobs,
    response_masks,
    sp_size: int,
    trainer_model_step: int | None = None,
    input_token_ids=None,
    token_ids=None,
    prompt_masks=None,
    attention_masks=None,
    rollout_sample_ids=None,
    model_steps=None,
    vllm_logprobs=None,
    rank: int | None = None,
    world_size: int | None = None,
    dp_rank: int | None = None,
    sp_rank: int | None = None,
) -> None:
    """Append per-sample trainer logprobs for ``step`` to a step-keyed JSONL."""
    rank_suffix = "" if rank is None or world_size == 1 else f"_rank{rank:05d}"
    filename = f"{run_name}_trainer_logprobs_step{step:06d}{rank_suffix}.jsonl"
    filepath = os.path.join(save_path, filename)
    os.makedirs(save_path, exist_ok=True)
    with open(filepath, "a") as f:
        for i, (lp, mask) in enumerate(zip(trainer_logprobs, response_masks)):
            lp_cpu = _flatten_tensor(lp, torch.float32)
            mask_cpu = _flatten_tensor(mask.bool())
            record = {
                "step": step,
                "sample_idx": i,
                "sp_size": sp_size,
                "trainer_logprobs_shape": list(lp.shape),
                "trainer_logprobs": lp_cpu,
                "response_mask_shape": list(mask.shape),
                "response_mask": mask_cpu,
                "logprob_token_offset": 1,
            }
            if trainer_model_step is not None:
                record["trainer_model_step"] = trainer_model_step
            if rank is not None:
                record["rank"] = rank
            if world_size is not None:
                record["world_size"] = world_size
            if dp_rank is not None:
                record["dp_rank"] = dp_rank
            if sp_rank is not None:
                record["sp_rank"] = sp_rank
            if input_token_ids is not None:
                input_ids = input_token_ids[i]
                record["input_token_ids_shape"] = list(input_ids.shape)
                record["input_token_ids"] = _flatten_tensor(input_ids, torch.long)
            if token_ids is not None:
                ids = token_ids[i]
                record["token_ids_shape"] = list(ids.shape)
                record["token_ids"] = _flatten_tensor(ids, torch.long)
            if rollout_sample_ids is not None:
                ids = rollout_sample_ids[i]
                record["rollout_sample_ids_shape"] = list(ids.shape)
                record["rollout_sample_ids"] = _flatten_tensor(ids, torch.long)
            if model_steps is not None:
                steps = model_steps[i]
                record["model_steps_shape"] = list(steps.shape)
                flattened_steps = _flatten_tensor(steps, torch.long)
                record["model_steps"] = flattened_steps
                unique_steps = sorted({int(s) for s in flattened_steps if int(s) >= 0})
                if len(unique_steps) == 1:
                    record["model_step"] = unique_steps[0]
            if vllm_logprobs is not None:
                logprobs = vllm_logprobs[i]
                record["vllm_logprobs_shape"] = list(logprobs.shape)
                record["vllm_logprobs"] = _flatten_tensor(logprobs, torch.float32)
            if prompt_masks is not None:
                prompt_mask = prompt_masks[i].bool()
                record["prompt_mask_shape"] = list(prompt_mask.shape)
                record["prompt_mask"] = _flatten_tensor(prompt_mask)
            if attention_masks is not None:
                attention_mask = attention_masks[i]
                record["attention_mask_shape"] = list(attention_mask.shape)
                record["attention_mask"] = _flatten_tensor(attention_mask, torch.long)
            if prompt_masks is not None and attention_masks is not None:
                record["prompt_boundaries"] = _prompt_boundaries_from_masks(
                    prompt_masks[i],
                    attention_masks[i],
                    rollout_sample_ids[i] if rollout_sample_ids is not None else None,
                )
            f.write(json.dumps(record, default=_json_default) + "\n")
    logger.info(f"Saved trainer logprobs for step {step} ({len(trainer_logprobs)} samples) to {filepath}")


def save_trainer_logprobs_to_disk(
    save_path: str,
    run_name: str,
    step: int,
    trainer_logprobs,
    response_masks,
    sp_size: int = 1,
    trainer_model_step: int | None = None,
    input_token_ids=None,
    token_ids=None,
    prompt_masks=None,
    attention_masks=None,
    rollout_sample_ids=None,
    model_steps=None,
    vllm_logprobs=None,
    rank: int | None = None,
    world_size: int | None = None,
    dp_rank: int | None = None,
    sp_rank: int | None = None,
) -> None:
    """Asynchronously save trainer-side logprobs alongside rollouts for offline
    analysis (e.g. diagnosing vLLM-vs-trainer logprob divergence).

    With ``sp_size > 1`` each trainer rank sees a partial sequence, so the
    saved tensors represent only the local SP slice; downstream consumers
    must gather across ranks themselves.

    ``token_ids`` should be shifted the same way as ``trainer_logprobs`` (i.e.
    ``query_responses[:, 1:]``); ``input_token_ids`` can carry the full
    unshifted model input for easier reconstruction.

    When ``rank`` is provided, each rank writes to its own file to avoid
    cross-process appends to the same JSONL.
    """
    trainer_logprobs = _to_cpu_tensor_list(trainer_logprobs)
    response_masks = _to_cpu_tensor_list(response_masks)
    input_token_ids = _to_cpu_tensor_list(input_token_ids)
    token_ids = _to_cpu_tensor_list(token_ids)
    prompt_masks = _to_cpu_tensor_list(prompt_masks)
    attention_masks = _to_cpu_tensor_list(attention_masks)
    rollout_sample_ids = _to_cpu_tensor_list(rollout_sample_ids)
    model_steps = _to_cpu_tensor_list(model_steps)
    vllm_logprobs = _to_cpu_tensor_list(vllm_logprobs)

    future = _rollout_executor.submit(
        _save_trainer_logprobs,
        save_path,
        run_name,
        step,
        trainer_logprobs,
        response_masks,
        sp_size,
        trainer_model_step,
        input_token_ids,
        token_ids,
        prompt_masks,
        attention_masks,
        rollout_sample_ids,
        model_steps,
        vllm_logprobs,
        rank,
        world_size,
        dp_rank,
        sp_rank,
    )
    future.add_done_callback(_log_rollout_save_failure)


def save_rollouts_to_disk(
    save_path: str,
    run_name: str,
    step: int,
    batch: model_utils.Batch,
    result: data_types.GenerationResult,
    advantages: np.ndarray,
    num_samples_per_prompt: int,
    total_samples_written: int,
) -> None:
    """Asynchronously save rollout records to disk.

    Submits the rollout saving task to a thread pool executor for non-blocking I/O.
    Records are saved to JSONL files, sharded by total_samples_written.

    Args:
        save_path: Directory to save rollout files.
        run_name: Experiment run name for file naming.
        step: Training step number.
        batch: Batch containing queries, scores, ground_truths, datasets.
        result: Generation result with responses, finish_reasons, request_info.
        advantages: Calculated advantage values per sample.
        num_samples_per_prompt: Number of samples generated per prompt.
        total_samples_written: Total samples written so far, used for sharding.
    """
    shard_idx = total_samples_written // ROLLOUT_SHARD_SIZE
    future = _rollout_executor.submit(
        _save_rollouts, save_path, run_name, step, batch, result, advantages, num_samples_per_prompt, shard_idx
    )
    future.add_done_callback(_log_rollout_save_failure)


@dataclass
class Timer(contextlib.ContextDecorator):
    """A context manager and decorator for timing code blocks"""

    description: str
    noop: bool = False
    start_time: float = field(init=False)
    end_time: float = field(init=False)
    duration: float = field(init=False)

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        if not self.noop:
            logger.info(f"{self.description}: {self.duration:.3f} seconds")


@dataclass
class PackedSequences(Generic[T]):
    query_responses: list[torch.Tensor]
    """packed query and response (batch_size, pack_length)"""
    attention_masks: list[torch.Tensor]
    """3D attention mask for packed sequences (batch_size, pack_length, pack_length);
    it basically uses a intra-document mask for each query response pair;
    see https://huggingface.co/blog/sirluk/llm-sequence-packing for more details
    """
    response_masks: list[torch.Tensor]
    """bool response mask for packed sequences (batch_size, pack_length)"""
    original_responses: list[list[int]]
    """need the original response for broadcast (batch_size, response_length)"""
    prompt_masks: list[torch.Tensor] | None = None
    """bool prompt mask for packed sequences (batch_size, pack_length)"""
    rollout_sample_ids: list[torch.Tensor] | None = None
    """original rollout sample index for each packed token (batch_size, pack_length)"""
    model_steps: list[torch.Tensor] | None = None
    """model step that produced each packed token's rollout logprob (batch_size, pack_length)"""
    advantages: list[torch.Tensor] | None = None
    """packed advantages (batch_size, pack_length) (to be filled in by the main process)"""
    num_actions: list[torch.Tensor] | None = None
    """packed number of actions (batch_size, pack_length)"""
    position_ids: list[torch.Tensor] | None = None
    """packed position ids (batch_size, pack_length)"""
    packed_seq_lens: list[torch.Tensor] | None = None
    """packed sequence lengths (batch_size, pack_length)"""
    vllm_logprobs: list[torch.Tensor] | None = None
    """packed vLLM logprobs for comparison (batch_size, pack_length)"""
    dones: list[torch.Tensor] | None = None
    """packed dones (batch_size, pack_length), specifies the sequence boundaries
    E.g., [0, 0, 0, 0, 1, 0, 0, 0, 0, 2] means the first sequence ends at index 4, and the
    second sequence ends at index 9
    """
    rewards: list[torch.Tensor] | None = None
    """packed rewards (batch_size, pack_length)"""


def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def pack_sequences(
    queries: list[list[int]],
    responses: list[list[int]],
    masks: list[list[int]],
    pack_length: int,
    pad_token_id: int,
    vllm_logprobs: list[list[float]],
    rollout_sample_ids: list[int] | None = None,
    model_steps: list[int | None] | None = None,
    min_num_batches: int = 1,
    mask_tool_use: bool = False,
) -> PackedSequences:
    """Pack query-response pairs into sequences for training.

    Args:
        queries: List of query token sequences
        responses: List of response token sequences
        masks: List of tool masks for each response
        pack_length: Maximum length of each packed sequence
        pad_token_id: Token ID used for padding
        vllm_logprobs: Log probabilities from vLLM for each response
        rollout_sample_ids: Original rollout trace sample indices for each
            query/response pair. Defaults to the current list index.
        model_steps: Model step that produced each response's vLLM logprobs.
        min_num_batches: Minimum number of packed batches to produce.
            Used to ensure we have a batch for each rank in distributed training.

    Returns:
        PackedSequences containing the packed training data.
    """
    assert not any(pad_token_id in query for query in queries)
    if rollout_sample_ids is None:
        rollout_sample_ids = list(range(len(queries)))
    assert len(rollout_sample_ids) == len(queries)
    input_rollout_sample_ids = rollout_sample_ids
    if model_steps is None:
        model_steps = [None] * len(queries)
    assert len(model_steps) == len(queries)

    # Calculate total tokens to determine effective pack_length
    total_tokens = 0
    for query, response in zip(queries, responses):
        query_len = len(query)
        response_len = sum(1 for t in response if t != pad_token_id)
        total_tokens += query_len + response_len

    # Reduce pack_length if needed to ensure min_num_batches
    # Note: sequences longer than effective_pack_length will naturally get their own pack(s)
    # since the packing loop starts a new pack when a sequence doesn't fit
    if total_tokens > 0 and min_num_batches > 1:
        target_pack_length = total_tokens // min_num_batches
        # Don't exceed the original pack_length
        effective_pack_length = min(target_pack_length, pack_length)
    else:
        effective_pack_length = pack_length

    # TODO: for some reason vLLM *can* generate the padding token in the responses; investigate
    # assert not any(pad_token_id in response for response in responses)

    query_responses = []
    attention_masks = []
    response_masks = []
    prompt_masks = []
    rollout_sample_ids = []
    packed_model_steps = []
    dones = []
    num_actions = []
    packed_seq_lens = []
    packed_vllm_logprobs = []
    cur_data = []
    cur_response_mask = []
    cur_prompt_mask = []
    cur_rollout_sample_ids = []
    cur_model_steps = []
    cur_num_actions = []
    cur_packed_seq_lens = []
    cur_attention_mask = []
    cur_dones = []
    cur_vllm_logprobs = []
    offset = 0
    for i in range(len(queries)):
        query = queries[i]
        response = responses[i]
        mask = masks[i]
        rollout_sample_id = input_rollout_sample_ids[i]
        model_step = -1 if model_steps[i] is None else int(model_steps[i])
        # remove padding (but using vllm so this should not be needed, but just in case)
        query = [t for t in query if t != pad_token_id]

        # Filter out padding tokens from response, mask, and logprobs together
        response_logprobs_unfiltered = vllm_logprobs[i]

        assert len(response_logprobs_unfiltered) == len(response), (
            f"Response {i}: logprobs length ({len(response_logprobs_unfiltered)}) != response length ({len(response)})"
        )

        filtered_response = []
        filtered_mask = []
        filtered_logprobs = []
        for j, (token, mask_val) in enumerate(zip(response, mask)):
            if token != pad_token_id:
                filtered_response.append(token)
                filtered_mask.append(mask_val)
                filtered_logprobs.append(response_logprobs_unfiltered[j])

        response = filtered_response
        response_tool_mask = filtered_mask
        response_logprobs = filtered_logprobs

        query_response = query + response

        # Process vLLM logprobs
        # For query tokens, we set logprobs to NaN, for response tokens we use vLLM logprobs
        query_logprobs = [float("nan")] * len(query)
        assert len(response_logprobs) == len(response), (
            f"Response {i}: logprobs length {len(response_logprobs)} != response length {len(response)}. "
            f"Original lengths before filtering: response={len(responses[i])}, logprobs={len(vllm_logprobs[i])}. "
            f"This can happen if vLLM returns N-1 logprobs for N tokens (missing first token logprob)."
        )
        combined_logprobs = query_logprobs + response_logprobs
        # only flush if we have data and we exceed the pack length.
        if len(query_response) + len(cur_data) > effective_pack_length and len(cur_data) > 0:
            query_responses.append(cur_data)
            response_masks.append(cur_response_mask)
            prompt_masks.append(cur_prompt_mask)
            rollout_sample_ids.append(cur_rollout_sample_ids)
            packed_model_steps.append(cur_model_steps)
            attention_masks.append(cur_attention_mask)
            num_actions.append(cur_num_actions)
            packed_seq_lens.append(cur_packed_seq_lens)
            dones.append(cur_dones)
            packed_vllm_logprobs.append(cur_vllm_logprobs)
            cur_data = []
            cur_response_mask = []
            cur_prompt_mask = []
            cur_rollout_sample_ids = []
            cur_model_steps = []
            cur_attention_mask = []
            cur_num_actions = []
            cur_packed_seq_lens = []
            cur_dones = []
            cur_vllm_logprobs = []
            offset = i
        cur_data.extend(query_response)
        cur_vllm_logprobs.extend(combined_logprobs)
        cur_num_actions.append(len(response))
        cur_packed_seq_lens.append(len(query_response))

        query_mask = [0] * len(query)
        response_mask = [i + 1 if m else 0 for m in response_tool_mask] if mask_tool_use else [i + 1] * len(response)
        cur_response_mask.extend(query_mask + response_mask)
        cur_prompt_mask.extend([1] * len(query) + [0] * len(response))
        cur_rollout_sample_ids.extend([rollout_sample_id] * len(query_response))
        cur_model_steps.extend([model_step] * len(query_response))
        cur_attention_mask.extend([i + 1 - offset for _ in range(len(query_response))])
        cur_dones.extend([0 for _ in range(len(query) + len(response) - 1)] + [i + 1])

    # Handle leftover data
    if len(cur_data) > 0:
        query_responses.append(cur_data)
        response_masks.append(cur_response_mask)
        prompt_masks.append(cur_prompt_mask)
        rollout_sample_ids.append(cur_rollout_sample_ids)
        packed_model_steps.append(cur_model_steps)
        attention_masks.append(cur_attention_mask)
        num_actions.append(cur_num_actions)
        packed_seq_lens.append(cur_packed_seq_lens)
        dones.append(cur_dones)
        packed_vllm_logprobs.append(cur_vllm_logprobs)
    attention_masks_list = [torch.tensor(t) for t in attention_masks]
    return PackedSequences(
        query_responses=[torch.tensor(t) for t in query_responses],
        attention_masks=attention_masks_list,
        position_ids=[reset_position_ids(t.unsqueeze(0)).squeeze(0) for t in attention_masks_list],
        response_masks=[torch.tensor(t, dtype=torch.long) for t in response_masks],
        original_responses=responses,
        prompt_masks=[torch.tensor(t, dtype=torch.long) for t in prompt_masks],
        rollout_sample_ids=[torch.tensor(t, dtype=torch.long) for t in rollout_sample_ids],
        model_steps=[torch.tensor(t, dtype=torch.long) for t in packed_model_steps],
        num_actions=[torch.tensor(t) for t in num_actions],
        packed_seq_lens=[torch.tensor(t) for t in packed_seq_lens],
        dones=[torch.tensor(t) for t in dones],
        vllm_logprobs=[torch.tensor(t, dtype=torch.float) for t in packed_vllm_logprobs],
    )


def print_diff(actual: torch.Tensor, expected: torch.Tensor):
    atol = torch.abs(actual - expected)
    rtol = atol / expected
    print(f"{atol.mean()=}, {rtol.mean()=}")


def calculate_advantages(values: np.ndarray, rewards: np.ndarray, gamma: float, lam: float):
    """Vanilla implementation of GAE. Each row is a separate padded sequence."""
    lastgaelam = 0
    advantages_reversed = []
    gen_length = values.shape[1]
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = np.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def calculate_advantages_packed(
    values: np.ndarray, rewards: np.ndarray, gamma: float, lam: float, dones: np.ndarray, response_masks: np.ndarray
):
    """Packed implementation of GAE. Each row is a packed sequence.
    The `dones` specifies the sequence boundaries, and the `response_masks` specifies the query boundaries.
    """
    response_masks = response_masks.clip(0, 1)
    dones = dones.clip(0, 1)
    lastgaelam = 0
    advantages_reversed = []
    gen_length = values.shape[1]
    for t in reversed(range(gen_length)):
        nonterminal = 1 - dones[:, t]
        nonquery = response_masks[:, t]
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues * nonterminal * nonquery - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam * nonterminal * nonquery
        # print(
        #     f"t: {t}, rewards: {rewards[:, t]}, nextvalues: {nextvalues}, nonterminal: {nonterminal}, "
        #     f"delta: {delta}, lastgaelam: {lastgaelam}"
        # )
        advantages_reversed.append(lastgaelam)
    advantages = np.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: int | None = None, denominator: float | None = None
) -> torch.Tensor:
    """Compute mean of tensor with masked values.

    Returns 0 if mask is empty (no valid elements) to avoid division by zero.
    This can happen with sequence parallel when a chunk contains only query/padding tokens.
    """
    extra_dims = values.ndim - mask.ndim
    if axis is None:
        sum_dims = tuple(range(extra_dims, values.ndim))
    elif axis >= 0:
        sum_dims = axis + extra_dims
    else:
        sum_dims = axis
    numerator = (values * mask).sum(dim=sum_dims)
    denom = mask.sum(dim=axis) if denominator is None else denominator
    # Handle empty mask case (e.g., SP chunk with no response tokens)
    if isinstance(denom, torch.Tensor):
        result = torch.where(denom > 0, numerator / denom, torch.zeros_like(numerator))
    else:
        result = numerator / denom if denom > 0 else torch.zeros_like(numerator)
    return result.flatten(extra_dims).mean(-1) if result.ndim > extra_dims else result
