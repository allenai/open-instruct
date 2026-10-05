"""Token normalization for the legacy Accelerate/DeepSpeed SFT loop."""

import contextlib
from collections.abc import Iterable, Iterator

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType


def count_causal_lm_tokens(batch: dict) -> int:
    # Ulysses shifts before sharding, including labels at shard boundaries. Ordinary
    # causal LM labels are shifted by the model, so label zero is never predicted.
    labels = batch["shift_labels"] if "shift_labels" in batch else batch["labels"][..., 1:]
    return int(labels.ne(-100).sum().item())


def prepare_empty_causal_lm_batch(batch: dict, local_tokens: int) -> dict:
    if local_tokens:
        return batch
    # Fused CE can omit logits and produce NaN loss/gradients for an all-masked
    # shard. Give the forward one dummy target, then multiply its finite loss by
    # zero. The original token count remains zero, and router aux loss is separate.
    key = "shift_labels" if "shift_labels" in batch else "labels"
    dummy_index = 0 if key == "shift_labels" else 1
    labels = batch[key].clone()
    if labels.shape[-1] <= dummy_index:
        raise ValueError("An empty causal LM microbatch must have a prediction position")
    labels[..., dummy_index] = 0
    batch = {**batch, key: labels}
    if key == "shift_labels" and "labels" in batch:
        batch["labels"] = labels
    return batch


def iter_token_normalized_batches(
    dataloader: Iterable[dict], accelerator: Accelerator, accumulation_steps: int
) -> Iterator[tuple[dict, int, torch.Tensor, bool, int]]:
    """Read one optimizer window before backward so its denominator is known.

    Only inputs are buffered, on CPU; forward activations still live for one
    microbatch. The prepared dataloader must give every rank the same number of
    batches, as Accelerate's default even_batches=True configuration does.
    """
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    iterator = iter(dataloader)
    while True:
        batches = []
        for _ in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batches.append(
                {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            )
        if not batches:
            return
        token_counts = [count_causal_lm_tokens(batch) for batch in batches]
        global_tokens = accelerator.reduce(
            torch.tensor(sum(token_counts), dtype=torch.int64, device=accelerator.device), reduction="sum"
        )
        if global_tokens.item() == 0:
            raise ValueError("The accumulation window has no supervised causal LM tokens on any rank")
        for index, (batch, token_count) in enumerate(zip(batches, token_counts)):
            yield batch, token_count, global_tokens, index == len(batches) - 1, len(batches)


@contextlib.contextmanager
def accumulate_token_window(
    accelerator: Accelerator, model: torch.nn.Module, is_last_microbatch: bool, sync_each_batch: bool = False
):
    # Prefetching reaches end_of_dataloader before the last forward, so the usual
    # accumulate() dataloader heuristic would step early. Set the actual boundary
    # explicitly, including a short final window. DeepSpeed's wrapper reads this
    # flag before backward and performs its clipping/step inside backward.
    accelerator.sync_gradients = is_last_microbatch
    context = contextlib.nullcontext() if is_last_microbatch or sync_each_batch else accelerator.no_sync(model)
    with context:
        yield


def gradient_reduction_divisor(accelerator: Accelerator, sequence_parallel_size: int = 1) -> int:
    """Read the effective plugin config after prepare(), which resolves defaults.

    ZeRO-1/2 average data replicas and sum SP shards. ZeRO-3's default coalesced
    reduce-scatter averages all ranks instead. Its reduce_scatter=False path mixes
    these two divisors by bucket size, so a scalar loss cannot compensate for it.
    """
    world_size = accelerator.num_processes
    if sequence_parallel_size < 1 or world_size % sequence_parallel_size:
        raise ValueError("sequence_parallel_size must be positive and divide the process count")
    if sequence_parallel_size == 1:
        return world_size
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        raise ValueError("Sequence-parallel loss normalization requires DeepSpeed")
    zero_config = accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {})
    stage = zero_config.get("stage", 0)
    if stage in (1, 2):
        return world_size // sequence_parallel_size
    if stage == 3:
        if not zero_config.get("reduce_scatter", True):
            raise ValueError(
                "ZeRO-3 sequence parallelism requires reduce_scatter=true for consistent gradient scaling"
            )
        if zero_config.get("zero_quantized_gradients", False) or zero_config.get("zeropp_loco_param"):
            raise ValueError("Sequence-parallel loss normalization does not support quantized gradient reduction")
        return world_size
    raise ValueError("Sequence-parallel loss normalization supports DeepSpeed ZeRO stages 1, 2, and 3")


def token_normalized_loss(
    mean_loss: torch.Tensor,
    local_tokens: int,
    global_tokens: torch.Tensor,
    accelerator: Accelerator,
    num_microbatches: int,
    sequence_parallel_size: int = 1,
    aux_loss: torch.Tensor | None = None,
    aux_loss_coefficient: float = 0.0,
    gradient_divisor: int | None = None,
) -> torch.Tensor:
    """Normalize CE over all predicted tokens, before clipping or a ZeRO step.

    Compensate the backend's gradient divisor, which differs between ZeRO-1/2
    and ZeRO-3 with SP. Router regularization remains a mean over microbatches
    and ranks, not CE tokens.
    """
    if gradient_divisor is None:
        gradient_divisor = gradient_reduction_divisor(accelerator, sequence_parallel_size)
    loss = mean_loss * (local_tokens * gradient_divisor / global_tokens)
    if aux_loss is not None:
        loss = loss + aux_loss_coefficient * aux_loss * gradient_divisor / (
            num_microbatches * accelerator.num_processes
        )
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        # Accelerator.backward divides by configured GAS even for a short window.
        loss = loss * accelerator.gradient_accumulation_steps
    return loss


def mean_causal_lm_loss(
    model_loss: torch.Tensor,
    local_tokens: int,
    aux_loss: torch.Tensor | None = None,
    aux_loss_coefficient: float = 0.0,
) -> torch.Tensor:
    if local_tokens == 0:
        # prepare_empty_causal_lm_batch made this finite, including fused CE
        # implementations that return no logits during training.
        return model_loss * 0.0
    if aux_loss is not None:
        return model_loss - aux_loss_coefficient * aux_loss
    return model_loss


def backward_token_normalized_loss(accelerator: Accelerator, loss: torch.Tensor) -> None:
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        # The window denominator already normalizes the sum; disable DeepSpeed's
        # separate GAS divisor (also important for partial accumulation windows).
        accelerator.backward(loss, scale_wrt_gas=False)
    else:
        accelerator.backward(loss)
