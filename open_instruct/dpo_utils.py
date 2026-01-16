# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DPO utils
Adapted from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
"""

import contextlib
import hashlib
import json
import pathlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq

from open_instruct import logger_utils
from open_instruct.dataset_transformation import TokenizerConfig, compute_config_hash, load_dataset_configs
from open_instruct.dpo_config import DPOConfigProtocol, DPOLossType
from open_instruct.model_utils import TensorCache, log_softmax_and_gather
from open_instruct.padding_free_collator import concatenated_inputs as pf_concatenated_inputs
from open_instruct.padding_free_collator import get_batch_logps as pf_get_batch_logps

__all__ = ["DPOLossType"]

logger = logger_utils.setup_logger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True


def config_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: config_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [config_to_json_serializable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def compute_reference_logprobs_cache_hash(
    model_name_or_path: str,
    model_revision: str | None,
    dpo_loss_type: str,
    concatenated_forward: bool,
    packing: bool,
    use_lora: bool,
    use_qlora: bool,
    max_train_samples: int | None,
    dataset_config_hash: str,
) -> str:
    """Compute deterministic hash for reference logprobs cache.

    This unified function is used by both dpo.py (OLMo-core) and
    dpo_tune_cache.py (Accelerate) to generate consistent cache keys.
    """
    cache_key = {
        "concatenated_forward": concatenated_forward,
        "dataset_config_hash": dataset_config_hash,
        "dpo_loss_type": dpo_loss_type if isinstance(dpo_loss_type, str) else dpo_loss_type.value,
        "max_train_samples": max_train_samples,
        "model_name_or_path": model_name_or_path,
        "model_revision": model_revision,
        "packing": packing,
        "use_lora": use_lora,
        "use_qlora": use_qlora,
    }
    config_str = json.dumps(cache_key, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_reference_cache_hash(args: DPOConfigProtocol, tc: TokenizerConfig) -> str:
    """Compute deterministic hash for reference logprobs cache from config object.

    This convenience wrapper extracts the required fields from a DPO config object
    and delegates to compute_reference_logprobs_cache_hash. Both DPOExperimentConfig
    (OLMo-core) and FlatArguments (Accelerate) satisfy DPOConfigProtocol.
    """
    assert args.model_name_or_path is not None, "model_name_or_path is required"
    assert args.dataset_mixer_list is not None, "dataset_mixer_list is required"
    assert args.dataset_mixer_list_splits is not None, "dataset_mixer_list_splits is required"
    assert args.dataset_transform_fn is not None, "dataset_transform_fn is required"

    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
    dcs = load_dataset_configs(
        args.dataset_mixer_list,
        args.dataset_mixer_list_splits,
        args.dataset_transform_fn,
        transform_fn_args,
        args.dataset_target_columns,
    )
    dataset_config_hash = args.dataset_config_hash or compute_config_hash(dcs, tc)
    return compute_reference_logprobs_cache_hash(
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        dpo_loss_type=args.dpo_loss_type,
        concatenated_forward=args.concatenated_forward,
        packing=args.packing,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        max_train_samples=args.max_train_samples,
        dataset_config_hash=dataset_config_hash,
    )


def build_reference_logprobs_cache(
    model: nn.Module,
    dataloader: Iterable,
    average_log_prob: bool,
    forward_fn: Callable,
    full_dataset_size: int,
    use_lora: bool,
    device: torch.device,
    cache_path: pathlib.Path | None,
    disable_adapter_context: Callable[[], contextlib.AbstractContextManager],
    is_distributed: Callable[[], bool],
    all_reduce_fn: Callable[[torch.Tensor], None],
    is_main_process: bool,
    show_progress: bool = True,
) -> TensorCache:
    """Build a TensorCache with reference logprobs by computing logprobs once for all samples.

    This unified function works with both OLMo-core and Accelerate frameworks by accepting
    framework-specific callbacks for distributed operations.

    Args:
        model: The model to compute logprobs with.
        dataloader: Iterable yielding batches with "index", "chosen_input_ids", etc.
        average_log_prob: Whether to average log probabilities.
        forward_fn: Function(model, batch, average_log_prob) -> (chosen_logps, rejected_logps, aux).
        full_dataset_size: Total number of samples in the dataset.
        use_lora: Whether the model uses LoRA adapters.
        device: Device to place tensors on.
        cache_path: Path to save/load cache. If None, cache is not persisted.
        disable_adapter_context: Callable returning context manager to disable LoRA adapters.
        is_distributed: Callable returning True if distributed training is active.
        all_reduce_fn: Callable that performs MAX all-reduce on a tensor in-place.
        is_main_process: Whether this is the main process (for logging/saving).
        show_progress: Whether to show progress bar.

    Returns:
        TensorCache with "chosen_logps" and "rejected_logps" tensors.

    Raises:
        RuntimeError: If any indices are missing after cache building.
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"Loading reference logprobs cache from {cache_path}")
        return TensorCache.from_disk(cache_path, device=device)

    model.eval()
    chosen_tensor = torch.full((full_dataset_size,), float("-inf"), dtype=torch.float32, device=device)
    rejected_tensor = torch.full((full_dataset_size,), float("-inf"), dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not show_progress, desc="Caching reference logprobs"):
            if use_lora:
                with disable_adapter_context():
                    chosen_logps, rejected_logps, _ = forward_fn(model, batch, average_log_prob=average_log_prob)
            else:
                chosen_logps, rejected_logps, _ = forward_fn(model, batch, average_log_prob=average_log_prob)

            indices = batch["index"]
            chosen_tensor[indices] = chosen_logps
            rejected_tensor[indices] = rejected_logps

    if is_distributed():
        all_reduce_fn(chosen_tensor)
        all_reduce_fn(rejected_tensor)

    missing_chosen = torch.where(chosen_tensor == float("-inf"))[0]
    missing_rejected = torch.where(rejected_tensor == float("-inf"))[0]
    if len(missing_chosen) > 0 or len(missing_rejected) > 0:
        missing_indices = torch.unique(torch.cat([missing_chosen, missing_rejected]))
        raise RuntimeError(
            f"Missing {len(missing_indices)} indices during reference logprobs caching. "
            f"First 10: {missing_indices[:10].tolist()}"
        )

    model.train()
    cache = TensorCache(tensors={"chosen_logps": chosen_tensor, "rejected_logps": rejected_tensor})

    if cache_path is not None and is_main_process:
        logger.info(f"Saving reference logprobs cache to {cache_path}")
        cache.to_disk(cache_path)

    return cache


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    reference_free: bool = False,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model
            for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model
            for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model
            for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model
            for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something
            in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model
            and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards
            for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def wpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    chosen_loss_mask: torch.Tensor,
    rejected_loss_mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Weighted Preference Optimization (WPO) loss.
    Paper: https://arxiv.org/abs/2406.11827

    WPO extends DPO by weighting the loss based on the policy model's confidence,
    computed from the average log probabilities of chosen and rejected responses.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for chosen responses.
        policy_rejected_logps: Log probabilities of the policy model for rejected responses.
        reference_chosen_logps: Log probabilities of the reference model for chosen responses.
        reference_rejected_logps: Log probabilities of the reference model for rejected responses.
        beta: Temperature parameter for the loss.
        label_smoothing: Label smoothing parameter.
        chosen_loss_mask: Boolean mask for chosen response tokens.
        rejected_loss_mask: Boolean mask for rejected response tokens.

    Returns:
        A tuple of (losses, chosen_rewards, rejected_rewards).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # compute average logps and use them to compute the weights
    policy_chosen_logps_average = (policy_chosen_logps * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1)
    policy_rejected_logps_average = (policy_rejected_logps * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1)
    policy_weights = torch.clamp(torch.exp(policy_chosen_logps_average + policy_rejected_logps_average), max=1)

    logits = pi_logratios - ref_logratios

    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing) * policy_weights
        - F.logsigmoid(-beta * logits) * label_smoothing * policy_weights
    )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


# From https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py#L560C1-L595C56
def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float,
    gamma_beta_ratio: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the SimPO loss for a batch of policy model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the SimPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    logits = pi_logratios - gamma_beta_ratio

    # sigmoid loss type from SimPO.
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * policy_chosen_logps.detach()
    rejected_rewards = beta * policy_rejected_logps.detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, average_log_prob: bool = False) -> torch.Tensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized).
            Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities.
            Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token.
            Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum
            log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = log_softmax_and_gather(logits, labels)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def process_batch(batch: dict[str, list | torch.Tensor], prefix: str, pad_value: int = 0) -> dict[str, torch.Tensor]:
    """Process either chosen or rejected inputs separately.

    Args:
        batch: Input batch dictionary
        prefix: Either 'chosen' or 'rejected'
        pad_value: Value to use for padding (0 for input_ids, -100 for labels)

    Returns:
        Processed batch dictionary for the specified prefix
    """
    processed = {}
    for k in batch:
        if k.startswith(prefix) and isinstance(batch[k], torch.Tensor):
            new_key = k.replace(prefix + "_", "")
            processed[new_key] = batch[k]
    return processed


def concatenated_inputs(batch: dict[str, list | torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids'
            and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    chosen_input_ids: torch.Tensor = batch["chosen_input_ids"]  # type: ignore[assignment]
    rejected_input_ids: torch.Tensor = batch["rejected_input_ids"]  # type: ignore[assignment]
    max_length = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
    concatenated_batch: dict[str, torch.Tensor] = {}
    for k in batch:
        v = batch[k]
        if k.startswith("chosen") and isinstance(v, torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(v, max_length, pad_value=pad_value)
    for k in batch:
        v = batch[k]
        if k.startswith("rejected") and isinstance(v, torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (concatenated_batch[concatenated_key], pad_to_length(v, max_length, pad_value=pad_value)), dim=0
            )
    return concatenated_batch


def concatenated_forward(
    model: nn.Module,
    batch: dict[str, list | torch.Tensor],
    average_log_prob: bool = False,
    output_router_logits: bool = False,
    packing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    Uses HuggingFace model interface: model(**inputs) returns outputs with .logits attribute.

    Args:
        model: The model to run (HuggingFace-style model).
        batch: Dictionary containing chosen and rejected inputs.
        average_log_prob: Whether to average the log probabilities.
        output_router_logits: Whether to output router logits for MoE models.
        packing: Whether to use padding-free packing.

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss).
    """
    if not packing:
        concatenated_batch = concatenated_inputs(batch)
    else:
        concatenated_batch, bs = pf_concatenated_inputs(batch)

    inputs = {
        k.replace("concatenated_", ""): v
        for k, v in concatenated_batch.items()
        if k.startswith("concatenated_") and not k.endswith("labels")
    }
    if output_router_logits:
        outputs = model(**inputs, output_router_logits=True)
        logits = outputs.logits.to(torch.float32)
        aux_loss = outputs.aux_loss
    else:
        logits = model(**inputs).logits.to(torch.float32)
        aux_loss = None

    if not packing:
        all_logps = _get_batch_logps(
            logits, concatenated_batch["concatenated_labels"], average_log_prob=average_log_prob
        )
        bs = batch["chosen_input_ids"].shape[0]  # type: ignore[union-attr]
    else:
        all_logps = pf_get_batch_logps(
            logits,
            concatenated_batch["concatenated_labels"],
            inputs["cu_seq_lens_k"],
            average_log_prob=average_log_prob,
        )
    chosen_logps = all_logps[:bs]
    rejected_logps = all_logps[bs:]
    return chosen_logps, rejected_logps, aux_loss


def separate_forward(
    model: nn.Module,
    batch: dict[str, list | torch.Tensor],
    average_log_prob: bool = False,
    output_router_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the model on chosen and rejected inputs separately.

    Uses HuggingFace model interface: model(**inputs) returns outputs with .logits attribute.

    Args:
        model: The model to run (HuggingFace-style model).
        batch: Dictionary containing chosen and rejected inputs.
        average_log_prob: Whether to average the log probabilities.
        output_router_logits: Whether to output router logits for MoE models.

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss).
    """
    chosen_batch = process_batch(batch, "chosen")

    if output_router_logits:
        chosen_outputs = model(
            input_ids=chosen_batch["input_ids"],
            attention_mask=chosen_batch["attention_mask"],
            output_router_logits=True,
        )
        chosen_logits = chosen_outputs.logits.to(torch.float32)
        chosen_aux_loss = chosen_outputs.aux_loss
    else:
        chosen_logits = model(
            input_ids=chosen_batch["input_ids"], attention_mask=chosen_batch["attention_mask"]
        ).logits.to(torch.float32)
        chosen_aux_loss = None

    chosen_logps = _get_batch_logps(chosen_logits, chosen_batch["labels"], average_log_prob=average_log_prob)
    del chosen_batch, chosen_logits
    if output_router_logits:
        del chosen_outputs
    torch.cuda.empty_cache()

    rejected_batch = process_batch(batch, "rejected")

    if output_router_logits:
        rejected_outputs = model(
            input_ids=rejected_batch["input_ids"],
            attention_mask=rejected_batch["attention_mask"],
            output_router_logits=True,
        )
        rejected_logits = rejected_outputs.logits.to(torch.float32)
        rejected_aux_loss = rejected_outputs.aux_loss
    else:
        rejected_logits = model(
            input_ids=rejected_batch["input_ids"], attention_mask=rejected_batch["attention_mask"]
        ).logits.to(torch.float32)
        rejected_aux_loss = None

    rejected_logps = _get_batch_logps(rejected_logits, rejected_batch["labels"], average_log_prob=average_log_prob)
    del rejected_batch, rejected_logits
    if output_router_logits:
        del rejected_outputs
    torch.cuda.empty_cache()

    if output_router_logits and chosen_aux_loss is not None and rejected_aux_loss is not None:
        aux_loss = torch.cat([chosen_aux_loss, rejected_aux_loss], dim=0)
    else:
        aux_loss = None
    return chosen_logps, rejected_logps, aux_loss


def concatenated_forward_olmo(
    model: nn.Module,
    batch: dict[str, list | torch.Tensor],
    average_log_prob: bool = False,
    packing: bool = False,
    output_router_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    Uses OLMo-core Transformer interface: model(input_ids) returns logits tensor directly.

    Args:
        model: The model to run (OLMo-core style model).
        batch: Dictionary containing chosen and rejected inputs.
        average_log_prob: Whether to average the log probabilities.
        packing: Whether to use padding-free packing.
        output_router_logits: Unused for OLMo-core models (MoE aux loss handled separately).

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss). aux_loss is always None for OLMo-core.
    """
    del output_router_logits
    if not packing:
        concatenated_batch = concatenated_inputs(batch)
    else:
        concatenated_batch, bs = pf_concatenated_inputs(batch)

    logits = model(concatenated_batch["concatenated_input_ids"]).to(torch.float32)

    if not packing:
        all_logps = _get_batch_logps(
            logits, concatenated_batch["concatenated_labels"], average_log_prob=average_log_prob
        )
        bs = batch["chosen_input_ids"].shape[0]  # type: ignore[union-attr]
    else:
        all_logps = pf_get_batch_logps(
            logits,
            concatenated_batch["concatenated_labels"],
            concatenated_batch["concatenated_cu_seq_lens"],
            average_log_prob=average_log_prob,
        )
    chosen_logps = all_logps[:bs]
    rejected_logps = all_logps[bs:]
    return chosen_logps, rejected_logps, None


def separate_forward_olmo(
    model: nn.Module,
    batch: dict[str, list | torch.Tensor],
    average_log_prob: bool = False,
    output_router_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the model on chosen and rejected inputs separately.

    Uses OLMo-core Transformer interface: model(input_ids) returns logits tensor directly.
    Note: OLMo-core handles MoE aux loss via compute_auxiliary_metrics() in the train module.

    Args:
        model: The model to run (OLMo-core style model).
        batch: Dictionary containing chosen and rejected inputs.
        average_log_prob: Whether to average the log probabilities.
        output_router_logits: Unused for OLMo-core models (MoE aux loss handled separately).

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss). aux_loss is always None for OLMo-core.
    """
    del output_router_logits
    chosen_batch = process_batch(batch, "chosen")
    chosen_logits = model(chosen_batch["input_ids"]).to(torch.float32)

    chosen_logps = _get_batch_logps(chosen_logits, chosen_batch["labels"], average_log_prob=average_log_prob)
    del chosen_batch, chosen_logits
    torch.cuda.empty_cache()

    rejected_batch = process_batch(batch, "rejected")
    rejected_logits = model(rejected_batch["input_ids"]).to(torch.float32)

    rejected_logps = _get_batch_logps(rejected_logits, rejected_batch["labels"], average_log_prob=average_log_prob)
    del rejected_batch, rejected_logits
    torch.cuda.empty_cache()

    return chosen_logps, rejected_logps, None


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int | float, dim: int = -1) -> torch.Tensor:
    """Pad a tensor to a specified length along a given dimension.

    Args:
        tensor: The input tensor to pad.
        length: The target length for the specified dimension.
        pad_value: The value to use for padding.
        dim: The dimension along which to pad.

    Returns:
        The padded tensor, or the original tensor if already at least the target length.
    """
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
        )


@dataclass
class DataCollatorForSeq2SeqDPO(DataCollatorForSeq2Seq):
    """
    Alternate version of the hf DataCollatorForSeq2Seq for use with DPO.
    adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L517C1
    """

    def __call__(self, features, return_tensors=None):
        # call the original collator on chosen and rejected separately, then combine
        def filter_batch(match_string, features):
            return [{k.replace(match_string, ""): v for k, v in f.items() if match_string in k} for f in features]

        chosen_features = super().__call__(filter_batch("chosen_", features), return_tensors=return_tensors)
        rejected_features = super().__call__(filter_batch("rejected_", features), return_tensors=return_tensors)
        result = {}
        for k in chosen_features:
            result["chosen_" + k] = chosen_features[k]
        for k in rejected_features:
            result["rejected_" + k] = rejected_features[k]
        if "index" in features[0]:
            result["index"] = torch.tensor([f["index"] for f in features])
        max_len = max(result["chosen_input_ids"].shape[1], result["rejected_input_ids"].shape[1])
        chosen_padded = torch.nn.functional.pad(
            result["chosen_input_ids"],
            (0, max_len - result["chosen_input_ids"].shape[1]),
            value=self.tokenizer.pad_token_id,
        )
        rejected_padded = torch.nn.functional.pad(
            result["rejected_input_ids"],
            (0, max_len - result["rejected_input_ids"].shape[1]),
            value=self.tokenizer.pad_token_id,
        )
        result["input_ids"] = torch.cat([chosen_padded, rejected_padded], dim=0)
        return result
