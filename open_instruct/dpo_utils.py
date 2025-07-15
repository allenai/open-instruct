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

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq

from open_instruct.model_utils import log_softmax_and_gather
from open_instruct.padding_free_collator import concatenated_inputs as pf_concatenated_inputs
from open_instruct.padding_free_collator import get_batch_logps as pf_get_batch_logps

torch.backends.cuda.matmul.allow_tf32 = True


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False,
    label_smoothing: float = 0.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    chosen_loss_mask: torch.BoolTensor = None,
    rejected_loss_mask: torch.BoolTensor = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    beta: float,
    gamma_beta_ratio: float,
    label_smoothing: float = 0.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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


def _get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
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


def process_batch(
    batch: Dict[str, Union[List, torch.LongTensor]], prefix: str, pad_value: int = 0
) -> Dict[str, torch.LongTensor]:
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


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids'
            and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (concatenated_batch[concatenated_key], pad_to_length(batch[k], max_length, pad_value=pad_value)), dim=0
            )
    return concatenated_batch


def concatenated_forward(
    model: nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    average_log_prob: bool = False,
    output_router_logits: bool = False,
    packing: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
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
        bs = batch["chosen_input_ids"].shape[0]
    else:
        all_logps = pf_get_batch_logps(
            logits,
            concatenated_batch["concatenated_labels"],
            inputs["cu_seq_lens_k"],  # assume same as q
            average_log_prob=average_log_prob,
        )
    chosen_logps = all_logps[:bs]
    rejected_logps = all_logps[bs:]
    return chosen_logps, rejected_logps, aux_loss


def separate_forward(
    model: nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    average_log_prob: bool = False,
    output_router_logits: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, Union[torch.FloatTensor, None]]:
    """Run the model on chosen and rejected inputs separately.

    Args:
        model: The model to run
        batch: Dictionary containing chosen and rejected inputs
        average_log_prob: Whether to average the log probabilities
        output_router_logits: Whether to output router logits for MoE models

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss)
    """
    # Process chosen inputs
    chosen_batch = process_batch(batch, "chosen")

    if output_router_logits:
        chosen_outputs = model(
            input_ids=chosen_batch["input_ids"],
            attention_mask=chosen_batch["attention_mask"],
            output_router_logits=True,
        )
        chosen_logits = chosen_outputs.logits.to(torch.float32)
        aux_loss = chosen_outputs.aux_loss
    else:
        chosen_logits = model(
            input_ids=chosen_batch["input_ids"], attention_mask=chosen_batch["attention_mask"]
        ).logits.to(torch.float32)
        aux_loss = None

    chosen_logps = _get_batch_logps(chosen_logits, chosen_batch["labels"], average_log_prob=average_log_prob)
    del chosen_batch, chosen_logits
    if output_router_logits:
        del chosen_outputs
    torch.cuda.empty_cache()

    # Process rejected inputs
    rejected_batch = process_batch(batch, "rejected")

    if output_router_logits:
        rejected_outputs = model(
            input_ids=rejected_batch["input_ids"],
            attention_mask=rejected_batch["attention_mask"],
            output_router_logits=True,
        )
        rejected_logits = rejected_outputs.logits.to(torch.float32)
        aux_loss = rejected_outputs.aux_loss
    else:
        rejected_logits = model(
            input_ids=rejected_batch["input_ids"], attention_mask=rejected_batch["attention_mask"]
        ).logits.to(torch.float32)
        aux_loss = None

    rejected_logps = _get_batch_logps(rejected_logits, rejected_batch["labels"], average_log_prob=average_log_prob)
    del rejected_batch, rejected_logits
    if output_router_logits:
        del rejected_outputs
    torch.cuda.empty_cache()
    if output_router_logits:
        aux_loss = torch.cat([chosen_outputs.aux_loss, rejected_outputs.aux_loss], dim=0)

    return chosen_logps, rejected_logps, aux_loss


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
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
        return result
