from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers import DefaultDataCollator

from open_instruct import logger_utils, tensor_utils

logger = logger_utils.setup_logger(__name__)


@dataclass
class PackedTensors:
    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor | None
    seq_idx: torch.Tensor | None
    cu_seq_lens: torch.Tensor | None
    num_valid_seqs: int
    sequences_dropped: int


def _fit_to_max_length(
    max_seq_length: int,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    cu_seq_lens: torch.Tensor | None,
    num_features: int,
) -> PackedTensors:
    current_len = input_ids.shape[1]

    if current_len > max_seq_length:
        input_ids = input_ids[:, :max_seq_length]
        labels = labels[:, :max_seq_length]
        if position_ids is not None:
            position_ids = position_ids[:, :max_seq_length]
        if seq_idx is not None:
            seq_idx = seq_idx[:, :max_seq_length]

        num_valid_seqs = num_features
        sequences_dropped = 0
        if cu_seq_lens is not None:
            cu_seq_lens = torch.unique_consecutive(cu_seq_lens.clamp_max_(max_seq_length))
            num_valid_seqs = len(cu_seq_lens) - 1
            sequences_dropped = num_features - num_valid_seqs
            if sequences_dropped > 0:
                logger.warning(
                    f"Truncation dropped {sequences_dropped} sequences "
                    f"(packed {num_features} sequences into {current_len} tokens, "
                    f"truncated to {max_seq_length}, {num_valid_seqs} sequences remain)"
                )

        return PackedTensors(input_ids, labels, position_ids, seq_idx, cu_seq_lens, num_valid_seqs, sequences_dropped)

    if current_len < max_seq_length:
        input_ids = tensor_utils.pad_to_length(input_ids, max_seq_length, pad_value=0)
        labels = tensor_utils.pad_to_length(labels, max_seq_length, pad_value=-100)
        if position_ids is not None:
            position_ids = tensor_utils.pad_to_length(position_ids, max_seq_length, pad_value=0)
        if seq_idx is not None:
            seq_idx = tensor_utils.pad_to_length(seq_idx, max_seq_length, pad_value=-1)

    return PackedTensors(input_ids, labels, position_ids, seq_idx, cu_seq_lens, num_features, 0)


def _collect_flattened_features(
    features: list[dict[str, Any]],
    separator_id: int,
    return_flash_attn_kwargs: bool,
    return_position_ids: bool,
    return_seq_idx: bool,
) -> tuple[dict[str, Any], list[torch.Tensor] | None, list[torch.Tensor] | None, list[int] | None, int]:
    """
    Flatten a list of example dicts into concatenated tensors plus optional
    metadata used for padding-free training.
    """
    is_labels_provided = "labels" in features[0]
    ret: dict[str, Any] = {"input_ids": [], "labels": []}
    pos_ids: list[torch.Tensor] | None = [] if return_position_ids else None
    seq_idx: list[torch.Tensor] | None = [] if return_seq_idx else None
    cu_seq_lens: list[int] | None = [0] if return_flash_attn_kwargs else None
    max_length = 0

    separator = torch.tensor(
        [separator_id], dtype=features[0]["input_ids"].dtype, device=features[0]["input_ids"].device
    )
    for s_idx, item in enumerate(features):
        input_ids = item["input_ids"]
        ret["input_ids"].append(input_ids)

        # Labels are next-token shifted: insert a separator, then drop the first token.
        label_source = item["labels"] if is_labels_provided else input_ids
        ret["labels"].append(separator)
        ret["labels"].append(label_source[1:])

        if return_flash_attn_kwargs and cu_seq_lens is not None:
            cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
            max_length = max(max_length, len(input_ids))
        if return_position_ids and pos_ids is not None:
            pos_ids.append(torch.arange(input_ids.numel(), device=input_ids.device))
        if return_seq_idx and seq_idx is not None:
            seq_idx.append(torch.full_like(input_ids, s_idx, dtype=torch.int32))

    return ret, pos_ids, seq_idx, cu_seq_lens, max_length


def _filter_feature_dicts(features: list[dict[str, Any]], match_string: str) -> list[dict[str, Any]]:
    return [{k.replace(match_string, ""): v for k, v in f.items() if match_string in k} for f in features]


def _split_prefixed_batch(batch: dict[str, list | torch.Tensor]) -> tuple[dict[str, Any], dict[str, Any]]:
    chosen_features: dict[str, Any] = {}
    rejected_features: dict[str, Any] = {}
    for k in batch:
        if k.startswith("chosen_"):
            chosen_features[k.removeprefix("chosen_")] = batch[k]
        elif k.startswith("rejected_"):
            rejected_features[k.removeprefix("rejected_")] = batch[k]
    return chosen_features, rejected_features


@dataclass
class TensorDataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator for padding-free training along the lines of https://huggingface.co/blog/packing-with-FA2

    Eliminates use of padding which is generically needed for per_gpu_batch_size > 1, thereby
    reducing memory costs and increasing throughput. Your model class must support padding-free
    training to use this collator correctly. Examples which support padding free include
    LlamaForCausalLM and BambaForCausalLM.

    The `input_ids` and `labels` from separate examples are concatenated together into a tensor of
    batch size 1, with additional information included in the batch to demarcate example boundaries.

    `cu_seq_lens` (cumulative sequence lengths) stores the boundary offsets of each packed sequence,
    including a leading 0. For example, 3 sequences of lengths 5, 3, 7 give cu_seq_lens = [0, 5, 8, 15],
    so len(cu_seq_lens) == num_seqs + 1.
    """

    return_flash_attn_kwargs: bool = True
    return_position_ids: bool = True
    return_seq_idx: bool = True
    separator_id: int = -100
    max_seq_length: int | None = None

    def __call__(self, features, return_tensors=None, separator_id=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id

        ret, pos_ids, seq_idx, cu_seq_lens, max_length = _collect_flattened_features(
            features=features,
            separator_id=separator_id,
            return_flash_attn_kwargs=self.return_flash_attn_kwargs,
            return_position_ids=self.return_position_ids,
            return_seq_idx=self.return_seq_idx,
        )

        if self.return_flash_attn_kwargs and cu_seq_lens is not None:
            ret["cu_seq_lens_q"] = ret["cu_seq_lens_k"] = torch.tensor(
                cu_seq_lens, dtype=torch.int32, device=features[0]["input_ids"].device
            )
            ret["max_length_q"] = ret["max_length_k"] = max_length
        position_ids_tensor = None
        seq_idx_tensor = None
        if self.return_position_ids and pos_ids is not None:
            position_ids_tensor = torch.cat(pos_ids, dim=0)[None]
        if self.return_seq_idx and seq_idx is not None:
            seq_idx_tensor = torch.cat(seq_idx, dim=0)[None]
        input_ids_tensor = torch.cat(ret["input_ids"], dim=0)[None]
        labels_tensor = torch.cat(ret["labels"], dim=0)[None]

        num_valid_seqs = len(features)
        sequences_dropped = 0
        if self.max_seq_length is not None:
            cu_seq_lens_tensor = cast(torch.Tensor, ret["cu_seq_lens_q"]) if self.return_flash_attn_kwargs else None
            packed = _fit_to_max_length(
                self.max_seq_length,
                input_ids_tensor,
                labels_tensor,
                position_ids_tensor,
                seq_idx_tensor,
                cu_seq_lens_tensor,
                len(features),
            )
            input_ids_tensor, labels_tensor = packed.input_ids, packed.labels
            position_ids_tensor, seq_idx_tensor = packed.position_ids, packed.seq_idx
            num_valid_seqs, sequences_dropped = packed.num_valid_seqs, packed.sequences_dropped
            if packed.cu_seq_lens is not None:
                ret["cu_seq_lens_q"] = ret["cu_seq_lens_k"] = packed.cu_seq_lens

        ret["input_ids"] = input_ids_tensor
        ret["labels"] = labels_tensor
        ret["_num_valid_seqs"] = num_valid_seqs
        ret["_sequences_dropped"] = sequences_dropped
        if position_ids_tensor is not None:
            ret["position_ids"] = position_ids_tensor
        if seq_idx_tensor is not None:
            ret["seq_idx"] = seq_idx_tensor
        return ret


def _align_dpo_seq_counts(chosen_features: dict[str, Any], rejected_features: dict[str, Any]) -> tuple[int, int]:
    chosen_valid = chosen_features.pop("_num_valid_seqs")
    rejected_valid = rejected_features.pop("_num_valid_seqs")
    chosen_dropped = chosen_features.pop("_sequences_dropped")
    rejected_dropped = rejected_features.pop("_sequences_dropped")
    num_valid = min(chosen_valid, rejected_valid)

    if num_valid < chosen_valid and "cu_seq_lens_q" in chosen_features:
        chosen_features["cu_seq_lens_q"] = chosen_features["cu_seq_lens_q"][: num_valid + 1]
        chosen_features["cu_seq_lens_k"] = chosen_features["cu_seq_lens_k"][: num_valid + 1]
    if num_valid < rejected_valid and "cu_seq_lens_q" in rejected_features:
        rejected_features["cu_seq_lens_q"] = rejected_features["cu_seq_lens_q"][: num_valid + 1]
        rejected_features["cu_seq_lens_k"] = rejected_features["cu_seq_lens_k"][: num_valid + 1]

    return num_valid, max(chosen_dropped, rejected_dropped)


@dataclass
class TensorDataCollatorWithFlatteningDPO(TensorDataCollatorWithFlattening):
    def __call__(self, features, return_tensors=None, separator_id=None):
        # call the original collator on chosen and rejected separately, then combine
        chosen_features = super().__call__(
            _filter_feature_dicts(features, "chosen_"), return_tensors=return_tensors, separator_id=separator_id
        )
        rejected_features = super().__call__(
            _filter_feature_dicts(features, "rejected_"), return_tensors=return_tensors, separator_id=separator_id
        )

        num_valid, sequences_dropped = _align_dpo_seq_counts(chosen_features, rejected_features)

        result = {"_sequences_dropped": sequences_dropped, "_num_valid_seqs": num_valid}
        for k in chosen_features:
            result["chosen_" + k] = chosen_features[k]
        for k in rejected_features:
            result["rejected_" + k] = rejected_features[k]
        if "index" in features[0]:
            all_indices = torch.tensor([f["index"] for f in features])
            result["index"] = all_indices[:num_valid]
        return result


def concatenated_inputs(
    batch: dict[str, list | torch.Tensor], tag: str = "concatenated_"
) -> tuple[dict[str, torch.Tensor], int]:
    chosen_features, rejected_features = _split_prefixed_batch(batch)

    ret = {f"{tag}input_ids": torch.cat([chosen_features["input_ids"], rejected_features["input_ids"]], dim=-1)}
    if "labels" in chosen_features:
        ret[f"{tag}labels"] = torch.cat([chosen_features["labels"], rejected_features["labels"]], dim=-1)

    if "cu_seq_lens_q" in chosen_features:
        # Skip rejected's leading 0 to avoid a duplicate boundary, and offset by chosen length.
        chosen_input_len = chosen_features["input_ids"].shape[-1]
        ret[f"{tag}cu_seq_lens_q"] = torch.cat(
            [chosen_features["cu_seq_lens_q"], rejected_features["cu_seq_lens_q"][1:] + chosen_input_len]
        )
        ret[f"{tag}cu_seq_lens_k"] = torch.cat(
            [chosen_features["cu_seq_lens_k"], rejected_features["cu_seq_lens_k"][1:] + chosen_input_len]
        )
        ret[f"{tag}max_length_q"] = max(chosen_features["max_length_q"], rejected_features["max_length_q"])
        ret[f"{tag}max_length_k"] = max(chosen_features["max_length_k"], rejected_features["max_length_k"])

    if "position_ids" in chosen_features:
        ret[f"{tag}position_ids"] = torch.cat(
            [chosen_features["position_ids"], rejected_features["position_ids"]], dim=-1
        )

    if "seq_idx" in chosen_features:
        # cu_seq_lens has num_seqs + 1 elements (includes leading 0), so subtract 1.
        chosen_num_seqs = len(chosen_features["cu_seq_lens_k"]) - 1
        ret[f"{tag}seq_idx"] = torch.cat(
            [chosen_features["seq_idx"], rejected_features["seq_idx"] + chosen_num_seqs], dim=-1
        )

    return ret, len(chosen_features["cu_seq_lens_k"]) - 1


def get_batch_logps(
    logits: torch.Tensor, labels: torch.Tensor, cu_seq_lens: torch.Tensor, average_log_prob: bool = False
) -> torch.Tensor:
    assert logits.shape[:-1] == labels.shape

    # - we are going to get crossings at labels / logits
    #   cont batch boundaries, but we assume that the
    #   loss mask == True at those places
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels = labels.masked_fill(~loss_mask, 0)

    # Compensate for the next-token shift above: each boundary moved left by 1,
    # but the leading 0 must stay at 0.
    cu_seq_lens = cu_seq_lens.clone() - 1
    cu_seq_lens[0] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    num_seqs = cu_seq_lens.shape[0] - 1
    seq_len = per_token_logps.shape[1]
    # Map each token position to its segment index using cu_seq_lens boundaries.
    positions = torch.arange(seq_len, device=cu_seq_lens.device)
    segment_ids = torch.bucketize(positions, cu_seq_lens[1:], right=True).clamp(max=num_seqs - 1)

    masked_logps = (per_token_logps * loss_mask.float()).squeeze(0)
    mask_float = loss_mask.float().squeeze(0)

    segment_sums = torch.zeros(num_seqs, device=masked_logps.device, dtype=masked_logps.dtype)
    segment_sums.scatter_add_(0, segment_ids, masked_logps)

    if average_log_prob:
        segment_counts = torch.zeros(num_seqs, device=mask_float.device, dtype=mask_float.dtype)
        segment_counts.scatter_add_(0, segment_ids, mask_float)
        return segment_sums / segment_counts.clamp(min=1)
    return segment_sums
