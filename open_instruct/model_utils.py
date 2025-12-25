# Taken and modified from https://github.com/huggingface/trl
# Copyright 2022 The HuggingFace Team. All rights reserved.
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


import asyncio
import itertools
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Union

import deepspeed
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from deepspeed.runtime.engine import DeepSpeedEngine
from huggingface_hub import HfApi
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.nn.parallel.distributed import DistributedDataParallel

from open_instruct import logger_utils
from open_instruct.ground_truth_utils import VerifierFunction
from open_instruct.utils import retry_on_exception

logger = logger_utils.setup_logger(__name__)


@dataclass
class Batch:
    """Container for batch data including queries, ground truths, and datasets."""

    queries: list[list[int]]
    ground_truths: list[list[int]]
    datasets: list[str]
    raw_queries: list[str] | None
    decoded_responses: list[str] | None
    indices: list[int] | None
    scores: list[float] | None

    def __getitem__(self, key: slice | int | list[int]) -> "Batch":
        """Enable indexing and slicing: batch[5], batch[start:end], or batch[[1,3,5]]."""
        if isinstance(key, slice):
            # Handle slice object: batch[start:end]
            return Batch(
                queries=self.queries[key],
                ground_truths=self.ground_truths[key],
                datasets=self.datasets[key],
                raw_queries=self.raw_queries[key] if self.raw_queries is not None else None,
                decoded_responses=self.decoded_responses[key] if self.decoded_responses is not None else None,
                indices=self.indices[key] if self.indices is not None else None,
                scores=self.scores[key] if self.scores is not None else None,
            )
        elif isinstance(key, int):
            # Handle single index: batch[5]
            return Batch(
                queries=[self.queries[key]],
                ground_truths=[self.ground_truths[key]],
                datasets=[self.datasets[key]],
                raw_queries=[self.raw_queries[key]] if self.raw_queries is not None else None,
                decoded_responses=[self.decoded_responses[key]] if self.decoded_responses is not None else None,
                indices=[self.indices[key]] if self.indices is not None else None,
                scores=[self.scores[key]] if self.scores is not None else None,
            )
        else:
            # Handle list of indices: batch[[1,3,5]]
            return Batch(
                queries=[self.queries[i] for i in key],
                ground_truths=[self.ground_truths[i] for i in key],
                datasets=[self.datasets[i] for i in key],
                raw_queries=[self.raw_queries[i] for i in key] if self.raw_queries is not None else None,
                decoded_responses=[self.decoded_responses[i] for i in key]
                if self.decoded_responses is not None
                else None,
                indices=[self.indices[i] for i in key] if self.indices is not None else None,
                scores=[self.scores[i] for i in key] if self.scores is not None else None,
            )


@dataclass
class ModelConfig:
    model_name_or_path: str | None = None
    """The model checkpoint for weights initialization."""
    model_revision: str | None = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    dtype: str | None = None
    """The data type to load the model under. If specified, overrides the default `torch.dtype`."""
    attn_implementation: Literal["flash_attention_2"] | None = None
    """Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case
    you must install this manually by running `pip install flash-attn --no-build-isolation`"""
    use_cache: bool | None = None
    """Whether to use cache in the model."""
    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing in the model."""

    # PEFT-related args
    use_peft: bool = False
    """Whether to use PEFT or not for training."""
    lora_r: int | None = 16
    """LoRA R value."""
    lora_alpha: int | None = 32
    """LoRA alpha."""
    lora_dropout: float | None = 0.05
    """LoRA dropout."""
    lora_target_modules: list[str] | None = None
    """LoRA target modules."""
    lora_modules_to_save: list[str] | None = None
    """Model layers to unfreeze & train"""
    lora_task_type: str = "CAUSAL_LM"
    """The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"""

    # quantization args
    load_in_8bit: bool = False
    """use 8 bit precision for the base model - works only with LoRA"""
    load_in_4bit: bool = False
    """use 4 bit precision for the base model - works only with LoRA"""
    bnb_4bit_quant_type: str | None = "nf4"
    """precise the quantization type (fp4 or nf4)"""
    use_bnb_nested_quant: bool = False
    """use nested quantization"""

    def __post_init__(self):
        # `use_cache=True` is incompatible with gradient checkpointing.
        # https://github.com/huggingface/transformers/blob/d6751d91c8f58cdeb35af6adae182d7dc90aa883/src/transformers/models/llama/modeling_llama.py#L945
        if self.gradient_checkpointing:
            self.use_cache = False


# ----------------------------------------------------------------------------
# Model utilities; reward model stuff
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def maybe_load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device, rank: int, throw_on_error: bool = True
) -> None:
    """Load a checkpoint into a model, with optional error handling.

    Args:
        model: The model to load the checkpoint into.
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the checkpoint onto.
        rank: Global process rank for logging.
        throw_on_error: if true, throw an error if the checkpoint fails to load.
    """

    def _load_checkpoint(path: str, dev: torch.device):
        state_dict = torch.load(path, map_location=dev)
        if hasattr(model, "module"):
            # Needed if wrapped by DeepSpeed.
            model.module.load_state_dict(state_dict)
        else:
            # If a vanilla HF model.
            model.load_state_dict(state_dict)
        logger.info(f"{rank=}: Loaded checkpoint from {path}")

    if not throw_on_error:
        try:
            _load_checkpoint(checkpoint_path, device)
        except Exception as e:
            logger.error(
                f"{rank=}: Falling back to using base reference, Failed to load checkpoint from {checkpoint_path}: {e}"
            )
    else:
        _load_checkpoint(checkpoint_path, device)


def load_ref_policy(
    model_config: ModelConfig,
    ds_config: dict,
    deepspeed_stage: int,
    local_rank: int,
    device: torch.device,
    rank: int,
    checkpoint_path: str | None = None,
    ref_policy_update_freq: int | None = None,
    alpha: float = 0.0,
) -> transformers.PreTrainedModel:
    """Loads a reference policy model for evaluation.

    Args:
        model_config: Configuration containing model name and revision.
        ds_config: DeepSpeed configuration dictionary.
        deepspeed_stage: DeepSpeed ZeRO stage.
        local_rank: Local GPU rank for device mapping.
        device: Target device for loading checkpoint.
        rank: Global process rank for logging.
        checkpoint_path: Optional path to model checkpoint to load.
        ref_policy_update_freq: Frequency of reference policy updates. If None, no updates occur.
        alpha: Alpha value for polyak updates. If 0, no updates occur.

    Returns:
        Initialized reference policy model in evaluation mode.
    """
    # inference model only has stage 3 (sharding) or stage 0 (no sharding)
    # stage 2 is optimizer sharding which doesn't apply to inference
    ref_policy: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
        **({"device_map": {"": local_rank}} if deepspeed_stage != 3 else {}),
    )
    disable_dropout_in_model(ref_policy)
    ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=ds_config)
    ref_policy.eval()

    if checkpoint_path:
        # throw an error if we fail to load AND we are updating the reference.
        maybe_load_checkpoint(
            model=ref_policy,
            checkpoint_path=checkpoint_path,
            device=device,
            rank=rank,
            throw_on_error=ref_policy_update_freq is not None and alpha != 0,
        )
    return ref_policy


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of the logits.
    Borrowed from verl (https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py#L145)
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Finds the index of the first `True` value in each row of a boolean tensor. If no `True` value exists in a row,
    it returns the length of the row.

    Args:
        bools (torch.Tensor): A boolean tensor of shape (batch_size, sequence_length), where `True` values indicate
                              the positions of interest.
        dtype (torch.dtype): The data type to use for the output indices (default is torch.long).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the index of the first `True` value in each row.
                      If a row has no `True` value, the index will be the length of the row.
    """

    # Get the length of each row (i.e., the number of columns in the last dimension)
    # row_len is a scalar representing the length of each sequence (sequence_length)
    row_len = bools.size(-1)

    # Calculate the index positions for the first `True` in each row
    # ~bools: Invert the boolean values (True becomes False and vice versa)
    # ~bools.type(dtype): Convert the inverted boolean tensor to the specified dtype (0 for True, 1 for False)
    # row_len * (~bools).type(dtype): For `False` values, this will give `row_len`, for `True` values it gives 0.
    # torch.arange(row_len, dtype=dtype, device=bools.device): Generates a tensor with values [0, 1, 2, ..., row_len-1]
    # for each row. Shape: (sequence_length,)
    # zero_or_index: Shape (batch_size, sequence_length). This tensor contains the indices for `True` values and `row_len`
    # for `False` values.
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)

    # Return the minimum value in each row (i.e., the first `True` index or `row_len` if none exist)
    # torch.min(zero_or_index, dim=-1).values: This returns the minimum value in each row, which corresponds to the first
    # `True` value's index or `row_len` if there is no `True` in that row.
    # The returned tensor has shape (batch_size,)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function computes reward scores for a batch of query responses based on a pre-trained reward model.

    Args:
        model (torch.nn.Module): The pre-trained reward model.
        query_responses (torch.Tensor): Tensor containing the tokenized responses for which to compute rewards.
            Shape: (batch_size, sequence_length)
        pad_token_id (int): The ID used for padding tokens in the tokenized sequences.
        context_length (int): The length of the prompt or context preceding the completions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - reward_logits: The logits output from the model for all tokens in the sequences.
              Shape: (batch_size, sequence_length)
            - final_scores: The final reward scores, one for each sequence, after adjusting for sequence lengths.
              Shape: (batch_size,)
            - sequence_lengths: The lengths of each sequence (excluding padding).
              Shape: (batch_size,)
    """

    # Create an attention mask where tokens that are not padding have a value of 1, and padding tokens have a value of 0
    # Shape: (batch_size, sequence_length)
    attention_mask = query_responses != pad_token_id

    # Calculate position IDs for each token, considering the cumulative sum of the attention mask (to exclude padding)
    # Shape: (batch_size, sequence_length)
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum

    # Access the LM backbone from the reward model using its base model prefix
    lm_backbone = getattr(model, model.base_model_prefix)

    # Replace padding tokens with zeros in the input IDs (so padding tokens won't affect the model's processing)
    # Shape: (batch_size, sequence_length)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])  # (batch_size, sequence_length)

    # Calculate the length of each sequence by finding the first occurrence of a padding token after the context
    # sequence_lengths shape: (batch_size,)
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    assert reward_logits.shape[-1] == 1, (
        "Reward model should output a single scalar per token. Check if you added `num_labels=1` when doing `AutoModelForSequenceClassification.from_pretrained(...)`."
    )
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454

    # Return the reward logits for all tokens, the final reward scores for each sequence, and the sequence lengths
    return (
        # reward_logits shape: (batch_size, sequence_length)
        reward_logits,
        # final_scores shape: (batch_size,)
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(
            -1
        ),  # Shape: (batch_size,)
        sequence_lengths,
    )


async def apply_verifiable_reward(
    reward_fn_mapping: dict[str, VerifierFunction],
    responses: list[torch.Tensor],
    decoded_responses: list[str],
    ground_truths: list[float],
    datasets: list[str],
    reward_mult: int = 10,
    queries: list[str] | None = None,
):
    if queries is None:
        queries = [None] * len(responses)

    # Collect all async tasks for parallel execution
    async_tasks = []
    task_metadata = []

    for i, (tok_prediction, prediction, ground_truth, dataset, query) in enumerate(
        zip(responses, decoded_responses, ground_truths, datasets, queries)
    ):
        # allow multiple ground truths and datasets for a single response

        # TODO: both code and lm_judge might have list of ground_truth *per instance*
        ground_truth_list = [ground_truth] if isinstance(ground_truth, str) else ground_truth
        dataset_list = [dataset] if isinstance(dataset, str) else dataset
        assert len(ground_truth_list) == len(dataset_list), "Ground truth and dataset list lengths do not match."

        # Create async tasks for each ground truth/dataset pair
        for gt, ds in zip(ground_truth_list, dataset_list):
            reward_func = reward_fn_mapping.get(ds.lower())
            if reward_func is None:
                logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
                continue

            # Create async task
            task = reward_func.async_call(
                tokenized_prediction=tok_prediction, prediction=prediction, label=gt, query=query
            )
            async_tasks.append(task)
            # use reward_func.name to get the name of the verifier, rather than ds in case we have done remapping.
            task_metadata.append(
                {
                    "response_idx": i,
                    "dataset": reward_func.name,
                    "reward_weight": reward_func.weight,
                    "reward_mult": reward_mult,
                }
            )

    # Execute all tasks in parallel
    if async_tasks:
        reward_results = await asyncio.gather(*async_tasks)
        logger.debug(f"Applied {len(reward_results)} ground truth rewards in parallel 🤗")
    else:
        reward_results = []

    # Initialize results for each response
    response_rewards = [0] * len(responses)
    response_per_func_rewards = [{} for _ in range(len(responses))]

    # Process results
    for result, metadata in zip(reward_results, task_metadata):
        response_idx = metadata["response_idx"]
        dataset = metadata["dataset"]
        reward_weight = metadata["reward_weight"]
        reward_mult = metadata["reward_mult"]

        # Extract score from VerificationResult
        score = result.score if hasattr(result, "score") else result
        weighted_reward = reward_mult * score * reward_weight

        response_rewards[response_idx] += weighted_reward
        response_per_func_rewards[response_idx][dataset] = (
            response_per_func_rewards[response_idx].get(dataset, 0) + weighted_reward
        )

    return response_rewards, response_per_func_rewards


def get_olmo3_generation_config(tokenizer):
    return transformers.GenerationConfig(
        temperature=None,
        top_p=None,
        eos_token_id=[tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")],
    )


def save_with_accelerate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    output_dir: str,
    use_lora: bool = False,
    model_attribute_to_save: str | None = None,
    chat_template_name: str = "tulu",
) -> None:
    """`model_attribute_to_save` is for used to save PPO's policy instead of the full model"""
    # set the generation config to an empty setting to be safe.
    # we usually do greedy decoding for generation, so this should be okay.
    # otherwise, we get an error thrown at save time.
    if chat_template_name and "olmo" in chat_template_name:
        # New chat template has no bos token, and two eos tokens: <|im_end|> and <|endoftext|>
        logger.info(f"Detected olmo chat template: {chat_template_name}, updating model generation config.")
        model.generation_config = get_olmo3_generation_config(tokenizer)
    else:
        model.generation_config = transformers.GenerationConfig(
            temperature=None, top_p=None, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id
        )

    unwrapped_model: transformers.PreTrainedModel = accelerator.unwrap_model(model)
    if model_attribute_to_save is not None:
        unwrapped_model = getattr(unwrapped_model, model_attribute_to_save)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)

    # if we are saving a specific attribute of the model, we need to filter the state_dict
    # also the state_dict only lives in the main process; other processes just have state_dict = None
    if model_attribute_to_save is not None and accelerator.is_main_process:
        state_dict = OrderedDict(
            {
                k[len(f"{model_attribute_to_save}.") :]: v
                for k, v in state_dict.items()
                if k.startswith(f"{model_attribute_to_save}.")
            }
        )

    if use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
    # customize model card (TODO (Costa): this can be prettier)


@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    torch compiled version of the common `log_softmax -> gather` operation.


    The compiled version of this opration avoids the (significant) memory overhead of
    allocating a new (batch_size, seq_len, vocab_size) tensor to store the logprobs.


    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@retry_on_exception()
def push_folder_to_hub(
    output_dir: str, hf_repo_id: str | None = None, hf_repo_revision: str | None = None, private: bool = True
):
    """Push a folder to Hugging Face Hub.

    This function should only run on the main process. Callers are expected to gate calls themselves.
    """
    api = HfApi()
    if not api.repo_exists(hf_repo_id):
        api.create_repo(hf_repo_id, exist_ok=True, private=private)
    if hf_repo_revision is not None:
        api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo_id,
        revision=hf_repo_revision,
        folder_path=output_dir,
        commit_message="upload checkpoint",
        run_as_future=False,
    )
    logger.info(f"🔥 pushed to https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}")


# ----------------------------------------------------------------------------
# DeepSpeed utilities
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]


def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"], accelerator: "Accelerator", is_peft_model: bool = False
) -> Union["transformers.PreTrainedModel", "DeepSpeedEngine"]:
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield accelerator.unwrap_model(model)
            add_hooks(model)
    else:
        yield unwrapped_model


def prepare_deepspeed(model: torch.nn.Module, per_device_train_batch_size: int, mixed_precision: str):
    """
    Prepares the model for training with DeepSpeed (both for stage 2 and 3), configuring the appropriate settings based on the model and
    batch size.
    Args:
        model (`torch.nn.Module`):
            The model to be prepared for DeepSpeed training.
        per_device_train_batch_size (`int`):
            The training batch size per device.
        mixed_precision (`str`):
            The mixed precision setting to use.
    Returns:
        `torch.nn.Module`:
            The model initialized and configured with DeepSpeed for training.
    """
    import deepspeed

    deepspeed_plugin = AcceleratorState().deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
        config_kwargs = {
            "train_micro_batch_size_per_gpu": config_kwargs["train_micro_batch_size_per_gpu"],
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        if mixed_precision in ["bf16", "fp16"]:
            config_kwargs[mixed_precision] = {"enabled": True}
    else:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0,
                    }
                )
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Quality of life utilities
def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


def format_value(value):
    if isinstance(value, float):
        if abs(value) < 1e-5:
            return f"{value:.2e}"
        return f"{value:.2f}"
    return str(value)


def print_rich_single_line_metrics(metrics):
    # Create main table
    table = Table(show_header=False, box=None)
    table.add_column("Category", style="cyan")
    table.add_column("Values", style="magenta")

    # Group metrics by their prefix
    grouped_metrics = defaultdict(list)
    for key, value in metrics.items():
        category = key.split("/")[0] if "/" in key else "other"
        grouped_metrics[category].append((key, value))

    # Sort groups by category name
    for category in sorted(grouped_metrics.keys()):
        values = grouped_metrics[category]
        value_strings = []
        for key, value in values:
            # Use the last part of the key as the display name
            display_name = key.split("/")[-1]
            value_strings.append(f"{display_name}: {format_value(value)}")

        # Join all values for this category into a single string
        values_str = " | ".join(value_strings)
        table.add_row(category, values_str)

    # Create a panel with the table
    panel = Panel(table, title="Metrics", expand=False, border_style="bold green")

    # Print the panel
    rprint(panel)


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


def estimate_kl(ref_logprobs_diff: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    """Compute 4 different KL divergence estimators between current and reference policies.

    Args:
        ref_logprobs_diff: Log probability difference (new_logprobs - ref_logprobs), clamped
            to [-40, 40] for numerical stability. Shape: [B, T] or similar.
        ratio: Importance sampling ratio exp(new_logprobs - old_logprobs) between current
            policy and the policy at the start of the training step. Shape: [B, T] or similar.

    Returns:
        Tensor of shape [4, B, T] containing 4 KL estimators stacked along dim 0:
            [0]: linear approximation (ref_logprobs_diff)
            [1]: quadratic approximation (ref_logprobs_diff^2 / 2)
            [2]: numerically stable form (expm1(-ref_logprobs_diff) + ref_logprobs_diff)
            [3]: importance-weighted (ratio * ref_logprobs_diff)

        We tend to prefer [2] as a reasonable default.
    """
    return torch.stack(
        [
            ref_logprobs_diff,
            (ref_logprobs_diff) ** 2 / 2,
            torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff,
            ratio * ref_logprobs_diff,
        ]
    )
