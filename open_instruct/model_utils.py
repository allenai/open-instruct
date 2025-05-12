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


import itertools
import logging
import asyncio
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

try:
    import deepspeed
    from deepspeed.runtime.engine import DeepSpeedEngine
except ImportError:
    pass
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from huggingface_hub import HfApi
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import PreTrainedModel, PreTrainedTokenizer

# from open_instruct.ground_truth_utils import REWARD_FN_MAPPING
from open_instruct.ground_truth_utils_wip import LMJudgeVerifier, REWARD_FN_MAPPING, VerifierFunction
from open_instruct.utils import retry_on_exception
import openai # For type hint
from typing import Any # For client type hint
from open_instruct.judges._client import llm_client # Need llm_client to create the client

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    """The model checkpoint for weights initialization."""
    model_revision: Optional[str] = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    torch_dtype: Optional[str] = None
    """Override the default `torch.dtype` and load the model under this dtype."""
    attn_implementation: Optional[Literal["flash_attention_2"]] = None
    """Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case
    you must install this manually by running `pip install flash-attn --no-build-isolation`"""
    use_cache: Optional[bool] = None
    """Whether to use cache in the model."""
    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing in the model."""

    # PEFT-related args
    use_peft: bool = False
    """Whether to use PEFT or not for training."""
    lora_r: Optional[int] = 16
    """LoRA R value."""
    lora_alpha: Optional[int] = 32
    """LoRA alpha."""
    lora_dropout: Optional[float] = 0.05
    """LoRA dropout."""
    lora_target_modules: Optional[List[str]] = None
    """LoRA target modules."""
    lora_modules_to_save: Optional[List[str]] = None
    """Model layers to unfreeze & train"""
    lora_task_type: str = "CAUSAL_LM"
    """The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"""

    # quantization args
    load_in_8bit: bool = False
    """use 8 bit precision for the base model - works only with LoRA"""
    load_in_4bit: bool = False
    """use 4 bit precision for the base model - works only with LoRA"""
    bnb_4bit_quant_type: Optional[str] = "nf4"
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    assert (
        reward_logits.shape[-1] == 1
    ), "Reward model should output a single scalar per token. Check if you added `num_labels=1` when doing `AutoModelForSequenceClassification.from_pretrained(...)`."
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454

    # Return the reward logits for all tokens, the final reward scores for each sequence, and the sequence lengths
    return (
        # reward_logits shape: (batch_size, sequence_length)
        reward_logits,
        # final_scores shape: (batch_size,)
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(
            -1
        ),  # Shape: (batch_size,)
        sequence_lengths,
    )


def apply_verifiable_reward(
    responses: List[torch.Tensor],
    decoded_responses: List[str],
    ground_truths: List[str],
    datasets: List[Union[str, List[str]]],
    reward_mult: int = 10,
):
    rewards = []
    per_func_rewards = []
    total_cost = 0
    for tok_prediction, prediction, ground_truth, dataset in zip(
        responses, decoded_responses, ground_truths, datasets
    ):
        # allow multiple ground truths and datasets for a single response
        if isinstance(ground_truth, str):
            ground_truth_list = [ground_truth]
        else:
            ground_truth_list = ground_truth
        if isinstance(dataset, str):
            dataset_list = [dataset]
        else:
            dataset_list = dataset
        assert len(ground_truth_list) == len(dataset_list), "Ground truth and dataset list lengths do not match."
        # for now, we just assume rewards are additive, rather than more complex functions.
        reward = 0
        per_func_reward = {}
        # breakpoint()
        for gt, ds in zip(ground_truth_list, dataset_list):
            reward_func = REWARD_FN_MAPPING.get(ds.lower())
            if reward_func is None:
                logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
                continue
            reward_weight = reward_func.weight
            # compare with ground truth.
            # sometimes we need the tokenized pred.
            reward_result = reward_func(
                tokenized_prediction=tok_prediction,
                prediction=prediction,
                label=gt,
            )
            logger.info("Applying ground truth reward 🤗")
            reward += reward_mult * reward_result * reward_weight
            per_func_reward[ds] = per_func_reward.get(ds, 0) + (reward_mult * reward_result * reward_weight)
        rewards.append(reward)
        per_func_rewards.append(per_func_reward)
    # breakpoint()
    return rewards, per_func_rewards, total_cost

# TODO (Faeze): merge this with apply_verifiable_reward and use dataset name to select the function
def apply_llm_verifier_reward(
    responses: List[torch.Tensor],
    decoded_responses: List[str],
    ground_truths: List[str],
    datasets: List[Union[str, List[str]]],
    queries: List[str],
    local_judge: bool,
    reward_mult: int = 10,
    judge_model: str = "gpt-4o-mini",
):
    """Apply LLM-based verification to calculate rewards for responses.
    
    This function is synchronous but internally uses asyncio for concurrent execution.
    """
    return asyncio.run(_apply_llm_verifier_reward_async(
        responses,
        decoded_responses,
        ground_truths,
        datasets,
        queries,
        local_judge,
        reward_mult,
        judge_model,
    ))

async def _apply_llm_verifier_reward_async(
    responses: List[torch.Tensor],
    decoded_responses: List[str],
    ground_truths: List[str],
    datasets: List[Union[str, List[str]]],
    queries: List[str],
    local_judge: bool,
    reward_mult: int = 10,
    judge_model: str = "gpt-4o-mini",
):
    """Async implementation of apply_llm_verifier_reward that executes judgments concurrently."""
    rewards = []
    per_func_rewards = []
    total_cost = []
    judge_reasonings = []

    # --- Client Management --- 
    # Create the client once here if any LMJudgeVerifier (general) will be used.
    # We assume the judge_model applies to all LMJudge calls within this batch.
    # Check if any dataset indicates the use of the 'general' verifier which needs the client.
    needs_openai_client = False
    overall_judge_type = "quality" # Default if not specified in dataset string
    for ds_list in datasets:
        if isinstance(ds_list, str):
             ds_list = [ds_list]
        for ds in ds_list:
            verifier_name = ds.lower().split("-")[0]
            if verifier_name == "general":
                needs_openai_client = True
                if len(ds.lower().split("-")) > 1:
                    # Optionally capture the first judge_type seen, though it might vary per dataset
                    overall_judge_type = ds.lower().split("-")[1]
                break # Found one, no need to check further in this inner loop
        if needs_openai_client:
            break # Found one, no need to check further in the outer loop

    client = None
    try:
        if needs_openai_client:
            logger.info(f"Creating single OpenAI client for judge model: {judge_model}")
            # Assuming judge_model is NOT a path like 'huggingface/...' for LMJudge
            client = llm_client(model_type="huggingface" if '/' in judge_model else "openai") # Use the factory from _client
        else:
            logger.info("No 'general' verifier detected in datasets, skipping OpenAI client creation.")

        # Create tasks for each response, passing the single client instance
        tasks = []
        for i, (tok_prediction, prediction, ground_truth, dataset, query) in enumerate(zip(
            responses, decoded_responses, ground_truths, datasets, queries
        )):
            # Process the same initial logic as before
            if isinstance(ground_truth, str):
                ground_truth_list = [ground_truth]
            else:
                ground_truth_list = ground_truth

            if isinstance(dataset, str):
                dataset_list = [dataset]
            else:
                dataset_list = dataset

            if len(ground_truth_list) != len(dataset_list):
                 logger.error(f"Mismatch lengths for item {i}: {len(ground_truth_list)=} != {len(dataset_list)=}. Skipping.")
                 # Append default/error values for this item
                 rewards.append(0)
                 per_func_rewards.append({})
                 total_cost.append(0)
                 judge_reasonings.append("Error: Mismatched ground truth and dataset lists.")
                 continue # Skip creating a task for this item

            # Create a task for processing this response, passing the client
            task = asyncio.create_task(_process_single_response(
                tok_prediction,
                prediction,
                ground_truth_list,
                dataset_list,
                query,
                local_judge,
                reward_mult,
                overall_judge_type, # Pass default/detected judge_type
                judge_model,
                client # Pass the potentially created client
            ))
            tasks.append(task)

        # Wait for all tasks to complete
        # Use return_exceptions=True to handle failures gracefully
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result_or_exception in results_or_exceptions:
            if isinstance(result_or_exception, Exception):
                logger.error(f"Task failed in apply_llm_verifier_reward: {result_or_exception}", exc_info=result_or_exception)
                # Append default/error values for failed tasks
                rewards.append(0)
                per_func_rewards.append({})
                total_cost.append(0)
                judge_reasonings.append(f"Error: {result_or_exception}")
            else:
                # Unpack successful result
                reward, per_func_reward, cost, judge_reason = result_or_exception
                rewards.append(reward)
                per_func_rewards.append(per_func_reward)
                total_cost.append(cost)
                judge_reasonings.append(judge_reason)

    finally:
        # --- Client Cleanup --- Ensure client is closed if it was created
        if client is not None and hasattr(client, 'aclose'):
            logger.info("Closing the OpenAI client.")
            await client.aclose()

    return rewards, per_func_rewards, total_cost, judge_reasonings

async def _process_single_response(
    tok_prediction,
    prediction,
    ground_truth_list,
    dataset_list,
    query,
    local_judge,
    reward_mult,
    judge_type,
    judge_model,
    client: Optional[Union[openai.AsyncOpenAI, Any]] = None,
):
    """Process a single response with potentially multiple ground truths and datasets."""
    reward = 0
    per_func_reward = {}
    cost = 0
    reasoning = "No judge applied." # Default reasoning

    judgment_tasks = []
    for gt, ds in zip(ground_truth_list, dataset_list):
        # Split dataset name, e.g., "general-quality" -> ["general", "quality"]
        verifier_name_parts = ds.lower().split("-")
        verifier_name = verifier_name_parts[0]
        # Determine the specific judge type for LMJudgeVerifier if provided in ds name
        current_judge_type = verifier_name_parts[1] if len(verifier_name_parts) > 1 else judge_type # Fallback to overall judge_type

        # Get the factory function for the verifier
        factory = REWARD_FN_MAPPING.get(verifier_name)

        if factory is None:
            logger.warning("No judge verifier factory found for verifier name %s (from dataset %s). Skipping reward.", verifier_name, ds)
            continue

        reward_func: VerifierFunction = None
        is_lm_judge = False
        try:
            # Instantiate the verifier
            if verifier_name == "general":
                # Pass LMJudge specific args
                reward_func = factory(judge_type=current_judge_type, model_name=judge_model, local_judge=local_judge)
                # Check if the instantiated object is indeed an LMJudgeVerifier to confirm we need the client
                is_lm_judge = isinstance(reward_func, LMJudgeVerifier)
            else:
                # For other verifiers, assume they don't need LMJudge specific args
                reward_func = factory()
                # Ensure it's not accidentally an LMJudgeVerifier instance
                is_lm_judge = isinstance(reward_func, LMJudgeVerifier)

        except Exception as e:
             logger.error(f"Failed to instantiate verifier '{verifier_name}' for dataset '{ds}': {e}", exc_info=True)
             continue

        # Prepare arguments for async_call
        call_args = {
            "tokenized_prediction": tok_prediction,
            "prediction": prediction,
            "label": gt,
            "query": query,
        }
        # Add client only if it's an LMJudgeVerifier instance
        if is_lm_judge:
            call_args["client"] = client
            logger.debug(f"Passing client to LMJudgeVerifier for dataset {ds}")
        elif client is not None:
             logger.debug(f"Client provided but not passing to non-LMJudgeVerifier for dataset {ds}")


        # Create a task for this judgment
        # Use try-except around create_task to handle potential immediate errors during task creation/submission
        try:
            # Ensure reward_func was successfully instantiated before creating task
            if reward_func is None:
                logger.error(f"Reward function was not instantiated for verifier '{verifier_name}', dataset '{ds}'. Skipping task creation.")
                continue

            task = asyncio.create_task(reward_func.async_call(**call_args))
            # Store the type of the instantiated reward_func to check later
            judgment_tasks.append((task, ds, reward_func.weight, current_judge_type, type(reward_func)))
        except Exception as e:
            logger.error(f"Failed to create judgment task for dataset {ds} with verifier {verifier_name}: {e}", exc_info=True)


    # Wait for all judgments to complete concurrently
    # Use return_exceptions=True to prevent one failed task from stopping others
    judgment_results_with_status = await asyncio.gather(*[task for task, _, _, _, _ in judgment_tasks], return_exceptions=True)

    # Process the results
    for i, result_or_exception in enumerate(judgment_results_with_status):
        # task is not directly needed here, but other info is
        _, ds, weight, ds_judge_type, verifier_type = judgment_tasks[i] # Get corresponding task info

        if isinstance(result_or_exception, Exception):
            logger.error(f"Judgment task for dataset {ds} failed: {result_or_exception}", exc_info=result_or_exception)
            # Optionally assign a default/penalty reward/cost here
            continue # Skip processing this result

        # Unpack successful result (assuming it returns tuple like (score, cost, reasoning))
        try:
            # Check the type of verifier that produced the result
            if issubclass(verifier_type, LMJudgeVerifier):
                reward_result, api_cost, judge_reasoning = result_or_exception
                reasoning = judge_reasoning # Capture reasoning from the judge
            else:
                # Assume others return (score, cost, reasoning_str) or similar
                # Adapt based on the actual return signature of non-LMJudge verifiers
                if isinstance(result_or_exception, tuple) and len(result_or_exception) >= 2:
                    reward_result, api_cost = result_or_exception[:2]
                    # Optionally capture reasoning if returned as third element
                    judge_reasoning = result_or_exception[2] if len(result_or_exception) > 2 else f"Verifier {ds.lower().split('-')[0]} score: {reward_result}"
                elif isinstance(result_or_exception, (float, int)):
                    reward_result = result_or_exception
                    api_cost = 0 # Assume no cost if not returned
                    judge_reasoning = f"Verifier {ds.lower().split('-')[0]} score: {reward_result}" # Basic reasoning for non-judges
                else:
                    logger.warning(f"Unexpected result format from verifier for {ds}: {result_or_exception}. Assigning score=0, cost=0.")
                    reward_result = 0
                    api_cost = 0
                    judge_reasoning = f"Verifier {ds.lower().split('-')[0]} failed to produce score."
                reasoning = judge_reasoning # Update overall reasoning

        except ValueError as e:
             logger.error(f"Error unpacking result for dataset {ds}: {e}. Result was: {result_or_exception}", exc_info=True)
             continue # Skip processing this result
        except TypeError as e:
             logger.error(f"Type error unpacking result for dataset {ds}: {e}. Result was: {result_or_exception}", exc_info=True)
             continue # Skip processing this result


        logger.info("Applying reward for dataset %s (verifier: %s, judge_type: %s) 🤗", ds, verifier_name_parts[0], ds_judge_type)

        # Determine reward multiplier (e.g., don't multiply reference scores)
        actual_reward_mult = 1.0 if "ref" in ds_judge_type else reward_mult
        reward_value = actual_reward_mult * reward_result * weight

        reward += reward_value
        per_func_reward[ds] = per_func_reward.get(ds, 0) + reward_value
        cost += api_cost

    return reward, per_func_reward, cost, reasoning


def forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
) -> torch.nn.Module:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.
    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
    Returns:
        `torch.nn.Module`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def truncate_response(stop_token_id: int, pad_token_id: int, responses: torch.Tensor):
    """
    Truncates the responses at the first occurrence of the stop token, filling the rest with pad tokens.
    Args:
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs.
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses.
        responses (`torch.Tensor`):
            The tensor containing the responses to be truncated.
    Returns:
        `torch.Tensor`:
            The truncated responses tensor with pad tokens filled after the stop token.
    """
    trunc_idxs = first_true_indices(responses == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses


def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.
    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`dict`):
            The configuration dictionary for generation settings.
    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_logits=True,
    )
    logits = torch.stack(output.logits, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: dict,
):
    query_responses = []
    logitss = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)
    return torch.cat(query_responses, 0), torch.cat(logitss, 0)


def save_with_accelerate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    use_lora: bool = False,
    model_attribute_to_save: Optional[str] = None,
) -> None:
    """`model_attribute_to_save` is for used to save PPO's policy instead of the full model"""
    # set the generation config to an empty setting to be safe.
    # we usually do greedy decoding for generation, so this should be okay.
    # otherwise, we get an error thrown at save time.
    model.generation_config = transformers.GenerationConfig(
        temperature=None, top_p=None, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id
    )

    unwrapped_model: PreTrainedModel = accelerator.unwrap_model(model)
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
    accelerator: Accelerator,
    output_dir: str,
    hf_repo_id: Optional[str] = None,
    hf_repo_revision: Optional[str] = None,
    private: bool = True,
):
    if accelerator.is_main_process:
        hf_repo_url = f"https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}"
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
        print(f"🔥 pushed to {hf_repo_url}")


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
    panel = Panel(
        table,
        title="Metrics",
        expand=False,
        border_style="bold green",
    )

    # Print the panel
    rprint(panel)


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q
