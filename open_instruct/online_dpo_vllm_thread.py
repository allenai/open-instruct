import gc
import json
import os
import random
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import DatasetDict
from huggingface_hub import HfApi
from rich.pretty import pprint
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    get_scheduler,
)
from vllm import LLM, SamplingParams

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    INPUT_IDS_PROMPT_KEY,
    DatasetConfig,
    SFTDatasetProcessor,
    SimpleGenerateCollator,
    visualize_token,
)
from open_instruct.model_utils import (
    ModelConfig,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    log_softmax_and_gather,
    prepare_deepspeed,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
    save_with_accelerate,
    truncate_response,
    unwrap_model_for_generation,
)
from open_instruct.utils import (
    ArgumentParserPlus,
    combine_dataset,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    upload_metadata_to_hf,
)
from open_instruct.vllm_utils3 import create_vllm_engines, init_process_group

api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class Args:
    # required dataset args
    dataset_mixer: str = None
    """A dictionary of datasets (local or HF) to sample from."""
    dataset_train_splits: List[str] = None
    """The dataset splits to use for training"""
    dataset_eval_mixer: Optional[str] = None
    """A dictionary of datasets (local or HF) to sample from for evaluation"""
    dataset_eval_splits: Optional[List[str]] = None
    """The dataset splits to use for evaluation"""
    dataset_mixer_dict: Optional[dict] = None
    """The dataset mixer as a dictionary"""
    dataset_eval_mixer_dict: Optional[dict] = None
    """The dataset eval mixer as a dictionary"""

    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""

    # ray
    actor_num_gpus_per_node: List[int] = field(default_factory=lambda: [1])
    """number of gpus per node for actor"""
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference"""
    vllm_enforce_eager: bool = False
    """whether to enforce eager mode for vLLM -- slow inference but needed for multi-node"""
    vllm_sync_backend: str = "nccl"
    """DeepSpeed -> vLLM weight sync backend"""
    vllm_gpu_memory_utilization: float = 0.9
    """vLLM GPU memory utilization"""
    vllm_enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    deepspeed_stage: int = 0
    """the deepspeed stage"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast (not doable for 70B but can be faster for 8B)"""

    # optimizer args
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    num_train_epochs: int = 1
    """Number of epochs to train"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: Optional[int] = 1
    """The forward batch size per device (local_micro_batch_size)"""
    per_device_eval_batch_size: Optional[int] = 1
    """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: Optional[int] = 100000
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    num_training_steps: Optional[int] = None
    """The number of training_steps to train"""
    num_evals: int = 4
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """The frequency of evaluation steps"""
    local_dataloader_batch_size: Optional[int] = None
    """The batch size per GPU for the dataloader"""

    # online settings
    num_epochs: int = 4
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""
    reward_model_revision: Optional[str] = None
    """the revision of the reward model"""

    # generation config
    response_length: int = 53
    """the length of the response"""
    stop_token: Optional[Literal["eos", "period"]] = None
    """the stop token"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    min_response_length: int = 0
    """stop only after this many tokens"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: float = -1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    non_stop_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""

    # online DPO specific args
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    num_generation_per_prompt: int = 2
    """the number of generations per prompt (currently only support 2)"""
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    """the loss type for the DPO algorithm"""

    # wandb and HF tracking configs
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: Optional[str] = None
    """Where to save the model"""
    checkpoint_output_dir: Optional[str] = None
    """Where to save the model checkpoints in case of preemption"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    def __post_init__(self):
        self.dataset_mixer_dict, self.dataset_mixer = process_dataset_mixer(self.dataset_mixer)
        if self.dataset_eval_mixer is not None:
            self.dataset_eval_mixer_dict, self.dataset_eval_mixer = process_dataset_mixer(self.dataset_eval_mixer)


def process_dataset_mixer(value) -> Tuple[Optional[dict], Optional[str]]:
    # if passed through cli: convert the dataset mixers to dictionaries
    if isinstance(value, str):
        return json.loads(value), value
    # if passed through yaml: convert the dataset mixers to strings
    elif isinstance(value, dict):
        return value, json.dumps(value)
    else:
        raise ValueError("Input must be either a string or a dictionary")


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess:
    def __init__(
        self,
        rank: int,
        world_size: int,
        model_name_or_path: str,
        model_revision: Optional[str],
        max_model_len: int,
        vllm_gpu_memory_utilization: float,
        generation_config: SamplingParams,
        response_ids_Q: Queue,
        param_prompt_Q: Queue,
        num_training_steps: int,
        sample_evaluation_prompt_token_ids: Optional[List[int]],
        evaluation_Q: Queue,
        eval_freq: int,
        resume_training_step: int,
        num_engines: int = 1,
        tensor_parallel_size: int = 1,
        vllm_enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
        single_gpu_mode: bool = False,
        args: Optional[Args] = None,
        dataset_config: Optional[DatasetConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_name_or_path = model_name_or_path
        self.model_revision = model_revision
        self.max_model_len = max_model_len
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.generation_config = generation_config
        self.response_ids_Q = response_ids_Q
        self.param_prompt_Q = param_prompt_Q
        self.num_training_steps = num_training_steps
        self.sample_evaluation_prompt_token_ids = sample_evaluation_prompt_token_ids
        self.evaluation_Q = evaluation_Q
        self.eval_freq = eval_freq
        self.resume_training_step = resume_training_step
        self.num_engines = num_engines
        self.tensor_parallel_size = tensor_parallel_size
        self.vllm_enforce_eager = vllm_enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self.single_gpu_mode = single_gpu_mode
        self.args = args
        self.dataset_config = dataset_config
        self.model_config = model_config

        # Initialize distributed training
        init_process_group(backend="nccl")
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)

        # Create vLLM engines
        self.vllm_engines = create_vllm_engines(
            num_engines=num_engines,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=vllm_enforce_eager,
            pretrain=model_name_or_path,
            revision=model_revision,
            seed=42,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            single_gpu_mode=single_gpu_mode,
        )

        # Create model update group for broadcasting weights
        self.model_update_group = torch.distributed.new_group()

        # Initialize models and tokenizer
        self.setup_models_and_tokenizer()

    def setup_models_and_tokenizer(self):
        # Create tokenizer
        config = AutoConfig.from_pretrained(self.model_name_or_path, revision=self.model_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, revision=self.model_revision, padding_side="right"
        )
        if config.architectures == "LlamaForCausalLM" and config.bos_token_id == 128000:
            self.tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
        else:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
        if self.dataset_config.chat_template is not None:
            self.tokenizer.chat_template = CHAT_TEMPLATES[self.dataset_config.chat_template]

        # Create models
        self.policy = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.reward_model_path,
            revision=self.args.reward_model_revision,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )

        # Move models to device
        self.policy = self.policy.to(self.device)
        self.ref_model = self.ref_model.to(self.device)
        self.reward_model = self.reward_model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.args.learning_rate, eps=self.args.eps)
        self.scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warm_up_steps,
            num_training_steps=self.args.num_training_steps * self.args.num_train_epochs,
        )

    def generate_with_engines(self, prompts: List[List[int]], sampling_params: SamplingParams):
        # Split queries between engines
        queries_per_engine = (len(prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
        split_queries = [prompts[i : i + queries_per_engine] for i in range(0, len(prompts), queries_per_engine)]
        # Generate responses in parallel across engines
        futures = [
            vllm_engine.generate.remote(sampling_params=sampling_params, prompt_token_ids=queries, use_tqdm=False)
            for vllm_engine, queries in zip(self.vllm_engines, split_queries)
        ]
        # Gather all responses
        all_outputs = ray.get(futures)
        response_ids = []
        for outputs in all_outputs:
            response_ids.extend([list(out.token_ids) for output in outputs for out in output.outputs])
        return response_ids

    def broadcast_to_vllm(self, model):
        # avoid OOM
        torch.cuda.empty_cache()
        count, num_params = 0, len(list(model.named_parameters()))
        refss = []
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param
            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                refs = [
                    engine.update_weight.remote(
                        name, dtype=param.dtype, shape=param.shape, empty_cache=count == num_params
                    )
                    for engine in self.vllm_engines
                ]
                refss.extend(refs)
            if torch.distributed.get_rank() == 0:
                torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        if torch.distributed.get_rank() == 0:
            ray.get(refss)

    def train(self):
        print("🔥🔥🔥 vLLM engines loaded")
        
        # Initialize metrics
        stats_shape = (self.args.num_epochs, self.args.num_mini_batches, self.args.gradient_accumulation_steps)
        loss_stats = torch.zeros(stats_shape, device=self.device)
        chosen_rewards_stats = torch.zeros(stats_shape, device=self.device)
        rejected_rewards_stats = torch.zeros(stats_shape, device=self.device)
        chosen_logprobs_stats = torch.zeros(stats_shape, device=self.device)
        rejected_logprobs_stats = torch.zeros(stats_shape, device=self.device)
        local_metrics = torch.zeros((20,), device=self.device)
        episode = self.args.batch_size * (self.resume_training_step - 1)
        self.policy.train()

        # Create dataset and dataloader
        dataset_dict = DatasetDict()
        dataset_processor = SFTDatasetProcessor(tokenizer=self.tokenizer, config=self.dataset_config)
        train_dataset = combine_dataset(
            self.args.dataset_mixer_dict,
            splits=self.args.dataset_train_splits,
            columns_to_keep=[self.dataset_config.sft_messages_key],
        )
        if self.dataset_config.sanity_check:
            train_dataset = train_dataset.select(
                range(0, min(len(train_dataset), self.dataset_config.sanity_check_max_samples))
            )
        train_dataset = dataset_processor.tokenize(train_dataset)
        train_dataset = dataset_processor.filter(train_dataset)
        dataset_dict["train"] = train_dataset

        data_collator = SimpleGenerateCollator(pad_token_id=self.tokenizer.pad_token_id)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=True,
        )

        # Training loop
        start_time = time.time()
        iter_dataloader = iter(dataloader)
        data = next(iter_dataloader)
        queries_next = data[INPUT_IDS_PROMPT_KEY].to(self.device)
        queries_next = queries_next.repeat(self.args.num_generation_per_prompt, 1)
        self.send_queries(None, queries_next)

        for training_step in range(self.resume_training_step, self.num_training_steps + 1):
            episode += self.args.batch_size
            self.scheduler.step()
            queries = queries_next

            # Get responses from vLLM
            items = self.param_prompt_Q.get()
            if items is None:
                break
            unwrapped_model, g_queries_list = items
            if unwrapped_model is not None:
                start_time = time.time()
                self.broadcast_to_vllm(unwrapped_model)
                print(f"🔥🔥🔥 Loading weights using shared memory; Time to load weights: {time.time() - start_time:.2f} seconds")
            
            generation_start_time = time.time()
            response_ids = self.generate_with_engines(g_queries_list, self.generation_config)
            print(f"🔥🔥🔥 Generation time: {time.time() - generation_start_time:.2f} seconds")
            self.response_ids_Q.put(response_ids)

            # Process responses and compute rewards
            with torch.no_grad():
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []

                g_response_token_ids = response_ids
                DUMMY_PAD_TOKEN = 0
                g_padded_response_ids = [
                    response + [DUMMY_PAD_TOKEN] * (self.args.response_length - len(response))
                    for response in g_response_token_ids
                ]
                g_padded_response_ids = torch.tensor(g_padded_response_ids, device=self.device)
                local_vllm_responses = g_padded_response_ids[
                    self.rank * queries.shape[0] : (self.rank + 1) * queries.shape[0]
                ]
                query_responses = torch.cat((queries, local_vllm_responses), 1)

                for i in range(0, queries.shape[0], self.args.local_rollout_forward_batch_size):
                    query = queries[i : i + self.args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + self.args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]

                    # Get reference model logprobs
                    ref_output = forward(self.ref_model, query_response, self.tokenizer.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits = ref_logits / (self.args.temperature + 1e-7)
                    ref_logprob = log_softmax_and_gather(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # Process responses
                    postprocessed_response = response
                    if self.args.stop_token_id is not None:
                        postprocessed_response = truncate_response(
                            self.args.stop_token_id, self.tokenizer.pad_token_id, response
                        )

                    # Get rewards
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == self.tokenizer.pad_token_id) - 1
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, self.tokenizer.pad_token_id, context_length
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)

                # Process scores and compute chosen/rejected indices
                contain_stop_token = torch.any(postprocessed_responses == self.args.stop_token_id, dim=-1)
                contain_stop_token = contain_stop_token & (sequence_lengths >= self.args.min_response_length)
                if self.args.non_stop_penalty:
                    scores = torch.where(
                        contain_stop_token, scores, torch.full_like(scores, self.args.penalty_reward_value)
                    )

                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                num_examples = scores.size(0) // 2
                first_half = scores[:num_examples]
                second_half = scores[num_examples:]

                num_examples_range = torch.arange(num_examples).to(scores.device)
                chosen_indices = torch.where(
                    first_half >= second_half, num_examples_range.clone(), num_examples_range.clone() + num_examples
                )
                rejected_indices = torch.where(
                    first_half < second_half, num_examples_range.clone(), num_examples_range.clone() + num_examples
                )
                scores_margin = scores[chosen_indices] - scores[rejected_indices]

            # Training loop
            logprobs = []
            concat_indices = []
            for epoch_idx in range(self.args.num_epochs):
                b_inds = np.random.permutation(self.args.local_batch_size // self.args.num_generation_per_prompt)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0,
                    self.args.local_batch_size // self.args.num_generation_per_prompt,
                    self.args.local_mini_batch_size // self.args.num_generation_per_prompt,
                ):
                    mini_batch_end = mini_batch_start + self.args.local_mini_batch_size // self.args.num_generation_per_prompt
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                        0,
                        self.args.local_mini_batch_size // self.args.num_generation_per_prompt,
                        self.args.per_device_train_batch_size,
                    ):
                        micro_batch_end = micro_batch_start + self.args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        chosen_mb_inds = chosen_indices[micro_batch_inds]
                        chosen_responses = responses[chosen_mb_inds]
                        rejected_mb_inds = rejected_indices[micro_batch_inds]
                        rejected_responses = responses[rejected_mb_inds]

                        concat_mb_inds = torch.cat((chosen_mb_inds, rejected_mb_inds), dim=0)
                        concat_query_responses = query_responses[concat_mb_inds]
                        concat_output = forward(self.policy, concat_query_responses, self.tokenizer.pad_token_id)
                        num_examples = chosen_mb_inds.shape[0]
                        chosen_logits = concat_output.logits[:num_examples]
                        rejected_logits = concat_output.logits[num_examples:]

                        # Compute chosen logprobs
                        chosen_logits = chosen_logits[:, context_length - 1 : -1]
                        chosen_logits = chosen_logits / (self.args.temperature + 1e-7)
                        chosen_logprobs = log_softmax_and_gather(chosen_logits, chosen_responses)
                        chosen_logprobs = torch.masked_fill(
                            chosen_logprobs, padding_mask[chosen_mb_inds], INVALID_LOGPROB
                        )
                        chosen_ref_logprobs = ref_logprobs[chosen_mb_inds]
                        chosen_logprobs_sum = (chosen_logprobs * ~padding_mask[chosen_mb_inds]).sum(1)
                        chosen_ref_logprobs_sum = (chosen_ref_logprobs * ~padding_mask[chosen_mb_inds]).sum(1)

                        # Compute rejected logprobs
                        rejected_logits = rejected_logits[:, context_length - 1 : -1]
                        rejected_logits = rejected_logits / (self.args.temperature + 1e-7)
                        rejected_logprobs = log_softmax_and_gather(rejected_logits, rejected_responses)
                        rejected_logprobs = torch.masked_fill(
                            rejected_logprobs, padding_mask[rejected_mb_inds], INVALID_LOGPROB
                        )
                        rejected_ref_logprobs = ref_logprobs[rejected_mb_inds]
                        rejected_logprobs_sum = (rejected_logprobs * ~padding_mask[rejected_mb_inds]).sum(1)
                        rejected_ref_logprobs_sum = (rejected_ref_logprobs * ~padding_mask[rejected_mb_inds]).sum(1)

                        # Compute DPO loss
                        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
                        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum
                        logits = pi_logratios - ref_logratios

                        if self.args.loss_type == "sigmoid":
                            losses = -F.logsigmoid(self.args.beta * logits)
                        elif self.args.loss_type == "ipo":
                            losses = (logits - 1 / (2 * self.args.beta)) ** 2
                        else:
                            raise NotImplementedError(f"invalid loss type {self.args.loss_type}")

                        loss = losses.mean()
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Update metrics
                        with torch.no_grad():
                            if epoch_idx == 0:
                                concat_indices.append(concat_mb_inds)
                                response = concat_query_responses[:, context_length:]
                                logits = concat_output.logits[:, context_length - 1 : -1]
                                logits /= self.args.temperature + 1e-7
                                logprob = log_softmax_and_gather(logits, response)
                                logprob = torch.masked_fill(logprob, padding_mask[concat_mb_inds], INVALID_LOGPROB)
                                logprobs.append(logprob)
                            chosen_rewards = self.args.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
                            rejected_rewards = self.args.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
                            loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss
                            chosen_rewards_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                chosen_rewards.mean()
                            )
                            rejected_rewards_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                rejected_rewards.mean()
                            )
                            chosen_logprobs_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                chosen_logprobs_sum.mean()
                            )
                            rejected_logprobs_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                rejected_logprobs_sum.mean()
                            )
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    del (
                        loss, logits, concat_output, concat_query_responses,
                        chosen_logits, rejected_logits, chosen_logprobs, rejected_logprobs,
                        chosen_responses, rejected_responses,
                    )
                    torch.cuda.empty_cache()

            # Compute final metrics
            with torch.no_grad():
                logprobs = torch.cat(logprobs, 0)
                concat_indices = torch.cat(concat_indices, 0)
                restore_logprobs = torch.zeros_like(logprobs)
                restore_logprobs[concat_indices] = logprobs
                kl = restore_logprobs - ref_logprobs
                non_score_reward = -self.args.beta * kl
                non_score_reward_sum = non_score_reward.sum(1)
                rlhf_reward = scores + non_score_reward_sum
                local_metrics[0] = sequence_lengths.float().mean()
                local_metrics[1] = (responses == self.args.stop_token_id).sum().float().mean()
                local_metrics[2] = kl.sum(1).mean()
                local_metrics[3] = (-logprobs).sum(1).mean()
                local_metrics[4] = non_score_reward_sum.mean()
                local_metrics[5] = rlhf_reward.mean()
                local_metrics[6] = scores.mean()
                local_metrics[7] = scores_margin.mean()
                local_metrics[8] = loss_stats.mean()
                local_metrics[9] = chosen_rewards_stats.mean()
                local_metrics[10] = rejected_rewards_stats.mean()
                local_metrics[11] = (chosen_rewards_stats > rejected_rewards_stats).float().mean()
                local_metrics[12] = (chosen_rewards_stats - rejected_rewards_stats).mean()
                local_metrics[13] = chosen_logprobs_stats.mean()
                local_metrics[14] = rejected_logprobs_stats.mean()
                local_metrics[15] = ((kl) ** 2 / 2).sum(1).mean()
                local_metrics[16] = ((-kl).exp() - 1 + kl).sum(1).mean()

                # Gather metrics across processes
                global_metrics = torch.zeros_like(local_metrics)
                torch.distributed.all_reduce(local_metrics, op=torch.distributed.ReduceOp.SUM)
                global_metrics = local_metrics / self.world_size

                if self.rank == 0:
                    metrics = {
                        "episode": episode,
                        "training_step": training_step,
                        "lr": self.scheduler.get_last_lr()[0],
                        "epoch": episode / len(train_dataset),
                        "time/from_scratch": time.time() - start_time,
                        "time/training": time.time() - generation_start_time,
                        "val/sequence_lengths": global_metrics[0].item(),
                        "val/num_stop_token_ids": global_metrics[1].item(),
                        "objective/kl": global_metrics[2].item(),
                        "objective/kl2": global_metrics[15].item(),
                        "objective/kl3": global_metrics[16].item(),
                        "objective/entropy": global_metrics[3].item(),
                        "objective/non_score_reward": global_metrics[4].item(),
                        "objective/rlhf_reward": global_metrics[5].item(),
                        "objective/scores": global_metrics[6].item(),
                        "objective/scores_margin": global_metrics[7].item(),
                        "objective/loss": global_metrics[8].item(),
                        "rewards/chosen": global_metrics[9].item(),
                        "rewards/rejected": global_metrics[10].item(),
                        "rewards/accuracies": global_metrics[11].item(),
                        "rewards/margins": global_metrics[12].item(),
                        "logps/chosen": global_metrics[13].item(),
                        "logps/rejected": global_metrics[14].item(),
                    }
                    print_rich_single_line_metrics(metrics)
                    if self.args.with_tracking:
                        import wandb
                        wandb.log(metrics, step=episode)

            # Prepare next batch
            data = next(iter_dataloader)
            queries_next = data[INPUT_IDS_PROMPT_KEY].to(self.device)
            queries_next = queries_next.repeat(self.args.num_generation_per_prompt, 1)
            self.send_queries(self.policy, queries_next)

            # Clean up
            del (queries, responses, postprocessed_responses, logprobs, ref_logprobs, sequence_lengths, scores)
            del (metrics, kl, non_score_reward, rlhf_reward)
            gc.collect()
            torch.cuda.empty_cache()

    def send_queries(self, model, queries):
        g_queries_list = gather_object(queries.tolist())
        if torch.distributed.get_rank() == 0:
            g_queries_list = [
                [inneritem for inneritem in item if inneritem != self.tokenizer.pad_token_id] for item in g_queries_list
            ]  # remove padding
            self.param_prompt_Q.put((model, g_queries_list))


def send_queries(accelerator, unwrapped_model, tokenizer, param_prompt_Q, queries):
    g_queries_list = gather_object(queries.tolist())
    if accelerator.is_main_process:
        g_queries_list = [
            [inneritem for inneritem in item if inneritem != tokenizer.pad_token_id] for item in g_queries_list
        ]  # remove padding
        param_prompt_Q.put((unwrapped_model, g_queries_list))


def main(args: Args, dataset_config: DatasetConfig, model_config: ModelConfig):
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Calculate runtime args
    args.world_size = sum(args.actor_num_gpus_per_node)
    
    # Calculate batch sizes
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    
    # Set run name with timestamp
    time_tensor = torch.tensor(int(time.time()), device="cuda:0" if torch.cuda.is_available() else "cpu")
    time_int = time_tensor.item()
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    
    # Calculate mini batch sizes
    args.mini_batch_size = exact_div(
        args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
    )
    args.local_mini_batch_size = exact_div(
        args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
    )
    
    # Calculate training steps
    args.num_training_steps = args.total_episodes // args.batch_size
    args.eval_freq = max(1, args.num_training_steps // args.num_evals)
    
    # Calculate dataloader batch size for DPO
    args.local_dataloader_batch_size = exact_div(
        args.local_batch_size,
        args.num_generation_per_prompt,
        "`local_batch_size` must be a multiple of `num_generation_per_prompt`",
    )
    
    # Setup HuggingFace repository info
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    # Setup wandb
    if args.with_tracking:
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()

    # Create policy trainer processes
    policy_trainers = []
    for rank in range(args.world_size):
        trainer = PolicyTrainerRayProcess.remote(
            rank=rank,
            world_size=args.world_size,
            model_name_or_path=model_config.model_name_or_path,
            model_revision=model_config.model_revision,
            max_model_len=dataset_config.max_prompt_token_length + args.response_length,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            generation_config=SamplingParams(
                temperature=args.temperature,
                top_p=1.0,
                max_tokens=args.response_length,
                include_stop_str_in_output=True,
            ),
            response_ids_Q=Queue(maxsize=1),
            param_prompt_Q=Queue(maxsize=1),
            num_training_steps=args.num_training_steps,
            sample_evaluation_prompt_token_ids=None,  # TODO: Add evaluation support
            evaluation_Q=Queue(maxsize=1),
            eval_freq=args.eval_freq,
            resume_training_step=1,
            num_engines=args.vllm_num_engines,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_enforce_eager=args.vllm_enforce_eager,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            single_gpu_mode=args.single_gpu_mode,
            args=args,
            dataset_config=dataset_config,
            model_config=model_config,
        )
        policy_trainers.append(trainer)

    # Start training
    ray.get([trainer.train.remote() for trainer in policy_trainers])


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, ModelConfig))
    main(*parser.parse())
