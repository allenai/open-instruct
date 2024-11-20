import gc
import json
import os
import random
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
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
from torch.utils.tensorboard import SummaryWriter
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
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    DatasetConfig,
    SFTGroundTruthDatasetProcessor,
    SimpleGenerateCollatorWithGroundTruth,
    visualize_token,
)
from open_instruct.model_utils import (
    ModelConfig,
    apply_verifiable_reward,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
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
from open_instruct.vllm_utils import vllm_single_gpu_patch

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
    save_freq: int = -1
    """How many train steps to save the model"""

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
    init_value_from_scratch: bool = False
    """whether to initialize the value model from scratch"""

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
    number_samples_per_prompt: int = 1
    """the number of samples to generate per prompt, useful for easy-star"""

    # online PPO specific args
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    cliprange: float = 0.2
    """the clip range"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    kl_estimator: Literal["kl1", "kl2", "kl3"] = "kl1"
    """the KL estimator to use"""
    apply_verifiable_reward: bool = False
    """whether to apply verifiable reward"""
    reward_model_multiplier: float = 1.0
    """the reward model multiplier, for down/upscaling the reward model output"""
    answer_extraction_model: str = None

    # vLLM settings. NOTE: currently we need to place the vLLM model on a separate GPU
    # for generation to work properly because vLLM would pre-alocate the memory.
    # To do so, we would need to do a moneky patch `vllm_single_gpu_patch` to make sure
    # the vLLM model is placed on the correct GPU.
    vllm_device: str = "cuda:1"
    """the device placement of the vllm model; typically we place the vllm model on a decicated GPU"""
    vllm_gpu_memory_utilization: float = 0.8
    """the GPU memory utilization of the vllm model; passed to `gpu_memory_utilization` to the `vLLM` instance"""
    # async setting
    async_mode: bool = True
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""

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


def calculate_runtime_args_and_accelerator(args: Args, model_config: ModelConfig) -> Accelerator:
    """calculate (in-place) runtime args such as the effective batch size, word size, etc."""
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    # set a unique run name with the current timestamp
    time_int = broadcast(time_tensor, 0).item()
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    args.mini_batch_size = exact_div(
        args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
    )
    args.local_mini_batch_size = exact_div(
        args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
    )
    args.num_training_steps = args.total_episodes // (args.batch_size * args.number_samples_per_prompt)
    args.eval_freq = max(1, args.num_training_steps // args.num_evals)
    # PPO logic: do checks and set up dataloader batch size
    if args.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    args.local_dataloader_batch_size = args.local_batch_size
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

    if args.with_tracking and accelerator.is_main_process:
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
    return accelerator


def vllm_generate(
    model_name_or_path: str,
    model_revision: Optional[str],
    max_model_len: int,
    vllm_device: str,
    vllm_gpu_memory_utilization: float,
    generation_config: SamplingParams,
    response_ids_Q: Queue,
    param_prompt_Q: Queue,
    num_training_steps: int,
    sample_evaluation_prompt_token_ids: Optional[List[int]],
    evaluation_Q: Queue,
    eval_freq: int,
    resume_training_step: int,
):
    vllm_single_gpu_patch()
    llm = LLM(
        model=model_name_or_path,
        revision=model_revision,
        tokenizer_revision=model_revision,
        tensor_parallel_size=1,
        device=vllm_device,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    print("🔥🔥🔥 vllm loaded")
    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
    for training_step in range(resume_training_step, num_training_steps + 1):
        items = param_prompt_Q.get()
        if items is None:
            break
        unwrapped_model, g_queries_list = items
        if unwrapped_model is not None:
            start_time = time.time()
            llmp.load_weights(unwrapped_model.named_parameters())
            print(
                f"🔥🔥🔥 Loading weights using shared memory; Time to load weights: {time.time() - start_time:.2f} seconds"
            )
        generation_start_time = time.time()
        outputs = llm.generate(prompt_token_ids=g_queries_list, sampling_params=generation_config)
        response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
        print(f"🔥🔥🔥 Generation time: {time.time() - generation_start_time:.2f} seconds")
        response_ids_Q.put(response_ids)

        if sample_evaluation_prompt_token_ids is not None and (training_step - 1) % eval_freq == 0:
            outputs = llm.generate(
                prompt_token_ids=sample_evaluation_prompt_token_ids, sampling_params=generation_config
            )
            # for evaluation, even if we have multiple outputs, we only look at one of them for simplicity
            response_ids = [list(output.outputs[0].token_ids) for output in outputs]
            evaluation_Q.put(response_ids)


def send_queries(accelerator, unwrapped_model, tokenizer, param_prompt_Q, queries):
    g_queries_list = gather_object(queries.tolist())
    if accelerator.is_main_process:
        g_queries_list = [
            [inneritem for inneritem in item if inneritem != tokenizer.pad_token_id] for item in g_queries_list
        ]  # remove padding
        param_prompt_Q.put((unwrapped_model, g_queries_list))


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(torch.nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(
            **kwargs,
        )
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits

    def gradient_checkpointing_enable(self):
        self.policy.gradient_checkpointing_enable()
        self.value_model.gradient_checkpointing_enable()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def main(args: Args, dataset_config: DatasetConfig, model_config: ModelConfig):
    accelerator = calculate_runtime_args_and_accelerator(args, model_config)
    local_seed = args.seed + accelerator.process_index

    # set up experiment tracking and seeds
    all_configs = {}
    if is_beaker_job():
        args.checkpoint_output_dir = os.environ.get("CHECKPOINT_OUTPUT_DIR", None)
        beaker_config = maybe_get_beaker_config()
        # try saving to the beaker `/output`, which will be uploaded to the beaker dataset
        if len(beaker_config.beaker_dataset_id_urls) > 0:
            args.output_dir = "/output"
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(dataset_config), **asdict(model_config))
    if accelerator.is_main_process:
        if args.with_tracking:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=all_configs,
                name=args.run_name,
                save_code=True,
                tags=[args.exp_name] + get_wandb_tags(),
            )
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    device = torch.device(f"cuda:{accelerator.local_process_index}")
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    # create a tokenizer (pad from right)
    config = AutoConfig.from_pretrained(model_config.model_name_or_path, revision=model_config.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.model_revision, padding_side="right"
    )
    if config.architectures == "LlamaForCausalLM" and config.bos_token_id == 128000:
        tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
    tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]

    # create the dataset
    dataset_dict = DatasetDict()
    dataset_processor = SFTGroundTruthDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    if len(args.dataset_train_splits) != len(args.dataset_mixer_dict) and len(args.dataset_train_splits) == 1:
        args.dataset_train_splits = [args.dataset_train_splits[0]] * len(args.dataset_mixer_dict)
        print(
            f"Dataset splits not provided for all datasets. Using the same {args.dataset_train_splits[0]} split for all datasets."
        )
    if len(args.dataset_eval_splits) != len(args.dataset_eval_mixer_dict) and len(args.dataset_eval_splits) == 1:
        args.dataset_eval_splits = [args.dataset_eval_splits[0]] * len(args.dataset_eval_mixer_dict)
        print(
            f"Dataset splits not provided for all datasets. Using the same {args.dataset_eval_splits[0]} split for all datasets."
        )
    train_dataset = combine_dataset(
        args.dataset_mixer_dict,
        splits=args.dataset_train_splits,
        columns_to_keep=[
            dataset_config.sft_messages_key,
            dataset_config.ground_truths_key,
            dataset_config.dataset_source_key,
        ],
    )
    if dataset_config.sanity_check:
        train_dataset = train_dataset.select(
            range(0, min(len(train_dataset), dataset_config.sanity_check_max_samples))
        )
    with accelerator.main_process_first():
        train_dataset = dataset_processor.tokenize(train_dataset)
        train_dataset = dataset_processor.filter(train_dataset, need_contain_labels=False)
    dataset_dict["train"] = train_dataset
    eval_dataset = None
    if args.dataset_eval_mixer is not None:
        eval_dataset = combine_dataset(
            args.dataset_eval_mixer_dict,
            splits=args.dataset_eval_splits,
            columns_to_keep=[
                dataset_config.sft_messages_key,
                dataset_config.ground_truths_key,
                dataset_config.dataset_source_key,
            ],
        )
        eval_dataset = eval_dataset.select(range(0, min(len(eval_dataset), dataset_config.sanity_check_max_samples)))
        with accelerator.main_process_first():
            eval_dataset = dataset_processor.tokenize(eval_dataset)
            eval_dataset = dataset_processor.filter(eval_dataset, need_contain_labels=False)
        dataset_dict["eval"] = eval_dataset

    # some more runtime logging
    if accelerator.is_main_process:
        pprint([args, dataset_config, model_config])
        visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)
        if args.with_tracking:
            # upload the visualized token length
            dataset_processor.get_token_length_visualization(
                dataset_dict, save_path=f"runs/{args.run_name}/token_length.png"
            )
            wandb.log({"token_length": wandb.Image(f"runs/{args.run_name}/token_length.png")})

    # create the model and optimizer
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    ref_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    value_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        revision=args.reward_model_revision,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    if args.init_value_from_scratch:
        value_model.init_weights()  # re-initialize the value model from scratch
    reward_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        revision=args.reward_model_revision,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    if policy.config.vocab_size != reward_model.config.vocab_size:
        raise ValueError(
            "Policy and reward model must have the same vocab size. "
            f"Policy: {policy.config.vocab_size}, Reward: {reward_model.config.vocab_size}. "
            "If they don't have the same vocab size, the policy could generate tokens which "
            "is going to cause index out of bound error in the reward model."
        )
    model = PolicyAndValueWrapper(policy, value_model)
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    for module in [model, ref_model, reward_model]:
        disable_dropout_in_model(module)
    if args.stop_token:
        if args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        if args.stop_token == "period":
            args.stop_token_id = tokenizer.encode(".")[0]
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_training_steps * args.num_train_epochs,
    )
    data_collator = SimpleGenerateCollatorWithGroundTruth(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.local_dataloader_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True,  # needed; otherwise the last batch will be of ragged shape
    )
    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    torch.manual_seed(local_seed)

    # resume from preemption
    resume_training_step = 1
    if os.path.exists(args.checkpoint_output_dir):
        for item in os.listdir(args.checkpoint_output_dir):
            print(item)
            if "step_" in item:
                old_checkpoint_path = os.path.join(args.checkpoint_output_dir, item)
                # check if the directory is empty
                if len(os.listdir(old_checkpoint_path)) == 0:
                    continue
                accelerator.load_state(old_checkpoint_path)
                resume_training_step = int(item.split("_")[-1])
                print("Resuming training from step", resume_training_step)
                if accelerator.is_main_process:
                    shutil.rmtree(old_checkpoint_path)
                break
    resume_training_step > 1

    # deepspeed setup
    is_deepspeed_enabled = getattr(accelerator.state, "deepspeed_plugin", None) is not None
    mixed_precision = accelerator.state.mixed_precision
    if is_deepspeed_enabled:
        reward_model = prepare_deepspeed(reward_model, args.per_device_train_batch_size, mixed_precision)
        ref_model = prepare_deepspeed(ref_model, args.per_device_train_batch_size, mixed_precision)
    else:
        reward_model = reward_model.to(device)
        ref_model = ref_model.to(device)

    # online generation config
    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        n=args.number_samples_per_prompt,
    )
    param_prompt_Q = None
    response_ids_Q = None
    evaluation_Q = None
    if accelerator.is_main_process:
        response_ids_Q = Queue(maxsize=1)
        param_prompt_Q = Queue(maxsize=1)
        evaluation_Q = Queue(maxsize=1)
        LOCAL_NUM_EVAL_SAMPLES = 4
        num_eval_samples = LOCAL_NUM_EVAL_SAMPLES * accelerator.num_processes
        sample_evaluation_prompt_token_ids = None
        if eval_dataset is not None:
            sample_evaluation_prompt_token_ids = eval_dataset[:num_eval_samples][INPUT_IDS_PROMPT_KEY]
        thread = threading.Thread(
            target=vllm_generate,
            args=(
                model_config.model_name_or_path,
                model_config.model_revision,
                dataset_config.max_prompt_token_length + args.response_length,
                args.vllm_device,
                args.vllm_gpu_memory_utilization,
                generation_config,
                response_ids_Q,
                param_prompt_Q,
                args.num_training_steps,
                sample_evaluation_prompt_token_ids,
                evaluation_Q,
                args.eval_freq,
                resume_training_step,
            ),
        )
        thread.start()
    torch.cuda.set_device(device)

    g_vllm_responses = torch.zeros(
        (args.batch_size * args.number_samples_per_prompt, args.response_length), device=device, dtype=torch.long
    )

    # set up the metrics and initial states
    stats_shape = (
        args.num_epochs,
        args.num_mini_batches * args.number_samples_per_prompt,
        args.gradient_accumulation_steps,
    )
    approxkl_stats = torch.zeros(stats_shape, device=device)
    pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
    pg_loss_stats = torch.zeros(stats_shape, device=device)
    vf_loss_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)
    local_metrics = torch.zeros((20,), device=device)
    episode = args.batch_size * (resume_training_step - 1)
    model.train()

    # setup extraction model. For now keep on CPU?
    if args.answer_extraction_model:
        answer_extraction_model = AutoModelForCausalLM.from_pretrained(args.answer_extraction_model)
        answer_extraction_tokenizer = AutoTokenizer.from_pretrained(args.answer_extraction_model)
    else:
        answer_extraction_model = None
        answer_extraction_tokenizer = None

    # training loop
    start_time = time.time()
    data = next(iter_dataloader)
    queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
    ground_truths_next = data[GROUND_TRUTHS_KEY]
    datasets_next = data[DATASET_SOURCE_KEY]
    send_queries(accelerator, None, tokenizer, param_prompt_Q, queries_next)

    for _ in range(1, resume_training_step):  # we didn't store scheduler state
        scheduler.step()

    for training_step in range(resume_training_step, args.num_training_steps + 1):
        episode += args.batch_size * args.number_samples_per_prompt  # each sample is an episode
        scheduler.step()
        queries = queries_next
        ground_truths = ground_truths_next
        datasets = datasets_next

        if accelerator.is_main_process:
            try:
                evaluation_responses = evaluation_Q.get(timeout=0.01)
                print("🔥🔥🔥 Evaluation responses received")
                table = {}
                table["prompt"] = tokenizer.batch_decode(sample_evaluation_prompt_token_ids)
                table["response"] = tokenizer.batch_decode(evaluation_responses)
                table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
                df = pd.DataFrame(table)
                print_rich_table(df)
                if args.with_tracking:
                    wandb.log({"sample_completions": wandb.Table(dataframe=df)})
                else:
                    print_rich_table(df)
                del table
            except Empty:
                print("🙈 Evaluation responses not received")

        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            # (optionally) evaluate the model
            generation_model = unwrapped_model.policy
            if args.async_mode:
                if training_step != 1:
                    data = next(iter_dataloader)
                    queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
                    ground_truths_next = data[GROUND_TRUTHS_KEY]
                    datasets_next = data[DATASET_SOURCE_KEY]
                send_queries(accelerator, generation_model, tokenizer, param_prompt_Q, queries_next)
            else:
                if training_step != 1:
                    # NOTE: important: the indent here is different for sync mode
                    # we also set to use `queries = queries_next` immediately
                    data = next(iter_dataloader)
                    queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
                    ground_truths_next = data[GROUND_TRUTHS_KEY]
                    datasets_next = data[DATASET_SOURCE_KEY]
                    send_queries(accelerator, generation_model, tokenizer, param_prompt_Q, queries_next)
                    queries = queries_next
                    ground_truths = ground_truths_next
                    datasets = datasets_next

            # if we generate multiple samples per prompt, we need to repeat the queries and ground truths
            # to match the vllm outputs.
            if args.number_samples_per_prompt > 1:
                queries = queries.repeat_interleave(args.number_samples_per_prompt, dim=0)
                ground_truths = [gt for gt in ground_truths for _ in range(args.number_samples_per_prompt)]
                datasets = [ds for ds in datasets for _ in range(args.number_samples_per_prompt)]

            training_time_start = time.time()
            with torch.no_grad():
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                verifiable_counts = []
                sequence_lengths = []
                values = []
                if accelerator.is_main_process:
                    g_response_token_ids = response_ids_Q.get()
                    DUMMY_PAD_TOKEN = 0  # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
                    g_padded_response_ids = [
                        response + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                        for response in g_response_token_ids
                    ]
                    g_padded_response_ids = torch.tensor(g_padded_response_ids, device=device)
                    g_vllm_responses[:] = g_padded_response_ids
                broadcast(g_vllm_responses, 0)
                local_vllm_responses = g_vllm_responses[
                    accelerator.process_index * queries.shape[0] : (accelerator.process_index + 1) * queries.shape[0]
                ]
                query_responses = torch.cat((queries, local_vllm_responses), 1)
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    output = forward(generation_model, query_response, tokenizer.pad_token_id)
                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.temperature + 1e-7
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del output, logits, all_logprob
                    torch.cuda.empty_cache()

                    ref_output = forward(ref_model, query_response, tokenizer.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                    )
                    if args.reward_model_multiplier != 1.0:
                        score *= args.reward_model_multiplier
                    # also apply verifiable reward
                    if args.apply_verifiable_reward:
                        # we need to batch the gt to match query.
                        ground_truth = ground_truths[i : i + args.local_rollout_forward_batch_size]
                        dataset = datasets[i : i + args.local_rollout_forward_batch_size]
                        verifiable_reward, verifiable_count = apply_verifiable_reward(
                            postprocessed_query_response,
                            tokenizer,
                            ground_truth,
                            dataset,
                            verify_reward=10,
                            answer_extraction_model=answer_extraction_model,
                            answer_extraction_tokenizer=answer_extraction_tokenizer,
                        )
                        score += verifiable_reward
                    else:
                        verifiable_count = torch.tensor([0.0], device=device).float()
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, tokenizer.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                    verifiable_counts.append(verifiable_count)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                verifiable_counts = torch.cat(verifiable_counts, 0)
                verifiable_correct_rate = verifiable_counts.sum() / queries.shape[0]
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score)
                gc.collect()
                torch.cuda.empty_cache()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_stop_token = torch.any(postprocessed_responses == args.stop_token_id, dim=-1)
                # NOTE: only apply the stop token filter if the response is long enough
                # otherwise the model could learn to generate the first token as the stop token
                contain_stop_token = contain_stop_token & (sequence_lengths >= args.min_response_length)
                if args.non_stop_penalty:
                    scores = torch.where(
                        contain_stop_token, scores, torch.full_like(scores, args.penalty_reward_value)
                    )

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl1 = logprobs - ref_logprobs
                kl2 = (kl1) ** 2 / 2
                kl3 = (-kl1).exp() - 1 + kl1
                if args.kl_estimator == "kl1":
                    kl = kl1
                elif args.kl_estimator == "kl2":
                    kl = kl2
                elif args.kl_estimator == "kl3":
                    kl = kl3
                # print(f"{accelerator.local_process_index=}, {kl.sum(1)=}")
                non_score_reward = -args.beta * kl
                non_score_reward_sum = non_score_reward.sum(1)
                rlhf_reward = scores + non_score_reward_sum
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        for epoch_idx in range(args.num_epochs):
            b_inds = np.random.permutation(args.local_batch_size * args.number_samples_per_prompt)
            minibatch_idx = 0
            for mini_batch_start in range(
                0, args.local_batch_size * args.number_samples_per_prompt, args.local_mini_batch_size
            ):
                mini_batch_end = mini_batch_start + args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                    with accelerator.accumulate(model):
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]
                        mb_return = returns[micro_batch_inds]
                        mb_values = values[micro_batch_inds]

                        output, vpred_temp = forward(model, mb_query_responses, tokenizer.pad_token_id)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.cliprange_value,
                            mb_values + args.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                        loss = pg_loss + args.vf_coef * vf_loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                            )
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            entropy_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()

                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                # fmt: off
                del (
                    output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                    vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                    pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                    mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                )
                # fmt: on
                # del everything and empty cache
                torch.cuda.empty_cache()
        with torch.no_grad():
            local_metrics[0] = sequence_lengths.float().mean()
            local_metrics[1] = (responses == args.stop_token_id).sum().float().mean()
            local_metrics[2] = kl.sum(1).mean()
            local_metrics[3] = (-logprobs).sum(1).mean()
            local_metrics[4] = non_score_reward_sum.mean()
            local_metrics[5] = rlhf_reward.mean()
            local_metrics[6] = scores.mean()
            local_metrics[7] = approxkl_stats.mean()
            local_metrics[8] = pg_clipfrac_stats.mean()
            local_metrics[9] = pg_loss_stats.mean()
            local_metrics[10] = vf_loss_stats.mean()
            local_metrics[11] = vf_clipfrac_stats.mean()
            local_metrics[12] = entropy_stats.mean()
            local_metrics[13] = ratio_stats.mean()
            local_metrics[14] = ratio_stats.var()
            local_metrics[15] = ((kl) ** 2 / 2).sum(1).mean()
            local_metrics[16] = ((-kl).exp() - 1 + kl).sum(1).mean()
            local_metrics[17] = verifiable_correct_rate
            global_metrics = accelerator.reduce(local_metrics, reduction="mean").tolist()
            metrics = {
                "episode": episode,
                "training_step": training_step,
                "lr": scheduler.get_last_lr()[0],
                "epoch": episode / len(train_dataset),
                "time/from_scratch": time.time() - start_time,
                "time/training": time.time() - training_time_start,
                "val/sequence_lengths": global_metrics[0],
                "val/num_stop_token_ids": global_metrics[1],
                "objective/kl": global_metrics[2],
                "objective/kl2": global_metrics[15],
                "objective/kl3": global_metrics[16],
                "objective/entropy": global_metrics[3],
                "objective/non_score_reward": global_metrics[4],
                "objective/rlhf_reward": global_metrics[5],
                "objective/scores": global_metrics[6],
                "policy/approxkl_avg": global_metrics[7],
                "policy/clipfrac_avg": global_metrics[8],
                "loss/policy_avg": global_metrics[9],
                "loss/value_avg": global_metrics[10],
                "val/clipfrac_avg": global_metrics[11],
                "policy/entropy_avg": global_metrics[12],
                "val/ratio": global_metrics[13],
                "val/ratio_var": global_metrics[14],
                "objective/verifiable_correct_rate": global_metrics[17],
            }
            if accelerator.is_main_process:
                print_rich_single_line_metrics(metrics)
                for key, value in metrics.items():
                    writer.add_scalar(key, value, episode)
        del (queries, responses, postprocessed_responses, logprobs, ref_logprobs, sequence_lengths, scores)
        del (metrics, kl, non_score_reward, rlhf_reward)
        gc.collect()
        torch.cuda.empty_cache()

        # save steps
        if args.save_freq > 0 and training_step % args.save_freq == 0:
            step_dir = os.path.join(args.output_dir, f"step_{training_step}")
            os.makedirs(step_dir, exist_ok=True)
            original_tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_name_or_path, revision=model_config.model_revision
            )
            save_with_accelerate(
                accelerator,
                model,
                original_tokenizer,
                step_dir,
                model_attribute_to_save="policy",
            )
            del original_tokenizer

    # save model
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    original_tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.model_revision
    )
    save_with_accelerate(
        accelerator,
        model,
        original_tokenizer,
        args.output_dir,
        model_attribute_to_save="policy",
    )

    # Ai2 specific logic
    if is_beaker_job() and accelerator.is_main_process:
        if args.hf_metadata_dataset:
            dataset_list = list(args.dataset_mixer_dict.keys())
            # mainly just focussing here on what would be useful for the leaderboard.
            # wandb will have even more useful information.
            metadata_blob = {
                "model_name": args.exp_name,
                "model_type": "sft",
                "datasets": dataset_list,
                "base_model": model_config.model_name_or_path,
                "wandb_path": wandb.run.get_url(),
                "beaker_experiment": beaker_config.beaker_experiment_url,
                "beaker_datasets": beaker_config.beaker_dataset_id_urls,
            }
            upload_metadata_to_hf(
                metadata_blob,
                "metadata.json",
                args.hf_metadata_dataset,
                "results/" + args.hf_repo_revision,  # to match what the auto-evals name as.
            )

        if args.try_launch_beaker_eval_jobs and len(beaker_config.beaker_dataset_id_urls) > 0:
            command = f"""\
            python mason.py  \
                --cluster ai2/allennlp-cirrascale ai2/general-cirrascale-a5000 ai2/general-cirrascale-a5000 ai2/s2-cirrascale ai2/general-cirrascale \
                --priority low \
                --preemptible \
                --budget ai2/allennlp \
                --workspace ai2/tulu-2-improvements \
                --image nathanl/open_instruct_auto \
                --pure_docker_mode \
                --gpus 0 -- python scripts/wait_beaker_dataset_model_upload_then_evaluate_model.py \
                --beaker_workload_id {beaker_config.beaker_workload_id} \
                --model_name {args.hf_repo_revision}
            """
            process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
            print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
            print(f"Submit jobs after model training is finished - process return code: {process.returncode}")

    if args.push_to_hub:
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
        )

    if accelerator.is_main_process:
        # The `checkpoint_output_dir` is only used in case of preemption and should be deleted if the run was successful.
        # We use `--save_freq` to save intermediate checkpoints in the output folder instead.
        if args.checkpoint_output_dir is not None and os.path.exists(args.checkpoint_output_dir):
            shutil.rmtree(args.checkpoint_output_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, ModelConfig))
    main(*parser.parse())
