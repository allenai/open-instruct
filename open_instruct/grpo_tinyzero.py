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
# ---------------------------------------------------------------------
# Part of the code is adapted from https://github.com/OpenRLHF/OpenRLHF
# which has the following license:
# Copyright [yyyy] [name of copyright owner]
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
# isort: off
from collections import defaultdict
import os
import shutil

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
# isort: on


import logging
import os
import threading
import time
import traceback
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import ray
import torch
import torch.utils
import torch.utils.data
from datasets import load_dataset
from huggingface_hub import HfApi
from ray.util.placement_group import placement_group
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from tqdm import trange
from vllm import SamplingParams

from open_instruct.dataset_transformation import (
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    TokenizerConfig,
    visualize_token,
)
from open_instruct.ground_truth_utils import soft_format_reward_func
from open_instruct.grpo_fast import (
    ModelGroup,
    PolicyTrainerRayProcess,
    ShufflingIterator,
    Timer,
    data_preparation_thread,
    vllm_generate_thread,
)
from open_instruct.model_utils import (
    ModelConfig,
    apply_verifiable_reward,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.utils import (
    ArgumentParserPlus,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)
from open_instruct.vllm_utils2 import create_vllm_engines


api = HfApi()
INVALID_LOGPROB = 1.0
logger = logging.Logger(__name__)


@dataclass
class Args:
    # Dataset
    dataset_name: str = None
    """Dataset name"""
    dataset_split: str = "train"
    """Dataset split"""
    max_token_length: int = 512
    """The maximum token length to use for the dataset"""
    max_prompt_token_length: int = 256
    """The maximum prompt token length to use for the dataset"""

    # Experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """RUNTIME VALUE: A unique name of this run"""

    # Optimizer
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    warmup_ratio: float = 0.0
    """Ratio of warmup steps to total steps (takes precedence over `warm_up_steps`)"""
    weight_decay: float = 0.0
    """Weight decay for AdamW if we apply some."""
    set_weight_decay_on_bias_and_norm: bool = True
    """Whether to set weight decay on bias and norm layers"""
    fused_optimizer: bool = False
    """Whether to use fused optimizer"""

    # Batch sizes
    per_device_train_batch_size: int = 1
    """The forward batch size per device (local_micro_batch_size)"""
    total_episodes: int = 100000
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """RUNTIME VALUE: The number of processes (GPUs) to use"""
    num_training_steps: Optional[int] = None
    """RUNTIME VALUE: The number of training_steps to train"""
    num_evals: int = 10
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """RUNTIME VALUE: The frequency of evaluation steps"""
    save_freq: int = -1
    """How many train steps to save the model"""

    # Generation
    response_length: int = 256
    """the length of the response"""
    temperature: float = 0.7
    """the sampling temperature"""
    num_unique_prompts_rollout: int = 16
    """The number of unique prompts during rollout"""
    num_samples_per_prompt_rollout: int = 4
    """the number of samples to generate per prompt during rollout, useful for easy-star"""
    stop_strings: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings."""

    # Algorithm
    async_mode: bool = True
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""
    num_epochs: int = 1
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    cliprange: float = 0.2
    """the clip range"""
    kl_estimator: Literal["kl1", "kl2", "kl3", "kl4"] = "kl3"
    """the KL estimator to use"""
    pack_length: int = 512
    """the length of the pack (you should prob set to the max length of the model)"""
    masked_mean_axis: Optional[int] = None
    """the axis to compute the mean of the masked values"""

    # Reward
    # -- r1 style format reward
    apply_r1_style_format_reward: bool = False
    """whether to add the R1 style format reward"""
    r1_style_format_reward: float = 0.1
    """the reward value for R1 style format reward"""

    # -- verifiable reward
    apply_verifiable_reward: bool = True
    """whether to apply verifiable reward"""
    verification_reward: float = 0.9
    """the reward value for verifiable responses"""

    # -- non stop penalty
    non_stop_penalty: bool = False
    """whether to penalize responses which did not finish generation"""
    non_stop_penalty_value: float = 0.0
    """the reward value for responses which did not finish generation"""

    # Ray
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    num_learners_per_node: List[int] = field(default_factory=lambda: [1])
    """number of GPU deepspeed learners per node (e.g., --num_learners_per_node 2 4 means 2 learner processes
    on the first node and 4 learner processes on the second node; each process will have 1 GPU)"""
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

    # Experiment tracking
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    with_logging: bool = True
    """Whether to do debug print logging"""
    wandb_project_name: str = None
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = False
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "output"
    """Where to save the model"""
    save_traces: bool = False
    """Whether to save learning data traces"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    # Ai2 specific settings
    is_ai2: bool = False
    """flag for ai2 stuff"""
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = False
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: Optional[str] = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""

    def __post_init__(self):
        if self.single_gpu_mode and self.vllm_gpu_memory_utilization > 0.3:
            self.vllm_gpu_memory_utilization = 0.3
        assert self.num_samples_per_prompt_rollout > 1, "Number of samples per prompt must be greater than 1 for GRPO!"
        assert (
            self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty
        ), "At least one reward must be applied!"
        assert (
            self.pack_length >= self.max_prompt_token_length + self.response_length
        ), "The `pack_length` needs to be greater than the sum of `max_prompt_token_length` and `response_length`!"


def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    SYSTEM_MESSAGE: str,
    PROMPT_TEMPLATE: str,
):
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, continue_final_message=True)
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {
        "prompt": prompt,
        "messages": messages,
        INPUT_IDS_PROMPT_KEY: input_ids,
        GROUND_TRUTHS_KEY: [[target, *numbers]],  # we have to make ground truth length 1
    }


def create_datasets(args: Args, tokenizer):
    # SYSTEM_MESSAGE = "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
    SYSTEM_MESSAGE = (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
    )
    PROMPT_TEMPLATE = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(10 + 2) / 3</answer>."
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.add_column(DATASET_SOURCE_KEY, ["countdown"] * len(dataset))
    dataset = dataset.map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "tokenizer": tokenizer,
            "SYSTEM_MESSAGE": SYSTEM_MESSAGE,
            "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        },
    )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return train_dataset, test_dataset


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig, reward_fn: Callable):
    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != model_config.model_revision
        and tc.tokenizer_name_or_path != model_config.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{model_config.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{model_config.model_name_or_path=}`."""
        logger.warning(warning)
    tokenizer = tc.tokenizer

    # ------------------------------------------------------------
    # Set up runtime variables
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    # args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    # if is_beaker_job():
    #     args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    args.world_size = sum(args.num_learners_per_node)
    args.num_training_steps = args.total_episodes // (
        args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    )
    args.eval_freq = max(1, args.num_training_steps // args.num_evals)
    if args.is_ai2:
        args.try_launch_beaker_eval_jobs_on_weka = args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job()
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
    # if args.with_tracking:
    #     if args.wandb_entity is None:
    #         args.wandb_entity = maybe_use_ai2_wandb_entity()

    # ------------------------------------------------------------
    # Setup experiment tracking and seeds
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(tc), **asdict(model_config))
    if args.with_tracking:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=all_configs,
            name=args.run_name,
            # save_code=True,
            tags=[args.exp_name] + get_wandb_tags(),
        )
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),  # noqa
    )

    # ------------------------------------------------------------
    # Set up datasets
    train_dataset, eval_dataset = create_datasets(args, tokenizer)
    if args.cache_dataset_only:
        return

    # ------------------------------------------------------------
    # Runtime setups and quick logging
    pprint([args, model_config])
    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    # ------------------------------------------------------------
    # Create the model and optimizer
    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    inits = []
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.num_learners_per_node,
        args.single_gpu_mode,
    )
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    )
    max_len = args.max_prompt_token_length + args.response_length
    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.vllm_enforce_eager,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.vllm_enable_prefix_caching,
        max_len,
        args.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
    )
    ray.get(inits)
    logger.info("======== ✅ all models and vLLM engines initialized =========")

    ray.get([m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models])
    logger.info("======== ✅ model update group setup successfully =========")

    # Setup training
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        n=args.num_samples_per_prompt_rollout,
        stop=args.stop_strings,
    )
    eval_generation_config = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        n=1,  # since we are doing greedy sampling, don't need to generate more
        stop=args.stop_strings,
    )
    train_dataset_idxs = np.arange(len(train_dataset))
    iter_dataloader = ShufflingIterator(train_dataset_idxs, args.num_unique_prompts_rollout, seed=args.seed)

    inference_results_Q = Queue(maxsize=1)
    param_prompt_Q = Queue(maxsize=1)
    evaluation_inference_results_Q = Queue(maxsize=1)
    packed_sequences_Q = Queue(maxsize=1)
    queries_prompt_Q = Queue(maxsize=1)
    num_eval_samples = 32

    eval_prompt_token_ids = None
    eval_ground_truths = None
    if eval_dataset is not None:
        eval_prompt_token_ids = eval_dataset[:num_eval_samples][INPUT_IDS_PROMPT_KEY]
        eval_ground_truths = eval_dataset[:num_eval_samples][GROUND_TRUTHS_KEY]
    resume_training_step = 1
    thread = threading.Thread(
        target=vllm_generate_thread,
        args=(
            vllm_engines,
            generation_config,
            eval_generation_config,
            inference_results_Q,
            param_prompt_Q,
            args.num_training_steps,
            eval_prompt_token_ids,
            evaluation_inference_results_Q,
            args.eval_freq,
            resume_training_step,
        ),
    )
    thread.start()
    logger.info("======== ✅ vllm generate thread starts =========")

    packing_thread = threading.Thread(
        target=data_preparation_thread,
        args=(
            reward_fn,
            inference_results_Q,
            packed_sequences_Q,
            queries_prompt_Q,
            args,
            tokenizer,
            args.num_training_steps,
        ),
    )
    packing_thread.start()
    logger.info("======== ✅ data preparation thread starts =========")

    # Send initial data to both threads
    data_next = train_dataset[next(iter_dataloader)]
    queries_next = data_next[INPUT_IDS_PROMPT_KEY]
    ground_truths_next = data_next[GROUND_TRUTHS_KEY]
    datasets_next = data_next[DATASET_SOURCE_KEY]
    queries_prompt_Q.put((queries_next, ground_truths_next, datasets_next))
    param_prompt_Q.put((None, queries_next))

    episode = 0
    num_total_tokens = 0
    start_time = time.time()
    eval_futures = []
    try:
        for training_step in trange(resume_training_step, args.num_training_steps + 1):
            logger.info("-" * 100)
            episode += (
                args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
            )  # each sample is an episode

            # ------------------------------------------------------------------------------------------------
            # Optionally evaluate the model
            try:
                evaluation_responses, _ = evaluation_inference_results_Q.get(timeout=0.01)
                logger.info("[Main Thread] 📊 Evaluation responses received")
                table = {}
                table["prompt"] = tokenizer.batch_decode(eval_prompt_token_ids)
                table["response"] = tokenizer.batch_decode(evaluation_responses)
                table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
                table["ground_truth"] = eval_ground_truths
                df = pd.DataFrame(table)
                if args.with_tracking:
                    wandb.log({"sample_completions": wandb.Table(dataframe=df)})
                else:
                    print_rich_table(df.iloc[:1])
                del table
            except Empty:
                logger.info("[Main Thread] 🙈 Evaluation responses not received")

            # ------------------------------------------------------------------------------------------------
            # Sync weights and send the next batch of prompts to vLLM
            if args.async_mode:
                if training_step != 1:
                    data_next = train_dataset[next(iter_dataloader)]
                    queries_next = data_next[INPUT_IDS_PROMPT_KEY]
                    ground_truths_next = data_next[GROUND_TRUTHS_KEY]
                    datasets_next = data_next[DATASET_SOURCE_KEY]
                    with Timer("[Main Thread] 🔄 Loading weights using shared memory"):
                        ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
                queries_prompt_Q.put((queries_next, ground_truths_next, datasets_next))
                param_prompt_Q.put((None, queries_next))
            else:
                if training_step != 1:
                    # NOTE: important: the indent here is different for sync mode
                    # we also set to use `queries = queries_next` immediately
                    data_next = train_dataset[next(iter_dataloader)]
                    queries_next = data_next[INPUT_IDS_PROMPT_KEY]
                    ground_truths_next = data_next[GROUND_TRUTHS_KEY]
                    datasets_next = data_next[DATASET_SOURCE_KEY]
                    with Timer("🔄 Loading weights using shared memory"):
                        ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
                    queries_prompt_Q.put((queries_next, ground_truths_next, datasets_next))
                    param_prompt_Q.put((None, queries_next))

            # ------------------------------------------------------------------------------------------------
            # Get the packed sequences with advantages from the packing thread
            with Timer("[Main Thread] 📦 Getting packed sequences from thread"):
                packed_data = packed_sequences_Q.get()
                packed_sequences = packed_data["packed_sequences"]
                data_thread_metrics = packed_data["metrics"]
                B = packed_data["B"]
                collated_data = packed_data["collated_data"]
                num_total_tokens += packed_data["num_new_tokens"]

            # Log info about the packed sequences
            logger.info(
                f"Number of training examples per device: {B=}, packed sequence fraction of original sequences: {len(packed_sequences.query_responses) / packed_data['responses_count']}"
            )
            if B == 0:
                logger.info("[Main Thread] 🤡 After packing, there is not enough data to train")
                continue

            # ------------------------------------------------------------------------------------------------
            # Train the model
            with Timer("[Main Thread] 🗡️ Training"):
                metrics_list: List[dict[str, float]] = ray.get(
                    [
                        policy_group.models[i].train.remote(
                            **collated_data[i],
                            pad_token_id=tokenizer.pad_token_id,
                            num_mini_batches=args.num_mini_batches,
                        )
                        for i in range(args.world_size)
                    ]
                )
                average_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
                metrics = {
                    "episode": episode,
                    "training_step": training_step,
                    "val/num_total_tokens": num_total_tokens,
                    "epoch": episode / args.num_samples_per_prompt_rollout / len(train_dataset),
                    "tokens_per_second": num_total_tokens / (time.time() - start_time),
                    **data_thread_metrics,
                    **average_metrics,
                }
                print_rich_single_line_metrics(metrics)
                for key, value in metrics.items():
                    writer.add_scalar(key, value, episode)

                if args.save_freq > 0 and training_step % args.save_freq == 0:
                    with Timer("[Main Thread] 🗡️ Saving model"):
                        checkpoint_dir = f"{args.output_dir}_checkpoints"
                        step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                        logger.info(f"Saving model at step {training_step} to {step_dir}")
                        ray.get([policy_group.models[i].save_model.remote(step_dir) for i in range(args.world_size)])
                        if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                            leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                            eval_futures.extend(
                                [
                                    policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                                        step_dir,
                                        leaderboard_name,
                                        wandb_url,
                                        training_step,
                                    )
                                    for i in range(args.world_size)
                                ]
                            )

        logger.info(f"Saving final model at step {training_step} to {args.output_dir}")
        with Timer("[Main Thread] 🗡️ Saving model"):
            ray.get([policy_group.models[i].save_model.remote(args.output_dir) for i in range(args.world_size)])
            if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                leaderboard_name = args.hf_repo_revision
                eval_futures.extend(
                    [
                        policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                            args.output_dir, leaderboard_name, wandb_url, training_step
                        )
                        for i in range(args.world_size)
                    ]
                )

    except Exception as e:
        logger.info(f"Training error occurred: {str(e)}")
        logger.info(traceback.format_exc())
        ray.shutdown()
        os._exit(1)
        raise  # Re-raise the exception after shutdown

    # Clean up threads
    thread.join()
    logger.info("======== ✅ vllm generate thread ends =========")
    packing_thread.join()
    logger.info("======== ✅ data preparation thread ends =========")
    ray.shutdown()

    # Ai2 logic: we use /output to store the artifacts of the job, so we
    # make a copy of the model to `/output` in the end.
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    logger.info("finished training")

    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        logger.info("Pushing model to hub")
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
        )


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)

    if not args.with_logging:
        logger.setLevel(logging.WARNING)

    def reward_fn(
        responses: List[torch.Tensor],
        decoded_responses: List[str],
        ground_truths: List[Any],
        datasets: List[str],
        finish_reasons: List[str],
    ) -> List[float]:
        scores = [0] * len(decoded_responses)
        metrics = {}
        if args.apply_r1_style_format_reward:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🧮 Calculating format reward"):
                format_scores = soft_format_reward_func(decoded_responses, args.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                np_format_scores = np.array(format_scores)
                metrics["objective/format_reward"] = np_format_scores.mean()
                metrics["objective/format_correct_rate"] = (np_format_scores > 0).mean()

        if args.apply_verifiable_reward:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🏆 Applying verifiable reward"):
                verifiable_rewards, per_func_rewards = apply_verifiable_reward(
                    responses,
                    decoded_responses,
                    ground_truths,
                    datasets,
                    reward_mult=args.verification_reward,
                )
                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")
                for i in range(len(verifiable_rewards)):
                    scores[i] = verifiable_rewards[i] + scores[i]
                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
                # reshuffle around per_func rewards
                per_func_lists = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                # log per function rewards
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"objective/{key}_reward"] = np_value.mean()
                    metrics[f"objective/{key}_correct_rate"] = (np_value > 0.0).mean()

        # this gets applied at the very end since it replaces (rather than adds to) the existing reward.
        if args.non_stop_penalty:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🦖 Applying non stop penalty"):
                assert len(finish_reasons) == len(scores)
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = args.non_stop_penalty_value

        return scores, metrics

    main(args, tokenizer_config, model_config, reward_fn)
