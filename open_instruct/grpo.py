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
GRPO training with OLMo-core's Trainer.

This module provides GRPO (Group Relative Policy Optimization) training using
OLMo-core's native training infrastructure, replacing DeepSpeed with FSDP.
"""

import dataclasses
import logging
import os
import shutil
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

import ray
import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from olmo_core import train
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, LinearWithWarmup
from olmo_core.train import callbacks
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
from ray.util import queue as ray_queue
from rich.pretty import pprint

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_fast, logger_utils, utils, vllm_utils
from open_instruct.actor_manager import ActorManager
from open_instruct.data_loader import DataPreparationActor
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.ground_truth_utils import RewardConfig, build_all_verifiers
from open_instruct.grpo_callbacks import (
    DataPreparationActorCheckpointCallback,
    RefPolicyUpdateCallback,
    VLLMWeightSyncCallback,
)
from open_instruct.grpo_train_module import GRPOConfig, GRPOTrainModule
from open_instruct.model_utils import ModelConfig, push_folder_to_hub
from open_instruct.tool_utils import tools
from open_instruct.utils import (
    ArgumentParserPlus,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    ray_get_with_progress,
)

logger = logging.getLogger(__name__)


@dataclass
class GRPOExperimentConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    run_name: str | None = None

    learning_rate: float = 2e-5
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    warm_up_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    set_weight_decay_on_bias_and_norm: bool = True
    fused_optimizer: bool = False

    per_device_train_batch_size: int = 1
    total_episodes: int = 100000
    world_size: int | None = None
    num_training_steps: int | None = None
    local_eval_every: int = 100
    save_freq: int = 200
    backend_timeout: int = 120

    num_epochs: int = 1
    num_mini_batches: int = 1
    beta: float = 0.05
    clip_lower: float = 0.2
    clip_higher: float = 0.2
    truncated_importance_sampling_ratio_cap: float = 0.0
    kl_estimator: Literal[0, 1, 2, 3] = 2
    loss_denominator: str = "token"
    alpha: float = 0.6
    ref_policy_update_freq: int | None = None
    load_ref_policy: bool = True
    loss_fn: Literal["dapo", "cispo"] = "dapo"
    record_entropy: bool = False
    use_vllm_logprobs: bool = False

    single_gpu_mode: bool = False
    num_learners_per_node: list[int] = field(default_factory=lambda: [1])
    deepspeed_stage: int = 0
    deepspeed_zpg: int = 8
    deepspeed_offload_param: bool = False
    deepspeed_offload_optimizer: bool = False
    gather_whole_model: bool = True
    enable_queue_dashboard: bool = True
    queue_dashboard_port: int | None = None

    verbose: bool = False
    with_tracking: bool = False
    wandb_project_name: str = "open_instruct_internal"
    wandb_entity: str | None = None
    push_to_hub: bool = True
    hf_entity: str | None = None
    hf_repo_id: str | None = None
    hf_repo_revision: str | None = None
    hf_repo_url: str | None = None
    output_dir: str = "output"
    save_traces: bool = False
    cache_dataset_only: bool = False
    keep_last_n_checkpoints: int = 3
    checkpoint_state_freq: int = -1
    checkpoint_state_dir: str | None = None
    gs_checkpoint_state_dir: str | None = None

    try_launch_beaker_eval_jobs_on_weka: bool = False
    try_auto_save_to_beaker: bool = True
    gs_bucket_path: str | None = None
    oe_eval_tasks: list[str] | None = None
    oe_eval_max_length: int = 4096
    oe_eval_beaker_image: str | None = None
    oe_eval_gpu_multiplier: int | None = None
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    eval_workspace: str = "ai2/tulu-3-results"
    send_slack_alerts: bool = False

    eval_on_step_0: bool = False

    def __post_init__(self):
        if self.use_vllm_logprobs and self.truncated_importance_sampling_ratio_cap > 0.0:
            raise ValueError("Cannot use both `use_vllm_logprobs` and `truncated_importance_sampling_ratio_cap`.")
        self.loss_denominator = utils.get_denominator(self.loss_denominator)
        if not self.load_ref_policy and self.beta != 0.0:
            raise ValueError(
                "When load_ref_policy=False, beta must be 0.0. "
                f"Got beta={self.beta}. Set --beta 0.0 or --load_ref_policy to use KL penalty."
            )

    @property
    def grpo_config(self) -> GRPOConfig:
        return GRPOConfig(
            beta=self.beta,
            clip_lower=self.clip_lower,
            clip_higher=self.clip_higher,
            num_epochs=self.num_epochs,
            num_mini_batches=self.num_mini_batches,
            alpha=self.alpha,
            loss_fn=self.loss_fn,
            kl_estimator=self.kl_estimator,
            load_ref_policy=self.load_ref_policy,
            truncated_importance_sampling_ratio_cap=self.truncated_importance_sampling_ratio_cap,
            use_vllm_logprobs=self.use_vllm_logprobs,
            record_entropy=self.record_entropy,
            loss_denominator=self.loss_denominator if isinstance(self.loss_denominator, float) else None,
        )


def setup_experiment_tracking(args: GRPOExperimentConfig, tc: TokenizerConfig, model_config: ModelConfig):
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()

    if args.push_to_hub:
        if args.hf_repo_id is None:
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.exp_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()

    return beaker_config


def create_generation_config(
    args: GRPOExperimentConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
):
    return vllm_utils.SamplingConfig(
        temperature=streaming_config.temperature,
        top_p=vllm_config.vllm_top_p,
        max_tokens=streaming_config.response_length,
        n=streaming_config.num_samples_per_prompt_rollout,
        stop=streaming_config.stop_strings,
        seed=args.seed,
        logprobs=1,
    )


def setup_tools(streaming_config: data_loader_lib.StreamingDataLoaderConfig) -> dict[str, tools.Tool]:
    tool_objects: dict[str, tools.Tool] = {}
    if streaming_config.tools:
        for tool in streaming_config.tools:
            if tool.lower() == "search":
                from open_instruct.search_utils.search_tool import SearchTool

                search_tool = SearchTool(
                    start_str="<query>",
                    end_str="</query>",
                    api_endpoint=streaming_config.search_api_endpoint,
                    number_documents_to_search=streaming_config.number_documents_to_search,
                )
                tool_objects[search_tool.end_str] = search_tool
                streaming_config.stop_strings.append(search_tool.end_str)
            elif tool.lower() == "code":
                from open_instruct.tool_utils.tools import PythonCodeTool

                code_tool = PythonCodeTool(
                    start_str="<code>", end_str="</code>", api_endpoint=streaming_config.code_tool_api_endpoint
                )
                tool_objects[code_tool.end_str] = code_tool
                streaming_config.stop_strings.append(code_tool.end_str)
    return tool_objects


def main(
    args: GRPOExperimentConfig,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
) -> None:
    """Main entry point for GRPO training with OLMo-core Trainer."""
    if args.single_gpu_mode:
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
    backend = "cpu:gloo,cuda:nccl"  # Always initialize distributed (even single GPU)
    train.prepare_training_environment(seed=args.seed, backend=backend)

    rank = get_rank() if is_distributed() else 0
    world_size = get_world_size() if is_distributed() else 1
    is_main_process = rank == 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger_utils.setup_logger(rank=rank)

    tokenizer = grpo_fast.make_tokenizer(tc, model_config)

    args.num_training_steps = args.total_episodes // (
        streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout
    )

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    beaker_config = setup_experiment_tracking(args, tc, model_config)
    train_dataset, eval_dataset = grpo_fast.setup_datasets(args, tc, tokenizer, streaming_config)

    if len(train_dataset) < (
        needed := max(streaming_config.async_steps, 1) * streaming_config.num_unique_prompts_rollout
    ):
        raise ValueError(f"Train dataset is too small! Is {len(train_dataset)} prompts, but {needed} are needed.")

    if args.cache_dataset_only:
        logger.info("Dataset cached. Exiting because --cache_dataset_only was set.")
        train.teardown_training_environment()
        return

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    pprint([args, model_config])

    ray.init(dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)})

    num_eval_prompts = len(eval_dataset) if eval_dataset is not None else 0
    queue_size = (streaming_config.async_steps + 1) * streaming_config.num_unique_prompts_rollout + num_eval_prompts
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    prompt_Q = ray_queue.Queue(maxsize=queue_size)
    evaluation_inference_results_Q = ray_queue.Queue()

    reward_config = RewardConfig(
        apply_r1_style_format_reward=streaming_config.apply_r1_style_format_reward,
        r1_style_format_reward=streaming_config.r1_style_format_reward,
        apply_verifiable_reward=streaming_config.apply_verifiable_reward,
        verification_reward=streaming_config.verification_reward,
        non_stop_penalty=streaming_config.non_stop_penalty,
        non_stop_penalty_value=streaming_config.non_stop_penalty_value,
        only_reward_good_outputs=streaming_config.only_reward_good_outputs,
        additive_format_reward=streaming_config.additive_format_reward,
        verifier_functions=build_all_verifiers(args, streaming_config),
    )
    generation_config = create_generation_config(args, streaming_config, vllm_config)

    queues_to_monitor = {
        "Inference Results Queue": inference_results_Q,
        "Prompt Queue": prompt_Q,
        "Evaluation Queue": evaluation_inference_results_Q,
    }
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args, streaming_config, vllm_config)
    model_dims = utils.ModelDims.from_hf_config(model_config.model_name_or_path)

    data_prep_actor_name = "data_prep_singleton"
    _data_prep_actor = DataPreparationActor.options(name=data_prep_actor_name, num_cpus=2).remote(
        dataset=train_dataset,
        inference_results_Q=inference_results_Q,
        param_prompt_Q=prompt_Q,
        tokenizer=tokenizer,
        config=streaming_config,
        generation_config=generation_config,
        num_training_steps=args.num_training_steps,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        global_batch_size=streaming_config.num_unique_prompts_rollout,
        dp_world_size=world_size,
        max_possible_score=streaming_config.max_possible_score,
        actor_manager=actor_manager,
        model_dims=model_dims,
        verbose=args.verbose,
        work_dir=args.output_dir,
        initial_state=None,
        allow_world_padding=False,
    )

    tool_objects = setup_tools(streaming_config)

    vllm_engines = vllm_utils.create_vllm_engines(
        vllm_config.vllm_num_engines,
        vllm_config.vllm_tensor_parallel_size,
        vllm_config.vllm_enforce_eager,
        tc.tokenizer_name_or_path,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        vllm_config.vllm_enable_prefix_caching,
        streaming_config.max_prompt_token_length + streaming_config.response_length,
        vllm_config.vllm_gpu_memory_utilization,
        False,
        pg=None,
        tools=tool_objects,
        max_tool_calls=streaming_config.max_tool_calls,
        mask_tool_use=streaming_config.mask_tool_use,
        prompt_queue=prompt_Q,
        results_queue=inference_results_Q,
        eval_results_queue=evaluation_inference_results_Q,
        actor_manager=actor_manager,
        inflight_updates=streaming_config.inflight_updates,
        reward_config=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    logger.info("======== vLLM engines initialized =========")

    if vllm_engines:
        kv_cache_max_concurrency = ray.get(vllm_engines[0].get_kv_cache_info.remote())
        ray.get(actor_manager.set_kv_cache_max_concurrency.remote(kv_cache_max_concurrency))
    else:
        ray.get(actor_manager.set_kv_cache_max_concurrency.remote(-1))

    logger.info("Building OLMo-core model with Qwen3-1.7B config")
    model_config_olmo = TransformerConfig.qwen3_1_7B()
    model = model_config_olmo.build(init_device="cpu")

    logger.info(f"Loading HuggingFace weights from {model_config.model_name_or_path}")
    load_hf_model(model_config.model_name_or_path, model.state_dict(), work_dir=args.output_dir)
    model = model.to(device=device, dtype=torch.bfloat16)

    ref_policy = None
    if args.load_ref_policy:
        logger.info("Building reference policy...")
        ref_policy = model_config_olmo.build(init_device="cpu")
        load_hf_model(model_config.model_name_or_path, ref_policy.state_dict(), work_dir=args.output_dir)
        ref_policy = ref_policy.to(device=device, dtype=torch.bfloat16).eval()

    streaming_dataloader = streaming_config.build_dataloader(
        data_prep_actor_name=data_prep_actor_name,
        tokenizer=tokenizer,
        dp_rank=rank,
        fs_local_rank=rank,
        num_training_steps=args.num_training_steps,
        work_dir=args.output_dir,
        dp_world_size=world_size,
    )

    num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
    warmup_steps = args.warm_up_steps
    if args.warmup_ratio > 0.0:
        warmup_steps = int(num_scheduler_steps * args.warmup_ratio)

    if args.lr_scheduler_type == "cosine":
        scheduler = CosWithWarmup(warmup_steps=warmup_steps)
    else:
        scheduler = LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)

    optim_config = AdamWConfig(lr=args.learning_rate, weight_decay=args.weight_decay)

    dp_config = None
    if not args.single_gpu_mode:
        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        )

    grpo_config = args.grpo_config
    grpo_config.temperature = streaming_config.temperature

    train_module = GRPOTrainModule(
        model=model,
        optim=optim_config,
        rank_microbatch_size=args.per_device_train_batch_size,
        max_sequence_length=streaming_config.max_prompt_token_length + streaming_config.response_length,
        grpo_config=grpo_config,
        tokenizer=tokenizer,
        ref_policy=ref_policy,
        dp_config=dp_config,
        max_grad_norm=args.max_grad_norm,
        scheduler=scheduler,
        device=device,
    )

    model_update_group = None
    if vllm_engines and rank == 0:
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        vllm_world_size = vllm_config.vllm_num_engines * vllm_config.vllm_tensor_parallel_size + 1
        backend = vllm_config.vllm_sync_backend
        refs = [
            engine.init_process_group.remote(
                master_address,
                master_port,
                i * vllm_config.vllm_tensor_parallel_size + 1,
                vllm_world_size,
                "openrlhf",
                backend=backend,
                timeout_minutes=args.backend_timeout,
            )
            for i, engine in enumerate(vllm_engines)
        ]
        model_update_group = vllm_utils.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=vllm_world_size,
            rank=0,
            group_name="openrlhf",
            timeout=timedelta(minutes=args.backend_timeout),
        )
        ray_get_with_progress(refs, desc="Initializing vLLM process groups", timeout=600)
    if is_distributed():
        dist.barrier()

    json_config = dataclasses.asdict(args)
    trainer_callbacks: dict[str, callbacks.Callback] = {}

    trainer_callbacks["vllm_sync"] = VLLMWeightSyncCallback(
        vllm_engines=vllm_engines,
        model_update_group=model_update_group,
        actor_manager=None,
        gather_whole_model=args.gather_whole_model,
    )

    if args.load_ref_policy and args.ref_policy_update_freq:
        trainer_callbacks["ref_policy"] = RefPolicyUpdateCallback(
            ref_policy=ref_policy, alpha=args.alpha, update_interval=args.ref_policy_update_freq
        )

    trainer_callbacks["data_prep"] = DataPreparationActorCheckpointCallback(data_prep_actor_name=data_prep_actor_name)

    trainer_callbacks["checkpointer"] = CheckpointerCallback(
        save_interval=args.checkpoint_state_freq if args.checkpoint_state_freq > 0 else args.save_freq, save_async=True
    )

    trainer_callbacks["speed_monitor"] = callbacks.SpeedMonitorCallback(
        num_flops_per_token=model.num_flops_per_token(
            streaming_config.max_prompt_token_length + streaming_config.response_length
        )
    )
    trainer_callbacks["gpu_memory"] = callbacks.GPUMemoryMonitorCallback()

    if beaker_config is not None:
        trainer_callbacks["beaker"] = callbacks.BeakerCallback(config=json_config)

    if args.with_tracking:
        trainer_callbacks["wandb"] = callbacks.WandBCallback(
            name=args.run_name or args.exp_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=json_config,
        )

    trainer = train.TrainerConfig(
        save_folder=args.output_dir,
        max_duration=train.Duration.steps(args.num_training_steps),
        metrics_collect_interval=10,
        callbacks=trainer_callbacks,
    ).build(train_module, streaming_dataloader)

    logger.info("Starting OLMo-core GRPO training...")
    trainer.fit()
    logger.info("Training complete.")

    if args.push_to_hub and is_main_process:
        push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and beaker_config is not None
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
        and os.path.isdir(args.output_dir)
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if is_beaker_job() and is_main_process and args.try_launch_beaker_eval_jobs:
        wandb_url = None
        if args.with_tracking:
            wandb_tracker = trainer_callbacks.get("wandb")
            if wandb_tracker is not None and hasattr(wandb_tracker, "run") and wandb_tracker.run is not None:
                wandb_url = wandb_tracker.run.get_url()
        if args.hf_repo_revision is not None:
            eval_path = args.output_dir
            if beaker_config is not None and beaker_config.beaker_dataset_ids:
                eval_path = beaker_config.beaker_dataset_ids[-1]
            launch_ai2_evals_on_weka(
                path=eval_path,
                leaderboard_name=args.hf_repo_revision,
                oe_eval_max_length=args.oe_eval_max_length,
                wandb_url=wandb_url,
                oe_eval_tasks=args.oe_eval_tasks,
                gs_bucket_path=args.gs_bucket_path,
                eval_workspace=args.eval_workspace,
                eval_priority=args.eval_priority,
                oe_eval_gpu_multiplier=args.oe_eval_gpu_multiplier,
            )

    train.teardown_training_environment()
    logger.info("Finished GRPO training")


if __name__ == "__main__":
    parser = ArgumentParserPlus(
        (
            GRPOExperimentConfig,
            TokenizerConfig,
            ModelConfig,
            data_loader_lib.StreamingDataLoaderConfig,
            data_loader_lib.VLLMConfig,
        )
    )
    args, tc, model_config, streaming_config, vllm_config = parser.parse_args_into_dataclasses()
    main(args, tc, model_config, streaming_config, vllm_config)
