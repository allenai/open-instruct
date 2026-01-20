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

Uses Ray for distributed training with Beaker.
"""

import dataclasses
import logging
import os
import shutil
import time

import ray
from huggingface_hub import HfApi
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from rich.pretty import pprint

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_fast, grpo_utils, logger_utils, utils, vllm_utils
from open_instruct.actor_manager import ActorManager
from open_instruct.data_loader import DataPreparationActor
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.ground_truth_utils import RewardConfig, build_all_verifiers
from open_instruct.model_utils import ModelConfig, push_folder_to_hub
from open_instruct.tools.parsers import create_tool_parser
from open_instruct.tools.tools import TOOL_REGISTRY
from open_instruct.tools.utils import ParsedToolConfig, ToolsConfig
from open_instruct.utils import (
    ArgumentParserPlus,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    ray_get_with_progress,
)

logger = logger_utils.setup_logger(__name__)


def setup_experiment_tracking(args: grpo_utils.ExperimentConfig, tc: TokenizerConfig, model_config: ModelConfig):
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
    args: grpo_utils.ExperimentConfig,
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


def create_tools(parsed_tools: list[ParsedToolConfig]) -> list[ray.actor.ActorHandle]:
    from dataclasses import asdict

    tool_actors = []
    for parsed_tool in parsed_tools:
        if parsed_tool.name not in TOOL_REGISTRY:
            available_tools = ", ".join(TOOL_REGISTRY.keys())
            raise ValueError(f"Unknown tool: {parsed_tool.name}. Available tools: {available_tools}")

        tool_config_class = TOOL_REGISTRY[parsed_tool.name]
        config = tool_config_class(**parsed_tool.config)
        _kwarg_dict = asdict(config) | {"call_name": parsed_tool.call_name}
        tool_actors.append(ray.remote(tool_config_class.tool_class).options(max_concurrency=512).remote(**_kwarg_dict))

    return tool_actors


def initialize_tools(tools_config: ToolsConfig, tokenizer) -> tuple[list, list, list[str]]:
    tool_actors = create_tools(tools_config._parsed_tools)
    tool_definitions = (
        ray.get([actor.get_openai_tool_definitions.remote() for actor in tool_actors]) if tool_actors else []
    )

    stop_sequences = []
    if tool_actors:
        stop_sequences = create_tool_parser(
            parser_type=tools_config.tool_parser_type, tool_actors=tool_actors, tokenizer=tokenizer
        ).stop_sequences

    return tool_actors, tool_definitions, stop_sequences


def wait_for_gpus(expected_gpus: int, max_attempts: int = 60, poll_interval: int = 5) -> None:
    logger.info(f"Waiting for {expected_gpus} GPUs to be available in Ray cluster...")
    for i in range(max_attempts):
        cluster_resources = ray.cluster_resources()
        available_gpus = cluster_resources.get("GPU", 0)
        logger.info(f"Attempt {i + 1}: Ray cluster resources: {cluster_resources}")
        if available_gpus >= expected_gpus:
            logger.info(f"Found {available_gpus} GPUs, proceeding with placement group creation")
            return
        logger.info(f"Only {available_gpus} GPUs available, waiting for {expected_gpus}...")
        time.sleep(poll_interval)
    logger.error(f"Timeout waiting for GPUs. Only {available_gpus} available, needed {expected_gpus}")


def save_and_cleanup(
    args: grpo_utils.ExperimentConfig, tc: TokenizerConfig, policy_group, tokenizer, beaker_config
) -> None:
    final_output_dir = args.output_dir
    ray_get_with_progress(
        [m.save_model.remote(final_output_dir, tc.chat_template_name, tokenizer) for m in policy_group.models],
        desc="Saving final model",
    )

    if args.push_to_hub:
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

    if is_beaker_job() and args.try_launch_beaker_eval_jobs_on_weka and args.hf_repo_revision is not None:
        eval_path = args.output_dir
        if beaker_config is not None and beaker_config.beaker_dataset_ids:
            eval_path = beaker_config.beaker_dataset_ids[-1]
        launch_ai2_evals_on_weka(
            path=eval_path,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=None,
            oe_eval_tasks=args.oe_eval_tasks,
            gs_bucket_path=args.gs_bucket_path,
            eval_workspace=args.eval_workspace,
            eval_priority=args.eval_priority,
            oe_eval_gpu_multiplier=args.oe_eval_gpu_multiplier,
        )


def main(
    args: grpo_utils.ExperimentConfig,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    tools_config: ToolsConfig,
) -> None:
    """Main entry point for GRPO training with OLMo-core Trainer using Ray actors.

    This function coordinates distributed GRPO training across multiple GPUs and nodes
    using Ray actors for both training and inference. The same code path is used for
    single GPU mode and multi-node training.
    """

    from open_instruct.grpo_olmo_core_actor import OLMoCoreModelGroup

    logger_utils.setup_logger(rank=0)
    tokenizer = grpo_fast.make_tokenizer(tc, model_config)

    args.world_size = sum(args.num_learners_per_node)
    args.num_training_steps = args.total_episodes // (
        streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout
    )

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    beaker_config = setup_experiment_tracking(args, tc, model_config)

    os.makedirs(args.output_dir, exist_ok=True)
    pprint([args, model_config])

    ray.init(
        address="auto", dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)}
    )

    tool_actors, tool_definitions, tool_stop_sequences = initialize_tools(tools_config, tokenizer)
    logger.info(
        f"Initialized {len(tool_actors)} tool actors with definitions: {[d['function']['name'] for d in tool_definitions]}"
    )
    if tool_stop_sequences:
        logger.info(f"Adding tool stop sequences to config: {tool_stop_sequences}")
        streaming_config.stop_strings.extend(tool_stop_sequences)

    train_dataset, eval_dataset = grpo_fast.setup_datasets(
        args,  # type: ignore[arg-type]
        tc,
        tokenizer,
        streaming_config,
        tool_definitions if tools_config.pass_tools_to_chat_template else [],
    )

    if len(train_dataset) < (
        needed := max(streaming_config.async_steps, 1) * streaming_config.num_unique_prompts_rollout
    ):
        raise ValueError(f"Train dataset is too small! Is {len(train_dataset)} prompts, but {needed} are needed.")

    if args.cache_dataset_only:
        logger.info("Dataset cached. Exiting because --cache_dataset_only was set.")
        return

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
        only_reward_good_outputs=tools_config.only_reward_good_outputs,
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
    _data_prep_actor = DataPreparationActor.options(name=data_prep_actor_name, num_cpus=2).remote(  # type: ignore[attr-defined]
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
        dp_world_size=args.world_size,
        max_possible_score=streaming_config.max_possible_score,
        actor_manager=actor_manager,
        model_dims=model_dims,
        verbose=args.verbose,
        work_dir=args.output_dir,
        initial_state=None,
        allow_world_padding=False,
    )

    wait_for_gpus(sum(args.num_learners_per_node))

    bundles = [{"GPU": n, "CPU": n} for n in args.num_learners_per_node]
    logger.info(f"Requesting bundles: {bundles}")
    pg = placement_group(bundles, strategy="SPREAD")
    ray_get_with_progress([pg.ready()], desc="Waiting for placement group")

    policy_group = OLMoCoreModelGroup(
        pg=pg,
        num_gpus_per_node=args.num_learners_per_node,
        single_gpu_mode=args.single_gpu_mode,
        model_name_or_path=model_config.model_name_or_path,
        grpo_config=args,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warm_up_steps=args.warm_up_steps,
        warmup_ratio=args.warmup_ratio,
        num_training_steps=args.num_training_steps,
        num_epochs=args.num_epochs,
        num_mini_batches=args.num_mini_batches,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_sequence_length=streaming_config.max_prompt_token_length + streaming_config.response_length,
        load_ref_policy=args.load_ref_policy,
        beta=args.beta,
        seed=args.seed,
        output_dir=args.output_dir,
        streaming_config=streaming_config,
        vllm_config=vllm_config,
        data_prep_actor_name=data_prep_actor_name,
        tokenizer=tokenizer,
    )
    logger.info("======== Policy group created =========")

    model_setup_futures = [m.setup_model.remote() for m in policy_group.models]
    ray_get_with_progress(model_setup_futures, desc="Setting up OLMo-core models")
    logger.info("======== OLMo-core models initialized =========")

    assert tc.tokenizer_name_or_path is not None, "tokenizer_name_or_path must be set after make_tokenizer"
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
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
        tool_actors=tool_actors,
        tool_parser_type=tools_config.tool_parser_type,
        max_tool_calls=tools_config.max_tool_calls,
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

    ray_get_with_progress(
        [m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models],
        desc="Setting up model update group",
    )
    logger.info("======== Model update group setup successfully =========")

    json_config = dataclasses.asdict(args)
    ray_get_with_progress(
        [
            m.setup_callbacks.remote(
                actor_manager=actor_manager,
                with_tracking=args.with_tracking,
                wandb_project=args.wandb_project_name,
                wandb_entity=args.wandb_entity,
                run_name=args.run_name or args.exp_name,
                json_config=json_config,
                ref_policy_update_freq=args.ref_policy_update_freq,
            )
            for m in policy_group.models
        ],
        desc="Setting up callbacks",
    )

    logger.info("Starting OLMo-core GRPO training with Ray actors...")
    ray_get_with_progress([m.fit.remote() for m in policy_group.models], desc="Running OLMo-core GRPO training")
    logger.info("Training complete.")

    save_and_cleanup(args, tc, policy_group, tokenizer, beaker_config)
    logger.info("Finished GRPO training")


if __name__ == "__main__":
    parser = ArgumentParserPlus(
        [  # ty: ignore[invalid-argument-type]
            grpo_utils.ExperimentConfig,
            TokenizerConfig,
            ModelConfig,
            data_loader_lib.StreamingDataLoaderConfig,
            data_loader_lib.VLLMConfig,
            ToolsConfig,
        ]
    )
    args, tc, model_config, streaming_config, vllm_config, tools_config = parser.parse_args_into_dataclasses()

    main(args, tc, model_config, streaming_config, vllm_config, tools_config)  # type: ignore[arg-type]
