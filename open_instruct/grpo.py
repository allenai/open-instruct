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
import os
import shutil

import backoff
import ray
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from rich.pretty import pprint

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_fast, grpo_utils, logger_utils, utils, vllm_utils
from open_instruct.actor_manager import ActorManager
from open_instruct.data_loader import DataPreparationActor
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.environments.tools.utils import EnvsConfig
from open_instruct.ground_truth_utils import RewardConfig, build_all_verifiers
from open_instruct.grpo_olmo_core_actor import OLMoCoreModelGroup
from open_instruct.model_utils import ModelConfig, push_folder_to_hub

logger = logger_utils.setup_logger(__name__)

CLUSTER_STARTUP_TIMEOUT_S = 1200


@backoff.on_predicate(backoff.constant, interval=5, max_time=CLUSTER_STARTUP_TIMEOUT_S)
def wait_for_gpus(expected_gpus: int) -> bool:
    """Poll the Ray cluster until ``expected_gpus`` GPUs are available.

    Returns True (stopping the retry) once enough GPUs are found, or
    False to keep retrying.
    """
    resources = ray.cluster_resources()
    available = int(resources.get("GPU", 0))
    logger.info(f"Ray cluster resources: {resources}")
    if available >= expected_gpus:
        logger.info(f"Found {available} GPUs, proceeding with placement group creation")
        return True
    logger.info(f"Only {available} GPUs available, waiting for {expected_gpus}...")
    return False


def save_and_cleanup(
    args: grpo_utils.ExperimentConfig, tc: TokenizerConfig, policy_group, tokenizer, beaker_config
) -> None:
    """Save the final model, optionally push to Hub, and launch eval jobs."""
    final_output_dir = args.output_dir
    utils.ray_get_with_progress(
        [m.save_model.remote(final_output_dir, tc.chat_template_name, tokenizer) for m in policy_group.models],
        desc="Saving final model",
    )

    if args.push_to_hub:
        push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    if (
        args.try_auto_save_to_beaker
        and utils.is_beaker_job()
        and beaker_config is not None
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
        and os.path.isdir(args.output_dir)
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if utils.is_beaker_job() and args.try_launch_beaker_eval_jobs_on_weka and args.hf_repo_revision is not None:
        eval_path = args.output_dir
        if beaker_config is not None and beaker_config.beaker_dataset_ids:
            eval_path = beaker_config.beaker_dataset_ids[-1]
        utils.launch_ai2_evals_on_weka(
            path=eval_path,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=None,
            oe_eval_tasks=args.oe_eval_tasks,
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
    tools_config: EnvsConfig,
) -> None:
    """Main entry point for GRPO training with OLMo-core Trainer using Ray actors.

    This function coordinates distributed GRPO training across multiple GPUs and nodes
    using Ray actors for both training and inference. The same code path is used for
    single GPU mode and multi-node training.
    """
    tokenizer = grpo_fast.make_tokenizer(tc, model_config)

    grpo_fast.setup_runtime_variables(args, streaming_config, tools_config)

    if args.verbose:
        logger.setLevel("DEBUG")

    beaker_config = utils.maybe_get_beaker_config()

    os.makedirs(args.output_dir, exist_ok=True)
    pprint([args, model_config])

    ray_init_kwargs = {
        "dashboard_host": "0.0.0.0",
        "runtime_env": {
            "excludes": [".git/"],
            "env_vars": {k: v for k, v in os.environ.items() if k not in grpo_fast.EXCLUDED_ENV_VARS},
        },
    }
    if ray_address := utils.get_ray_address():
        ray_init_kwargs["address"] = ray_address
    ray.init(**ray_init_kwargs)

    pool_size = tools_config.pool_size
    if pool_size is None:
        pool_size = streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout
    pools, tool_definitions, tool_stop_sequences = grpo_fast.initialize_tools_and_envs(
        tools_config,
        tokenizer,
        pool_size=pool_size,
        dataset_mixer_list=streaming_config.dataset_mixer_list,
        dataset_mixer_list_splits=streaming_config.dataset_mixer_list_splits,
    )
    if tool_stop_sequences:
        streaming_config.stop_strings.extend(tool_stop_sequences)

    train_dataset, eval_dataset = grpo_fast.setup_datasets(
        args,
        tc,
        tokenizer,
        streaming_config,
        tool_definitions if tools_config.pass_tools_to_chat_template else [],
        tools_config.pass_tools_to_chat_template,
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
        verification_reward=int(streaming_config.verification_reward),
        non_stop_penalty=streaming_config.non_stop_penalty,
        non_stop_penalty_value=streaming_config.non_stop_penalty_value,
        only_reward_good_outputs=tools_config.only_reward_good_outputs,
        additive_format_reward=streaming_config.additive_format_reward,
        verifier_functions=build_all_verifiers(args, streaming_config),
    )
    generation_config = grpo_fast.create_generation_configs(args, streaming_config, vllm_config)["train"]

    queues_to_monitor = {
        "Inference Results Queue": inference_results_Q,
        "Prompt Queue": prompt_Q,
        "Evaluation Queue": evaluation_inference_results_Q,
    }
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args, streaming_config, vllm_config)
    assert model_config.model_name_or_path is not None, "model_name_or_path must be set"
    model_dims = utils.ModelDims.from_hf_config(model_config.model_name_or_path)

    data_prep_actor_name = "data_prep_singleton"
    base_env_config = grpo_fast.build_base_env_config(tools_config, pools)

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
        tool_names=tools_config.tool_call_names if tools_config else [],
        run_name=args.run_name,
        model_name=model_config.model_name_or_path,
        base_env_config=base_env_config,
        initial_state=None,
    )

    wait_for_gpus(sum(args.num_learners_per_node))

    bundles = [{"GPU": n, "CPU": n * 10} for n in args.num_learners_per_node]
    logger.info(f"Requesting bundles: {bundles}")
    pg = placement_group(bundles, strategy="SPREAD")
    utils.ray_get_with_progress([pg.ready()], desc="Waiting for placement group")

    assert model_config.attn_implementation is not None
    policy_group = OLMoCoreModelGroup(
        pg=pg,
        num_gpus_per_node=args.num_learners_per_node,
        model_name_or_path=model_config.model_name_or_path,
        grpo_config=args,
        max_sequence_length=streaming_config.max_prompt_token_length + streaming_config.response_length,
        streaming_config=streaming_config,
        vllm_config=vllm_config,
        data_prep_actor_name=data_prep_actor_name,
        tokenizer=tokenizer,
        attn_implementation=model_config.attn_implementation,
    )
    logger.info("======== Policy group created =========")

    model_setup_futures = [m.setup_model.remote() for m in policy_group.models]
    utils.ray_get_with_progress(model_setup_futures, desc="Setting up OLMo-core models")
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
        tool_parser_type=tools_config.tool_parser_type,
        tool_definitions=tool_definitions,
        tool_stop_sequences=tool_stop_sequences,
        max_steps=tools_config.max_steps,
        per_turn_max_tokens=tools_config.per_turn_max_tokens,
        mask_tool_use=streaming_config.mask_tool_use,
        pools=pools,
        prompt_queue=prompt_Q,
        results_queue=inference_results_Q,
        eval_results_queue=evaluation_inference_results_Q,
        actor_manager=actor_manager,
        vllm_attention_backend=vllm_config.vllm_attention_backend,
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

    utils.ray_get_with_progress(
        [m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models],
        desc="Setting up model update group",
    )
    logger.info("======== Model update group setup successfully =========")

    json_config = dataclasses.asdict(args)
    utils.ray_get_with_progress(
        [
            m.setup_callbacks.remote(
                actor_manager=actor_manager,
                with_tracking=args.with_tracking,
                wandb_project=args.wandb_project,
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
    utils.ray_get_with_progress([m.fit.remote() for m in policy_group.models], desc="Running OLMo-core GRPO training")
    logger.info("Training complete.")

    save_and_cleanup(args, tc, policy_group, tokenizer, beaker_config)
    logger.info("Finished GRPO training")


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus(
        [  # ty: ignore[invalid-argument-type]
            grpo_utils.ExperimentConfig,
            TokenizerConfig,
            ModelConfig,
            data_loader_lib.StreamingDataLoaderConfig,
            data_loader_lib.VLLMConfig,
            EnvsConfig,
        ]
    )
    parser.set_defaults(
        exp_name="grpo", warmup_ratio=0.0, max_grad_norm=1.0, per_device_train_batch_size=1, fused_optimizer=False
    )
    args, tc, model_config, streaming_config, vllm_config, tools_config = parser.parse_args_into_dataclasses()

    main(args, tc, model_config, streaming_config, vllm_config, tools_config)  # type: ignore[arg-type]
