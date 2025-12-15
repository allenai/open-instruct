#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "open-instruct-integration-test"):
    print(f"Using Beaker image: {beaker_image}")

    dataset_config = DatasetConfig(
        dataset_mixer_list=["hamishivi/tulu_3_rewritten_100k_with_tool_prompt", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=512,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["hamishivi/tulu_3_rewritten_100k_with_tool_prompt", "32"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=512,
    )

    args = Args(
        exp_name="0605_general_tool_use_without_good_outputs",
        response_length=512,
        pack_length=1024,
        inflight_updates=True,
        per_device_train_batch_size=1,
        num_unique_prompts_rollout=8,
        num_samples_per_prompt_rollout=4,
        stop_strings=["</answer>"],
        apply_verifiable_reward=True,
        temperature=1.0,
        learning_rate=5e-7,
        total_episodes=640,  # 20 * 8 * 4
        deepspeed_stage=2,
        with_tracking=True,
        num_epochs=1,
        num_learners_per_node=[1],
        vllm_tensor_parallel_size=1,
        beta=0.01,
        seed=1,
        local_eval_every=10,
        vllm_sync_backend="gloo",
        vllm_gpu_memory_utilization=0.3,
        push_to_hub=False,
        single_gpu_mode=True,
        output_dir="/output",
        kl_estimator=2,
        non_stop_penalty=True,
        non_stop_penalty_value=0.0,
        num_mini_batches=1,
        lr_scheduler_type="constant",
        save_freq=100,
        try_launch_beaker_eval_jobs_on_weka=False,
        vllm_num_engines=1,
        max_tool_calls=5,
        vllm_enable_prefix_caching=True,
        tools=["code", "search"],
        search_api_endpoint="http://saturn-cs-aus-248.reviz.ai2.in:47479/search",
        code_tool_api_endpoint="https://open-instruct-tool-server-10554368204.us-central1.run.app/execute",
    )

    tokenizer_config = TokenizerConfig(
        ground_truths_key="ground_truth",
        sft_messages_key="messages",
    )

    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen3-1.7B",
        gradient_checkpointing=True,
    )

    experiment = ExperimentConfig(
        args=args,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        dataset_config=dataset_config,
        eval_dataset_config=eval_dataset_config,
    )

    if run_local:
        return experiment.run()

    launch_config = LaunchConfig(
        cluster=["ai2/jupiter", "ai2/augusta", "ai2/saturn"],
        budget="ai2/oe-adapt",
        image=beaker_image,
        description="Single GPU on Beaker with tool use test script.",
        workspace="ai2/open-instruct-dev",
        priority="urgent",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=1,
        max_retries=0,
        timeout="45m",
        gpus=1,
        no_host_networking=True,
        env=[
            {"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"},
        ],
    )

    launch_on_beaker(experiment, launch_config)


if __name__ == "__main__":
    fire.Fire(main)
