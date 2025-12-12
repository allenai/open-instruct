#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "open-instruct-integration-test"):
    print(f"Using Beaker image: {beaker_image}")

    dataset_config = DatasetConfig(
        dataset_mixer_list=["ai2-adapt-dev/rlvr_gsm8k_zs", "64"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=512,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["ai2-adapt-dev/rlvr_gsm8k_zs", "16"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=512,
    )

    args = Args(
        response_length=1024,
        pack_length=2048,
        per_device_train_batch_size=1,
        num_unique_prompts_rollout=8,
        num_samples_per_prompt_rollout=4,
        stop_strings=["</answer>"],
        apply_r1_style_format_reward=True,
        apply_verifiable_reward=True,
        temperature=0.7,
        learning_rate=3e-7,
        total_episodes=200,
        deepspeed_stage=2,
        num_epochs=1,
        num_learners_per_node=[1],
        vllm_tensor_parallel_size=1,
        beta=0.01,
        seed=3,
        local_eval_every=1,
        vllm_sync_backend="gloo",
        vllm_gpu_memory_utilization=0.3,
        save_traces=True,
        vllm_enforce_eager=True,
        push_to_hub=False,
        active_sampling=True,
        async_steps=8,
        no_resampling_pass_rate=0.6,
        verbose=True,
        single_gpu_mode=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="r1_simple_chat_postpend_think",
        ground_truths_key="ground_truth",
    )

    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-1.5B",
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
        description="Single GPU on Beaker integration test.",
        workspace="ai2/open-instruct-dev",
        priority="high",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=1,
        max_retries=0,
        gpus=1,
        no_host_networking=True,
        env=[{"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"}],
    )

    url = launch_on_beaker(experiment, launch_config)
    print(f"Launched: {url}")


if __name__ == "__main__":
    fire.Fire(main)
