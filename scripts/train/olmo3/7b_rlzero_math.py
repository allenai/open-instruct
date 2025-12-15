#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "open-instruct-integration-test"):
    print(f"Using Beaker image: {beaker_image}")

    model_name_or_path = "allenai/Olmo-3-1025-7B"
    exp_name = "olmo3_7b_rlzero_math"

    dataset_config = DatasetConfig(
        dataset_mixer_list=["allenai/Dolci-RLZero-Math-7B", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=[
            "allenai/aime2024-25-rlvr",
            "1.0",
            "allenai/aime2024-25-rlvr",
            "1.0",
        ],
        dataset_mixer_list_splits=["test_2024", "test_2024", "test_2025", "test_2025"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name=exp_name,
        beta=0.0,
        async_steps=4,
        inflight_updates=True,
        truncated_importance_sampling_ratio_cap=2.0,
        advantage_normalization_type="centered",
        active_fill_completions=True,
        no_resample_solve_rate=0.9,
        num_samples_per_prompt_rollout=16,
        num_unique_prompts_rollout=16,
        num_mini_batches=1,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        kl_estimator=2,
        response_length=12000,
        pack_length=32768,
        non_stop_penalty=False,
        temperature=1.0,
        total_episodes=512000,
        deepspeed_stage=3,
        num_learners_per_node=[8],
        vllm_num_engines=64,
        vllm_tensor_parallel_size=1,
        lr_scheduler_type="constant",
        apply_verifiable_reward=True,
        seed=1,
        local_eval_every=100,
        save_freq=100,
        checkpoint_state_freq=100,
        with_tracking=True,
        vllm_enable_prefix_caching=True,
        clip_higher=0.272,
        mask_truncated_completions=True,
        oe_eval_max_length=32768,
        try_launch_beaker_eval_jobs_on_weka=True,
        eval_priority="high",
        oe_eval_tasks="aime:zs_cot_r1::pass_at_32_2024_dapo,aime:zs_cot_r1::pass_at_32_2025_dapo",
        oe_eval_gpu_multiplier=4,
        oe_eval_beaker_image="michaeln/oe_eval_rlzero",
        output_dir="/output/olmo3-7b-rlzero-math/checkpoints",
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="olmo_thinker_rlzero",
    )

    model_config = ModelConfig(
        model_name_or_path=model_name_or_path,
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
        cluster=["ai2/augusta"],
        budget="ai2/oe-adapt",
        image=beaker_image,
        task_name=exp_name,
        workspace="ai2/olmo-instruct",
        priority="high",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=8,
        gpus=8,
        env=[
            {"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"},
            {"name": "VLLM_ATTENTION_BACKEND", "value": "FLASH_ATTN"},
        ],
    )

    launch_on_beaker(experiment, launch_config)


if __name__ == "__main__":
    fire.Fire(main)
