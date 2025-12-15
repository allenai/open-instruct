#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "open-instruct-integration-test"):
    print(f"Using Beaker image: {beaker_image}")

    model_name_or_path = "allenai/Olmo-3-1025-7B"
    exp_name = "olmo3_7b_rlzero_if"

    dataset_config = DatasetConfig(
        dataset_mixer_list=["allenai/Dolci-RLZero-IF-7B", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["allenai/Dolci-RLZero-IF-7B", "8"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name=exp_name,
        beta=0.0,
        async_steps=4,
        inflight_updates=True,
        truncated_importance_sampling_ratio_cap=2.0,
        num_samples_per_prompt_rollout=8,
        num_unique_prompts_rollout=32,
        num_mini_batches=1,
        num_epochs=1,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        kl_estimator=2,
        max_token_length=10240,
        response_length=16384,
        pack_length=18432,
        non_stop_penalty=False,
        temperature=1.0,
        total_episodes=10000000,
        deepspeed_stage=3,
        num_learners_per_node=[8],
        vllm_num_engines=32,
        vllm_tensor_parallel_size=1,
        lr_scheduler_type="constant",
        apply_verifiable_reward=True,
        seed=1,
        local_eval_every=50,
        save_freq=50,
        checkpoint_state_freq=200,
        with_tracking=True,
        vllm_enable_prefix_caching=True,
        clip_higher=0.272,
        keep_last_n_checkpoints=-1,
        mask_truncated_completions=True,
        oe_eval_max_length=16384,
        code_api_url="https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program",
        try_launch_beaker_eval_jobs_on_weka=True,
        oe_eval_tasks="ifeval::hamish_zs_reasoning_deepseek",
        eval_on_step_0=True,
        oe_eval_beaker_image="oe-eval-beaker/oe_eval_olmo2_retrofit_auto",
        allow_world_padding=True,
        output_dir="/output/olmo3-7b-rlzero-if/checkpoints",
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
        num_nodes=5,
        gpus=8,
        env=[
            {"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"},
            {"name": "VLLM_ATTENTION_BACKEND", "value": "FLASH_ATTN"},
        ],
    )

    launch_on_beaker(experiment, launch_config)


if __name__ == "__main__":
    fire.Fire(main)
