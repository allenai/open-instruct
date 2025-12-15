#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "nathanl/open_instruct_auto"):
    print(f"Using Beaker image: {beaker_image}")

    dataset_config = DatasetConfig(
        dataset_mixer_list=["vwxyzjn/rlvr_acecoder", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["vwxyzjn/rlvr_acecoder", "16"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name="qwen2.5_7b_grpo_fast_zero_orz",
        beta=0.0,
        num_unique_prompts_rollout=128,
        num_samples_per_prompt_rollout=64,
        kl_estimator=2,
        learning_rate=5e-7,
        max_token_length=8192,
        response_length=8192,
        pack_length=16384,
        stop_strings=["</answer>"],
        apply_r1_style_format_reward=True,
        apply_verifiable_reward=True,
        code_api_url="$CODE_API_URL/test_program",
        non_stop_penalty=True,
        non_stop_penalty_value=0.0,
        oe_eval_tasks="gsm8k::tulu,bbh:cot-v1::tulu,codex_humaneval::tulu,codex_humanevalplus::tulu,mbppplus::openinstruct",
        oe_eval_max_length=8192,
        temperature=1.0,
        masked_mean_axis=1,
        total_episodes=10000000,
        deepspeed_stage=2,
        per_device_train_batch_size=1,
        num_mini_batches=1,
        num_learners_per_node=[8, 8],
        num_epochs=1,
        vllm_tensor_parallel_size=1,
        vllm_num_engines=16,
        lr_scheduler_type="constant",
        seed=3,
        local_eval_every=5,
        save_freq=40,
        try_launch_beaker_eval_jobs_on_weka=True,
        with_tracking=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="r1_simple_chat_postpend_think",
    )

    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-7B",
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
        cluster=["ai2/jupiter"],
        budget="ai2/oe-adapt",
        image=beaker_image,
        workspace="ai2/tulu-3-dev",
        priority="high",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=4,
        gpus=8,
    )

    launch_on_beaker(experiment, launch_config)


if __name__ == "__main__":
    fire.Fire(main)
