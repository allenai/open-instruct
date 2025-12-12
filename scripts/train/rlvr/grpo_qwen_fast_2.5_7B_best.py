#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False):
    exp_name = "0302_qwen2.5_7B_math_grpo_fast1_1317"

    dataset_config = DatasetConfig(
        dataset_mixer_list=["ai2-adapt-dev/math_ground_truth_zs", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["ai2-adapt-dev/math_ground_truth_zs", "16"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name=exp_name,
        beta=0.0,
        num_samples_per_prompt_rollout=16,
        output_dir=f"/weka/oe-adapt-default/costah/models/{exp_name}",
        oe_eval_tasks="minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning",
        save_freq=40,
        try_launch_beaker_eval_jobs_on_weka=True,
        kl_estimator=2,
        learning_rate=5e-7,
        max_token_length=2048,
        response_length=2048,
        pack_length=4096,
        stop_strings=["</answer>"],
        apply_r1_style_format_reward=True,
        non_stop_penalty=False,
        temperature=1.0,
        total_episodes=1000000,
        deepspeed_stage=2,
        per_device_train_batch_size=1,
        num_mini_batches=1,
        num_learners_per_node=[6],
        num_epochs=1,
        vllm_tensor_parallel_size=1,
        vllm_num_engines=10,
        lr_scheduler_type="linear",
        apply_verifiable_reward=True,
        seed=1,
        local_eval_every=80,
        with_tracking=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="r1_simple_chat_postpend_think",
        ground_truths_key="ground_truth",
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
        workspace="ai2/tulu-3-dev",
        priority="urgent",
        preemptible=True,
        num_nodes=2,
        max_retries=0,
        gpus=8,
    )

    url = launch_on_beaker(experiment, launch_config)
    print(f"Launched: {url}")


if __name__ == "__main__":
    fire.Fire(main)
