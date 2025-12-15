#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "nathanl/open_instruct_auto"):
    print(f"Using Beaker image: {beaker_image}")

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
        exp_name="olmo2_13b_grpo_fast_zero",
        beta=0.0,
        num_unique_prompts_rollout=48,
        num_samples_per_prompt_rollout=16,
        kl_estimator=2,
        learning_rate=5e-7,
        max_token_length=2048,
        response_length=2048,
        pack_length=4096,
        stop_strings=["</answer>"],
        apply_r1_style_format_reward=True,
        apply_verifiable_reward=True,
        non_stop_penalty=True,
        non_stop_penalty_value=0.0,
        oe_eval_tasks="minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning",
        oe_eval_max_length=8192,
        temperature=1.0,
        total_episodes=1000000,
        deepspeed_stage=3,
        per_device_train_batch_size=1,
        num_mini_batches=1,
        num_learners_per_node=[8],
        num_epochs=1,
        vllm_tensor_parallel_size=1,
        vllm_num_engines=8,
        lr_scheduler_type="linear",
        seed=1,
        local_eval_every=5,
        save_freq=40,
        try_launch_beaker_eval_jobs_on_weka=True,
        with_tracking=True,
    )

    tokenizer_config = TokenizerConfig(
        tokenizer_name_or_path="allenai/OLMo-2-1124-7B",
        tokenizer_revision="main",
        add_bos=True,
        chat_template_name="r1_simple_chat_postpend_think",
    )

    model_config = ModelConfig(
        model_name_or_path="allenai/OLMo-2-1124-13B",
        model_revision="main",
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
        num_nodes=2,
        gpus=8,
    )

    launch_on_beaker(experiment, launch_config)


if __name__ == "__main__":
    fire.Fire(main)
