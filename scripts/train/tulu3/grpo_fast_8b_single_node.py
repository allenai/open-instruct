#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "nathanl/open_instruct_auto"):
    print(f"Using Beaker image: {beaker_image}")

    dataset_config = DatasetConfig(
        dataset_mixer_list=["allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", "16"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name="tulu3.1_8b_grpo_fast",
        beta=0.01,
        num_unique_prompts_rollout=64,
        num_samples_per_prompt_rollout=16,
        try_launch_beaker_eval_jobs_on_weka=True,
        kl_estimator=2,
        learning_rate=5e-7,
        max_token_length=2048,
        response_length=2048,
        pack_length=4096,
        apply_verifiable_reward=True,
        non_stop_penalty=True,
        non_stop_penalty_value=0.0,
        temperature=1.0,
        total_episodes=2000000,
        deepspeed_stage=2,
        per_device_train_batch_size=1,
        num_mini_batches=2,
        num_learners_per_node=[6],
        num_epochs=1,
        vllm_tensor_parallel_size=1,
        vllm_num_engines=2,
        lr_scheduler_type="constant",
        seed=1,
        local_eval_every=20,
        save_freq=40,
        gather_whole_model=False,
        with_tracking=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="tulu",
    )

    model_config = ModelConfig(
        model_name_or_path="allenai/Llama-3.1-Tulu-3-8B-DPO",
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
        num_nodes=1,
        gpus=8,
    )

    launch_on_beaker(experiment, launch_config)


if __name__ == "__main__":
    fire.Fire(main)
