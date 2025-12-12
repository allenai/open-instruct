#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.model_utils import ModelConfig


def main():
    exp_name = "base_smollm_grpo"

    dataset_config = DatasetConfig(
        dataset_mixer_list=["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=256,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=256,
    )

    args = Args(
        exp_name=exp_name,
        output_dir="output/dummy",
        max_token_length=256,
        response_length=128,
        pack_length=1024,
        per_device_train_batch_size=4,
        num_samples_per_prompt_rollout=4,
        stop_strings=["</answer>"],
        apply_r1_style_format_reward=True,
        non_stop_penalty=False,
        non_stop_penalty_value=0.0,
        temperature=0.7,
        learning_rate=3e-7,
        total_episodes=10000,
        deepspeed_stage=2,
        num_epochs=1,
        num_learners_per_node=[1],
        vllm_tensor_parallel_size=1,
        beta=0.01,
        apply_verifiable_reward=True,
        seed=3,
        local_eval_every=150,
        try_launch_beaker_eval_jobs_on_weka=False,
        vllm_sync_backend="nccl",
        vllm_gpu_memory_utilization=0.9,
        vllm_enforce_eager=True,
        single_gpu_mode=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="r1_simple_chat_postpend_think",
        ground_truths_key="ground_truth",
    )

    model_config = ModelConfig(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M",
        gradient_checkpointing=True,
    )

    experiment = ExperimentConfig(
        args=args,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        dataset_config=dataset_config,
        eval_dataset_config=eval_dataset_config,
    )

    experiment.run()


if __name__ == "__main__":
    fire.Fire(main)
