#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.model_utils import ModelConfig


def main():
    dataset_config = DatasetConfig(
        dataset_mixer_list=["saurabh5/the-algorithm-python", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["saurabh5/the-algorithm-python", "16"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name="test",
        beta=0.01,
        num_unique_prompts_rollout=48,
        num_samples_per_prompt_rollout=16,
        try_launch_beaker_eval_jobs_on_weka=True,
        kl_estimator=2,
        learning_rate=5e-7,
        max_token_length=4096,
        response_length=2048,
        pack_length=4096,
        apply_verifiable_reward=True,
        code_api_url="http://localhost:1234/test_program",
        code_max_execution_time=1.0,
        non_stop_penalty=True,
        oe_eval_tasks="gsm8k::tulu,bbh:cot-v1::tulu,codex_humaneval::tulu,codex_humanevalplus::tulu,mbppplus::openinstruct,drop::llama3,minerva_math::tulu,ifeval::tulu,popqa::tulu,mmlu:mc::tulu,mmlu:cot::summarize,alpaca_eval_v2::tulu,truthfulqa::tulu,cruxeval_input:pass@5,cruxeval_output:pass@5",
        non_stop_penalty_value=0.0,
        temperature=1.0,
        total_episodes=20000,
        num_training_steps=20000,
        deepspeed_stage=2,
        per_device_train_batch_size=1,
        num_mini_batches=2,
        num_epochs=1,
        vllm_tensor_parallel_size=1,
        vllm_num_engines=1,
        lr_scheduler_type="constant",
        seed=1,
        local_eval_every=200,
        save_freq=40,
        single_gpu_mode=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="tulu",
    )

    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
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
