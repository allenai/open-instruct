#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "saurabhs/code_dev"):
    print(f"Using Beaker image: {beaker_image}")

    base = "SFT"
    description = "test of https://github.com/allenai/open-instruct/pull/631"
    exp_name = f"rlvr_tulu3.1_8b_{base}_grpo_fast_code"

    dataset_config = DatasetConfig(
        dataset_mixer_list=[
            "saurabh5/open-code-reasoning-rlvr",
            "1.0",
            "saurabh5/tulu-3-personas-code-rlvr",
            "1.0",
            "saurabh5/rlvr_acecoder",
            "1.0",
            "saurabh5/the-algorithm-python",
            "1.0",
            "saurabh5/llama-nemotron-rlvr",
            "1.0",
        ],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=[
            "saurabh5/open-code-reasoning-rlvr",
            "16",
            "saurabh5/tulu-3-personas-code-rlvr",
            "16",
            "saurabh5/rlvr_acecoder",
            "16",
            "saurabh5/the-algorithm-python",
            "16",
            "saurabh5/llama-nemotron-rlvr",
            "16",
        ],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name=exp_name,
        beta=0.01,
        num_unique_prompts_rollout=48,
        num_samples_per_prompt_rollout=16,
        try_launch_beaker_eval_jobs_on_weka=True,
        kl_estimator=2,
        learning_rate=5e-7,
        max_token_length=10240,
        response_length=8192,
        pack_length=10240,
        apply_verifiable_reward=True,
        code_api_url="$CODE_API_URL/test_program",
        non_stop_penalty=True,
        oe_eval_tasks="codex_humanevalplus:0-shot-chat-n5,mbppplus::openinstruct,cruxeval_input:pass@5,cruxeval_output:pass@5",
        non_stop_penalty_value=0.0,
        temperature=1.0,
        total_episodes=20000000,
        deepspeed_stage=2,
        per_device_train_batch_size=1,
        num_mini_batches=2,
        num_learners_per_node=[6],
        num_epochs=1,
        vllm_tensor_parallel_size=1,
        vllm_num_engines=10,
        lr_scheduler_type="constant",
        seed=1,
        local_eval_every=125,
        save_freq=200,
        with_tracking=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="tulu_thinker_r1_style",
    )

    model_config = ModelConfig(
        model_name_or_path="allenai/open_instruct_dev",
        model_revision="code_sft_allenai_Llama-3.1-Tulu-3-8B-SFT_n_1__8__1746057575",
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
        description=description,
        workspace="ai2/oe-adapt-code",
        priority="high",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=4,
        gpus=8,
    )

    url = launch_on_beaker(experiment, launch_config)
    print(f"Launched: {url}")


if __name__ == "__main__":
    fire.Fire(main)
