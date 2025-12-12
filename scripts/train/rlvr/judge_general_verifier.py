#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "hamishivi/open_instruct_judge_8", judge_base_url: str = ""):
    print(f"Using Beaker image: {beaker_image}")
    if judge_base_url:
        print(f"Using judge base URL: {judge_base_url}")

    exp_name = "0906rl_judge_test"

    dataset_config = DatasetConfig(
        dataset_mixer_list=["hamishivi/WebInstruct-verified-general-verifier-judge", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["hamishivi/WebInstruct-verified-general-verifier-judge", "16"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name=exp_name,
        max_token_length=10240,
        response_length=8192,
        pack_length=16384,
        per_device_train_batch_size=1,
        num_unique_prompts_rollout=64,
        num_samples_per_prompt_rollout=16,
        stop_strings=["</answer>"],
        apply_verifiable_reward=True,
        apply_r1_style_format_reward=True,
        non_stop_penalty=True,
        non_stop_penalty_value=0.0,
        temperature=1.0,
        learning_rate=3e-7,
        total_episodes=200000,
        deepspeed_stage=2,
        num_epochs=1,
        num_learners_per_node=[8],
        vllm_num_engines=8,
        vllm_tensor_parallel_size=1,
        beta=0.0,
        seed=3,
        local_eval_every=10,
        vllm_enforce_eager=True,
        push_to_hub=False,
        llm_judge_timeout=600,
        llm_judge_model="hosted_vllm/hamishivi/general-verifier",
        with_tracking=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="tulu_thinker_r1_style",
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

    env = [{"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"}]
    if judge_base_url:
        env.append({"name": "HOSTED_VLLM_API_BASE", "value": judge_base_url})

    launch_config = LaunchConfig(
        cluster=["ai2/jupiter"],
        budget="ai2/oe-adapt",
        image=beaker_image,
        workspace="ai2/tulu-thinker",
        priority="high",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=2,
        max_retries=0,
        gpus=8,
        env=env,
    )

    url = launch_on_beaker(experiment, launch_config)
    print(f"Launched: {url}")


if __name__ == "__main__":
    fire.Fire(main)
