#!/usr/bin/env python
import json
import subprocess
import sys

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def get_beaker_user() -> str:
    result = subprocess.run(
        ["beaker", "account", "whoami", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return data[0]["name"]


def main():
    beaker_user = get_beaker_user()
    beaker_image = sys.argv[1] if len(sys.argv) > 1 else f"{beaker_user}/open-instruct-integration-test"

    print(f"Using Beaker image: {beaker_image}")

    args = Args(
        dataset_mixer_list=["ai2-adapt-dev/rlvr_gsm8k_zs", "64"],
        dataset_mixer_list_splits=["train"],
        dataset_mixer_eval_list=["ai2-adapt-dev/rlvr_gsm8k_zs", "16"],
        dataset_mixer_eval_list_splits=["train"],
        max_prompt_token_length=512,
        response_length=512,
        pack_length=1024,
        per_device_train_batch_size=1,
        num_unique_prompts_rollout=8,
        num_samples_per_prompt_rollout=4,
        stop_strings=["</answer>"],
        apply_r1_style_format_reward=True,
        apply_verifiable_reward=True,
        temperature=0.7,
        inflight_updates=True,
        learning_rate=3e-7,
        total_episodes=200,
        deepspeed_stage=2,
        with_tracking=True,
        num_epochs=1,
        num_learners_per_node=[1],
        vllm_tensor_parallel_size=1,
        beta=0.0,
        load_ref_policy=True,
        seed=3,
        local_eval_every=1,
        vllm_sync_backend="gloo",
        vllm_gpu_memory_utilization=0.3,
        save_traces=True,
        vllm_enforce_eager=True,
        push_to_hub=False,
        single_gpu_mode=True,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="r1_simple_chat_postpend_think",
        ground_truths_key="ground_truth",
    )

    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen3-1.7B",
        gradient_checkpointing=True,
    )

    experiment = ExperimentConfig(
        args=args,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
    )

    launch_config = LaunchConfig(
        cluster=["ai2/jupiter", "ai2/saturn", "ai2/ceres"],
        budget="ai2/oe-adapt",
        image=beaker_image,
        description="Single GPU on Beaker test script.",
        workspace="ai2/open-instruct-dev",
        priority="urgent",
        pure_docker_mode=True,
        num_nodes=1,
        max_retries=0,
        timeout="15m",
        gpus=1,
        env=[{"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"}],
    )

    url = launch_on_beaker(experiment, launch_config)
    print(f"Launched: {url}")


if __name__ == "__main__":
    main()
