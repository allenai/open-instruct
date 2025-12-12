#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "open-instruct-integration-test"):
    print(f"Using Beaker image: {beaker_image}")

    num_prompts = 25376
    exp_name = "rlvr_ace_fn_and_og_ocr_stdio_from_base_with_perf_penalty"

    dataset_config = DatasetConfig(
        dataset_mixer_list=[
            "saurabh5/rlvr_acecoder_filtered",
            str(num_prompts),
            "saurabh5/open-code-reasoning-rlvr-stdio",
            str(num_prompts),
        ],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=[
            "saurabh5/rlvr_acecoder_filtered",
            "8",
            "saurabh5/open-code-reasoning-rlvr-stdio",
            "8",
        ],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name=exp_name,
        beta=0.0,
        load_ref_policy=False,
        num_samples_per_prompt_rollout=16,
        num_unique_prompts_rollout=32,
        num_mini_batches=1,
        num_epochs=1,
        learning_rate=5e-7,
        per_device_train_batch_size=1,
        kl_estimator=2,
        response_length=4096,
        pack_length=20480,
        inflight_updates=True,
        stop_strings=["</answer>"],
        non_stop_penalty=False,
        temperature=1.0,
        total_episodes=10000,
        deepspeed_stage=2,
        num_learners_per_node=[8],
        vllm_num_engines=8,
        vllm_tensor_parallel_size=1,
        lr_scheduler_type="constant",
        apply_verifiable_reward=True,
        code_api_url="$CODE_API_URL/test_program",
        seed=1,
        local_eval_every=1,
        try_launch_beaker_eval_jobs_on_weka=True,
        with_tracking=True,
        vllm_enable_prefix_caching=True,
        oe_eval_max_length=32768,
        oe_eval_tasks="codex_humanevalplus:0-shot-chat-v1::tulu-thinker,mbppplus:0-shot-chat::tulu-thinker,livecodebench_codegeneration::tulu-thinker",
        dataset_skip_cache=True,
        active_sampling=True,
        async_steps=4,
        push_to_hub=False,
        verbose=False,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="tulu_thinker",
        ground_truths_key="ground_truth",
        sft_messages_key="messages",
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
        description="Large (multi-node) test script.",
        workspace="ai2/open-instruct-dev",
        priority="urgent",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=2,
        max_retries=0,
        timeout="1h",
        gpus=8,
        env=[{"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"}],
    )

    url = launch_on_beaker(experiment, launch_config)
    print(f"Launched: {url}")


if __name__ == "__main__":
    fire.Fire(main)
