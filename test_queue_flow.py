#!/usr/bin/env python3
"""
Simple test to verify queue flow logging in grpo_fast.py
"""

import sys
import subprocess

def test_queue_flow():
    """Run a minimal training to see queue flow."""
    
    # Create a minimal config to test with 1 step
    test_config = """
{
    "model_name_or_path": "EleutherAI/pythia-14m",
    "dataset_name": "test",
    "num_training_steps": 1,
    "vllm_num_engines": 1,
    "num_unique_prompts_rollout": 4,
    "num_samples_per_prompt_rollout": 1,
    "async_mode": false,
    "per_device_train_batch_size": 1,
    "output_dir": "./test_output",
    "exp_name": "queue_test",
    "seed": 42,
    "learning_rate": 1e-5,
    "warmup_ratio": 0.1,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": false,
    "num_mini_batches": 1,
    "pack_length": 1024,
    "local_eval_freq": 1,
    "checkpoint_state_freq": 0,
    "save_freq": 0,
    "eval_freq": 1,
    "num_epochs": 1,
    "weight_decay": 0.0,
    "alpha": 0.1,
    "ref_policy_update_freq": 10,
    "advantage_normalization_type": "standard",
    "world_size": 1,
    "num_learners_per_node": [1],
    "verification_reward": 1.0,
    "apply_verifiable_reward": true,
    "apply_r1_style_format_reward": false,
    "r1_style_format_reward": 0.0,
    "additive_format_reward": false,
    "mask_truncated_completions": false,
    "allow_world_padding": true,
    "resume_training_step": 1,
    "checkpoint_state_dir": null,
    "push_to_hub": false,
    "try_launch_beaker_eval_jobs_on_weka": false,
    "save_traces": false
}
"""
    
    with open("test_config.json", "w") as f:
        f.write(test_config)
    
    # Run with our config
    cmd = [
        "python", "-m", "open_instruct.grpo_fast",
        "--config", "test_config.json"
    ]
    
    print("Running minimal test to check queue flow...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_queue_flow()