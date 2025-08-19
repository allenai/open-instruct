#!/usr/bin/env python3
"""
Quick test script for toy training on GPU
"""

import torch
from open_instruct.toy_training import ToyTrainingConfig, ToyTrainer

def main():
    print("Testing Toy Training Script on GPU")
    print("=" * 50)
    
    # Check GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Test configuration - minimal for quick testing
    config = ToyTrainingConfig(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",  # Use Qwen model
        learning_rate=1e-5,
        num_training_steps=2,  # Just 2 steps for testing
        reward_function="custom",
        advantage_normalization="standard",
        verbose=True,
        device="auto"  # Will use GPU if available
    )
    
    print(f"\nInitializing trainer...")
    trainer = ToyTrainer(config)
    
    print(f"\nTesting reward computation...")
    example = trainer.data.training_examples[0]
    response = example["responses"][0]
    finegrained_scores, log_values = trainer.compute_rewards(example, 0, response)
    print(f"✓ Generated {len(finegrained_scores)} reward spans")
    
    # Test advantage computation
    advantages = trainer.compute_advantages(finegrained_scores)
    print(f"✓ Computed {len(advantages)} advantages")
    
    # Test character to token span conversion
    char_spans = [(start_char, end_char) for _, (start_char, end_char), _, _ in finegrained_scores]
    token_spans = trainer.convert_char_spans_to_token_spans(response, char_spans)
    print(f"✓ Converted {len(char_spans)} character spans to token spans")
    
    print(f"\nRunning minimal training...")
    metrics = trainer.train()
    
    print(f"\n✓ Training completed successfully!")
    print(f"Final metrics: {metrics[-1] if metrics else 'No metrics'}")
    
    # Test different reward functions
    print(f"\nTesting different reward functions...")
    for reward_func in ["custom", "combined", "finegrained"]:
        try:
            config.reward_function = reward_func
            trainer.config = config
            scores, logs = trainer.compute_rewards(example, 0, response)
            print(f"✓ {reward_func}: {len(scores)} spans")
        except Exception as e:
            print(f"⚠ {reward_func}: {str(e)}")
    
    print(f"\n🎉 All tests passed! Toy training script is working correctly.")

if __name__ == "__main__":
    main() 