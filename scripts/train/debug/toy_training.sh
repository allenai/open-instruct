#!/bin/bash

# Toy Training Script - Demonstrates fine-grained token-level reward control
# Similar to grpo_fast and fgrpo_fast but with pre-defined rollouts and rewards

echo "Running Toy Training Script"
echo "=========================="

# Run the toy training script
python open_instruct/toy_training.py

echo ""
echo "Toy training completed!"
echo ""
echo "This script demonstrates:"
echo "- Pre-defined training rollouts (prompts and responses)"
echo "- Configurable fine-grained rewards at token level"
echo "- Control over which tokens receive which rewards"
echo "- Multiple reward groups (format, content, reasoning, completeness)"
echo "- Different advantage normalization strategies"
echo ""
echo "You can modify the ToyTrainingConfig in toy_training.py to:"
echo "- Change reward functions: 'custom', 'combined', 'finegrained'"
echo "- Adjust advantage normalization: 'standard', 'centered', 'none'"
echo "- Set different learning rates and training steps"
echo "- Use different models" 