# Toy Training Script for Fine-Grained Token-Level Reward Control

This directory contains a toy training script (`open_instruct/toy_training.py`) that demonstrates how to control token-level rewards and backpropagation, similar to `grpo_fast` and `fgrpo_fast`, but with pre-defined rollouts and rewards for educational and debugging purposes.

## Features

- **Pre-defined training rollouts**: Fixed prompts and responses for reproducible experiments
- **Fine-grained token-level rewards**: Control which tokens receive which rewards
- **Multiple reward groups**: Support for different types of rewards (format, content, reasoning, completeness)
- **Configurable advantage normalization**: Standard, centered, or no normalization
- **Multiple reward functions**: Custom, combined rubric+citation, or simple finegrained rewards
- **Detailed logging**: See exactly which tokens get which rewards and advantages
- **Offline demo**: Works without model downloads for educational purposes

## Quick Start

```bash
# Run the offline demo (works without internet/model downloads)
python demo_toy_training.py

# Run the toy training script (requires model download)
bash scripts/train/debug/toy_training.sh

# Or run directly
python open_instruct/toy_training.py
```

## Files

- `open_instruct/toy_training.py` - Main toy training script with real models
- `demo_toy_training.py` - Offline demo with mock components (no model downloads needed)
- `scripts/train/debug/toy_training.sh` - Shell script to run toy training
- `TOY_TRAINING_README.md` - This documentation

## GPU Testing Results

✅ **Successfully tested on H200 GPU node** with the following configuration:
- 8x NVIDIA H200 GPUs
- PyTorch 2.6.0+cu124
- CUDA available and working
- All core concepts demonstrated successfully

The demo shows different advantage normalization strategies:
- **Standard normalization**: `(score - mean) / std` - advantages centered around 0
- **Centered normalization**: `score - mean` - removes mean but keeps scale
- **None normalization**: Raw scores - shows actual reward values as advantages

## How It Works

### 1. Pre-defined Training Data

The script includes three example scenarios:
- **Geography question**: "What is the capital of France?"
- **Math problem**: "Solve: 2 + 3 * 4"
- **Science explanation**: "Explain photosynthesis briefly"

Each example has multiple response variants with different quality levels.

### 2. Fine-Grained Reward Structure

Rewards are defined as tuples: `(score, (start_char, end_char), reward_group_id, response_idx)`

**Reward Groups:**
- `0`: Format/Structure - How well the response is formatted
- `1`: Content Accuracy - Factual correctness of the content
- `2`: Reasoning/Explanation - Quality of reasoning or explanation
- `3`: Completeness - How complete the response is

### 3. Token-Level Control

The script:
1. Converts character spans to token spans
2. Computes advantages per reward group (normalized separately)
3. Applies policy gradient loss only to specified token ranges
4. Uses different rewards for different parts of the same response

## Configuration Options

Modify the `ToyTrainingConfig` in `toy_training.py`:

```python
config = ToyTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",  # Model to use
    learning_rate=1e-5,                               # Learning rate
    num_training_steps=10,                            # Training steps
    reward_function="custom",                         # Reward function type
    advantage_normalization="standard",               # Normalization strategy
    verbose=True,                                     # Detailed logging
)
```

### Reward Functions

1. **`"custom"`**: Uses pre-defined rewards with precise control over token spans
2. **`"combined"`**: Uses the combined rubric + citation reward function
3. **`"finegrained"`**: Uses the simple finegrained reward function

### Advantage Normalization

1. **`"standard"`**: `(score - mean) / std` per reward group
2. **`"centered"`**: `score - mean` per reward group  
3. **`"none"`**: Raw scores without normalization

## Example Output

```
🎯 DEMONSTRATION: STANDARD Normalization
============================================================
EXAMPLE 0: What is the capital of France?
============================================================

Response 0: The capital of France is Paris. It's a beautiful city known for the Eiffel Tower.
----------------------------------------
Reward spans (3):
  1. Format: score=0.900, adv=0.000
      Chars [0:30]: 'The capital of France is Paris'
      Tokens [0:6]
  2. Accuracy: score=0.950, adv=0.000
      Chars [4:30]: 'capital of France is Paris'
      Tokens [1:6]
  3. Reasoning: score=0.700, adv=0.000
      Chars [31:81]: ' It's a beautiful city known for the Eiffel Tower.'
      Tokens [7:17]
Response loss: 0.0000
```

## Understanding the Training Process

### 1. Reward Computation
For each response, the script computes fine-grained scores for different text spans, assigning them to different reward groups.

### 2. Advantage Calculation
Advantages are computed per reward group:
- All spans in the same reward group are normalized together
- This allows different types of rewards to have different scales
- Higher scores within a group get positive advantages, lower scores get negative advantages

### 3. Policy Gradient Loss
For each token in a reward span:
```python
token_loss = -advantage * log_prob(actual_token)
```
- Positive advantages increase probability of generating those tokens
- Negative advantages decrease probability of generating those tokens

### 4. Backpropagation
Only tokens within the specified reward spans receive gradients, allowing precise control over which parts of the response are reinforced or penalized.

## Customizing Rewards

To add your own reward logic, modify the `get_custom_finegrained_scores` method in the `ToyTrainingData` class:

```python
def get_custom_finegrained_scores(self, example_idx: int, response_idx: int, response_text: str):
    # Your custom reward logic here
    return [
        (score, (start_char, end_char), reward_group_id, response_idx),
        # ... more reward spans
    ]
```

## Comparison with grpo_fast and fgrpo_fast

| Feature | grpo_fast | fgrpo_fast | toy_training | demo_toy_training |
|---------|-----------|------------|--------------|-------------------|
| Rollout generation | vLLM inference | vLLM inference | Pre-defined | Pre-defined |
| Reward computation | Real verifiers | Real verifiers | Configurable | Mock/Custom |
| Training data | Dynamic datasets | Dynamic datasets | Fixed examples | Fixed examples |
| Model requirements | Real models | Real models | Real models | Mock models |
| Network requirements | Yes | Yes | Yes | No |
| Complexity | Production-ready | Production-ready | Educational | Demo |
| Control level | High-level config | Fine-grained spans | Full token control | Full token control |

## Use Cases

1. **Debugging reward functions**: Test reward logic without expensive inference
2. **Understanding fine-grained training**: See exactly how token-level rewards work
3. **Prototyping new reward strategies**: Quickly test different reward structures
4. **Educational purposes**: Learn how RLHF with fine-grained rewards works
5. **Ablation studies**: Compare different normalization and reward strategies
6. **Offline development**: Work on reward logic without internet connectivity

## Dependencies

The script uses the same dependencies as the main training scripts:
- `torch`
- `transformers` (for real models)
- `numpy`
- Optional: `open_instruct.search_rewards` modules for advanced reward functions

The demo version (`demo_toy_training.py`) only requires:
- `torch`
- `numpy`

## Extending the Script

You can extend the script by:
1. Adding more training examples in `ToyTrainingData`
2. Implementing new reward functions
3. Adding different advantage computation strategies
4. Integrating with real datasets
5. Adding evaluation metrics
6. Creating more sophisticated mock models for the demo

## Testing

The scripts have been tested on:
- ✅ CPU environments (login nodes)
- ✅ GPU environments (H200 nodes with 8 GPUs)
- ✅ Both with and without internet connectivity
- ✅ Different PyTorch versions (2.6.0+cu124, 2.8.0+cu128)

This toy script provides a foundation for understanding and experimenting with fine-grained token-level reward control in language model training. 