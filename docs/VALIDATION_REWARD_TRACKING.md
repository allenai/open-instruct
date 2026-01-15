# Validation Reward Tracking

This document describes the validation reward tracking feature for GRPO training.

## Overview

When training with GRPO (or similar RL methods), it's important to monitor performance on held-out data to detect overfitting. The `validation_holdout_ratio` parameter enables this by splitting your training data into:

- **Training set**: Used for policy gradient updates
- **Validation set**: Used for tracking rewards during training (appears as `eval/` metrics)

## Usage

Add the `--validation_holdout_ratio` parameter to your training script:

```bash
uv run python open_instruct/grpo_fast.py \
    --model_name_or_path your_model \
    --dataset_mixer_list your_dataset 1000 \
    --validation_holdout_ratio 0.1 \  # Hold out 10% for validation
    ... other args
```

### Parameters

- `validation_holdout_ratio`: Float between 0.0 and 0.5 (exclusive)
  - `0.0` (default): No validation holdout
  - `0.1`: 10% of training data held out for validation
  - `0.2`: 20% of training data held out for validation

### What happens

1. **Data splitting**: Training data is shuffled (using the seed) and split into train + validation
2. **Evaluation**: Validation samples are evaluated every `local_eval_every` steps
3. **Metrics**: Validation metrics appear under the `eval/` prefix in wandb/logs

### Example output

```
🎯 Validation holdout: split 300 samples into 270 train + 30 validation (ratio=0.1)
🎯 Using validation holdout (30 samples) for evaluation metrics. This tracks accuracy on held-out training data to detect overfitting.
```

## Relationship with eval_dataset

The validation holdout is separate from the standard `dataset_mixer_eval_list` (test set evaluation):

- **Validation (holdout)**: Same distribution as training data, used to track overfitting
- **Eval (test set)**: Different data split (typically test set), used to track generalization

When `validation_holdout_ratio > 0`:
- The validation holdout is used for `eval/` metrics
- If `dataset_mixer_eval_list` is also specified, a warning is logged and validation takes precedence

## Monitoring for Overfitting

To detect overfitting, compare training and validation metrics:

| Scenario | Train Reward | Eval Reward | Interpretation |
|----------|--------------|-------------|----------------|
| Learning | ↑ | ↑ | Model is improving on both |
| Overfitting | ↑ | → or ↓ | Model memorizing training data |
| Underfitting | → | → | Model not learning |

## Limitations

- **Eval timeout**: During training, evaluation uses a short timeout (0.01s) to avoid blocking. This means eval metrics may not appear on every `local_eval_every` step. This is a known limitation of the current eval system.
  - **Workaround**: Increase `local_eval_every` to reduce eval frequency, or use a smaller validation set
  - **Future work**: A dedicated validation path with configurable timeout could address this
- **Dataset size**: With small datasets, validation holdout further reduces training data. Consider using at least 100+ samples in total.
- **Single eval source**: When validation holdout is enabled, it replaces the standard eval_dataset. You cannot track both simultaneously in the current implementation.

## Example Script (DGX Spark)

See `scripts/train/dgx-spark/grpo_smollm_gsm8k.sh` for a complete example using validation reward tracking with SmolLM2-135M on GSM8K.
