# Eval Wait Time Metric Implementation

## Overview

This implementation adds a new metric to track the time between eval jobs in the `grpo_fast.py` script. The metric measures how much time the eval job is waiting between runs, which helps identify potential bottlenecks in the evaluation pipeline.

## Implementation Details

### 1. Global Variable for Tracking

Added a global variable to track the last eval finish time:

```python
# Global variable to track the last eval finish time
# This is used to measure the time between when one eval job finishes and the next one is queued
last_eval_finish_time = None
```

### 2. Recording Eval Finish Time

Modified the `maybe_evaluate` function to record when an eval job finishes:

```python
# Record the time when eval job finishes
# This timestamp is used to calculate the wait time for the next eval job
global last_eval_finish_time
last_eval_finish_time = time.time()
logger.info("[Main Thread] 📊 Evaluation job finished")
```

### 3. Measuring Wait Time

Modified the `vllm_generate_thread` function to measure and log the wait time when the next eval job is queued:

```python
# Record the time when eval job is queued and measure wait time since last eval finished
global last_eval_finish_time
current_time = time.time()
if last_eval_finish_time is not None:
    eval_wait_time = current_time - last_eval_finish_time
    # Log the eval wait time to wandb if tracking is enabled
    # This metric measures how much time the eval job was waiting between runs
    if args.with_tracking and hasattr(wandb, 'run') and wandb.run is not None:
        wandb.log({"eval/wait_time_between_evals": eval_wait_time}, step=training_step)
        logger.info(f"[vLLM Thread] 📊 Eval wait time: {eval_wait_time:.2f}s")
```

### 4. Initialization

Modified the `main` function to initialize the global variable:

```python
# Initialize global eval tracking variable
global last_eval_finish_time
last_eval_finish_time = None
```

## Metric Details

- **Metric Name**: `eval/wait_time_between_evals`
- **Type**: Scalar (float, seconds)
- **Logged To**: Wandb (when `with_tracking=True`)
- **Step**: Training step when eval is queued
- **Description**: Time in seconds between when one eval job finishes and the next one is queued

## Usage

The metric will be automatically logged to Wandb when:
1. `args.with_tracking` is `True`
2. Wandb is properly initialized (`wandb.run` exists)
3. There is a previous eval finish time to calculate from (not the first eval)

## Benefits

1. **Identify Bottlenecks**: Helps identify if eval jobs are waiting too long between runs
2. **Optimize Pipeline**: Provides data to optimize the evaluation pipeline timing
3. **Monitor Performance**: Tracks evaluation efficiency over time
4. **Debug Issues**: Helps debug evaluation-related performance issues

## Testing

The implementation includes:
- Logic tests to verify the timing calculations work correctly
- Proper error handling for edge cases (first eval job)
- Appropriate logging and metric naming conventions
- Integration with existing Wandb tracking infrastructure

## Files Modified

- `open_instruct/grpo_fast.py`: Main implementation
- `tests/test_eval_wait_time.py`: Unit tests (created)

## Example Output

When the metric is logged, you'll see output like:
```
[vLLM Thread] 📊 Eval wait time: 1.23s
```

And in Wandb, you'll see the metric `eval/wait_time_between_evals` plotted over training steps.