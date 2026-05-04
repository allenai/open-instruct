# <Experiment short name>

## Status

- **State**: planning | running | completed | failed | preempted
- **Eval state**: not started | in progress | retrieved | analyzed
- **Last updated**: YYYY-MM-DD

## Purpose

One paragraph: what hypothesis or comparison this run is testing, and what
result would tell us.

## Beaker

- **URL**: https://beaker.org/ex/...
- **Workspace**: ai2/...
- **Cluster**: ai2/...
- **Resources**: N nodes × M GPUs
- **Launched**: YYYY-MM-DD HH:MM (UTC)
- **Started**: YYYY-MM-DD HH:MM (UTC)
- **Terminated**: YYYY-MM-DD HH:MM (UTC) (exit code N)

## Configuration

- **Launch script**: `cse-579-scripts/...sh`
- **Branch / commit**: `<branch>` @ `<sha>`
- **Base model**: `org/Model-Name`
- **Dataset**: `org/dataset-name`
- **Shaping**: method=..., decay=..., warmup=..., correctness_threshold=...
- **Other key hyperparams**: lr, episodes, response_length, num_samples, ...

## Outputs

- **Checkpoints**: /weka/.../checkpoint_dir
- **W&B run**: https://wandb.ai/...
- **Eval beaker jobs**: list links
- **Eval results path**: /weka/.../

## Pair / baseline

- **Compare to**: link to baseline experiment .md (or note that baseline still
  needs launching)

## Notes

Inline observations during/after the run. Anything surprising, anything that
would change how you set up the next run.

## Known issues

Bugs, instability, missing data, incomplete evals — anything that should
caveat the result before someone else uses it.
