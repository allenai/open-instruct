# Script Folder Guide

This note summarizes several script folders that are easy to confuse when navigating experiment launchers.

## `scripts/tmax`

Focused Tmax/SWERL experiment suite. These scripts mostly launch GRPO runs with the `swerl_sandbox` tool on datasets such as `hamishivi/swerl-tmax-10k`, `hamishivi/swerl-tmax-10k-verified`, and related agent-task datasets.

Common patterns:

- Qwen3, Qwen3.5, and Qwen3.6 model variants.
- Large Beaker jobs, often 4 nodes x 8 GPUs.
- Sandbox tool execution through Docker or Podman.
- Ablations for `beta`, truncation masking, non-submitting completion masking, turn caps, length penalties, verified data, and SFT-started models.
- Debug utilities for logprobs and truncation analysis in `scripts/tmax/debug` and the `diagnose_*.py` files.

In short: this is the specialized Tmax/SWERL experiment area, closer to real experiment configs than generic smoke tests.

## `scripts/train/debug/envs`

Environment/tool-loop RL debugging. These scripts test models interacting with environments, usually through GRPO plus tools.

Common patterns:

- Toy RL environments such as `counter`, `guess_number`, and `wordle`.
- Generic sandbox/code-execution environments such as `sandbox_lm_*`.
- SWERL sandbox runs such as `swerl_sandbox_*`.
- Local 1-GPU scripts, 8-GPU Beaker scripts, and larger 4-node SWERL runs.
- System prompts for sandbox/tool behavior.
- Apptainer/Docker backend smoke tests.

In short: this is the sandbox/environment proving ground. Some configs here are smaller or earlier versions of experiments that later appear under `scripts/tmax`.

## `scripts/train/debug`

Broad integration-test and smoke-test folder for training code.

Common patterns:

- GRPO and `grpo_fast` local, single-GPU, and multi-node tests.
- DPO integration, checkpoint, cache, and resume tests.
- SFT tests, reward modeling tests, and PPO tests.
- vLLM weight-sync/resume tests.
- Large code-RL tests.
- LLM-judge/rubric experiments.
- Tool-use debug scripts under `scripts/train/debug/tools`.
- DPO-specific debug scripts under `scripts/train/debug/dpo`.

In short: this is the catch-all "does this training path work?" area. `debug/envs` is the environment/tool-loop slice of it.

## `scripts/train/qwen`

Qwen-family training recipe folder. These are model-specific experiment launchers rather than generic debugging scripts.

Common patterns:

- Qwen2.5 SFT and GRPO recipes for 3B, 7B, and 32B models.
- Math RLVR / DAPO-style math training.
- Code RLVR training.
- Qwen3 and Qwen3.5 Tmax SFT scripts.
- Qwen-specific system prompt files for math.

In short: this is where Qwen model experiments live.

## Quick Mental Model

- `scripts/tmax`: specialized Tmax/SWERL experiment suite.
- `scripts/train/debug`: broad smoke/integration/debug launchers.
- `scripts/train/debug/envs`: environment/tool-loop RL debug launchers.
- `scripts/train/qwen`: Qwen-specific model training recipes.
