# Holmes Acceptance Testing Handoff

## Current State

- Branch: `jacobm/holmes-testing`
- Current commit before this handoff doc: `7b8cf5314`
- Remote branch pushed: `origin/jacobm/holmes-testing`
- Target cluster: `ai2/holmes`
- Target workspace: `ai2/holmes-testing`
- Priority: `urgent`

The PDF that started this work was moved out of the repo so the launch wrapper's clean-tree check can pass:

```bash
/private/tmp/holmes-acceptance-testing-post-training.pdf
```

## Workspace Secrets

The Holmes workspace was checked and exists. It is unarchived and allows urgent priority.

Secrets currently present in `ai2/holmes-testing`:

- `jacobm_HF_TOKEN`
- `jacobm_WANDB_API_KEY`
- `jacobm_BEAKER_TOKEN`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

`mason.py` lists secrets from the target workspace and auto-injects user-prefixed secrets based on the Beaker username. For user `jacobm`, these map to:

- `HF_TOKEN` from `jacobm_HF_TOKEN`
- `WANDB_API_KEY` from `jacobm_WANDB_API_KEY`
- `BEAKER_TOKEN` from `jacobm_BEAKER_TOKEN`

The benchmark scripts also explicitly pass:

```bash
--secret HF_TOKEN=jacobm_HF_TOKEN
```

## Added Scripts

Four Holmes-specific launch script copies were added:

- `scripts/benchmarking/launch_benchmark_single_gpu_holmes.sh`
- `scripts/benchmarking/launch_benchmark_single_node_holmes_tp4.sh`
- `scripts/benchmarking/launch_benchmark_single_node_holmes_tp8.sh`
- `scripts/train/olmo3/32b_think_rl_holmes.sh`

All four scripts:

- target `--cluster ai2/holmes`
- target `--workspace ai2/holmes-testing`
- use `--priority urgent`
- include `--no_auto_dataset_cache` before the `--` separator
- passed `bash -n`

The single-node scripts differ as follows:

- TP4 script: `--vllm_num_engines 2`, `--vllm_tensor_parallel_size 4`
- TP8 script: `--vllm_num_engines 1`, `--vllm_tensor_parallel_size 8`

## Why Launch From The Cluster

Launching from the laptop failed because `docker` is not installed locally:

```text
./scripts/train/build_image_and_launch.sh: line 37: docker: command not found
./scripts/train/build_image_and_launch.sh: line 47: docker: command not found
```

The launch wrapper expects to build a Docker image and upload/create a Beaker image, so use an interactive environment with Docker/buildx and Beaker credentials.

## Recommended First Launch

From an interactive cluster session:

```bash
git fetch origin
git switch jacobm/holmes-testing
./scripts/train/build_image_and_launch.sh scripts/benchmarking/launch_benchmark_single_gpu_holmes.sh 8192 allenai/Olmo-3-32B-Think
```

This is the PDF's single-GPU inference benchmark path for the 8k generation-length baseline.

If that works, launch single-node TP4:

```bash
./scripts/train/build_image_and_launch.sh scripts/benchmarking/launch_benchmark_single_node_holmes_tp4.sh 8192 allenai/Olmo-3-32B-Think
```

Then single-node TP8 for the 32k generation-length baseline:

```bash
./scripts/train/build_image_and_launch.sh scripts/benchmarking/launch_benchmark_single_node_holmes_tp8.sh 32768 allenai/Olmo-3-32B-Think
```

Once inference is healthy and interconnect issues are resolved, the main training acceptance test is:

```bash
./scripts/train/build_image_and_launch.sh scripts/train/olmo3/32b_think_rl_holmes.sh
```

## PDF Baselines

Inference H100 baselines:

| Model | TP | Generation length | TPS |
| --- | ---: | ---: | ---: |
| Olmo 3 32B | 1 | 8k | 300.50 |
| Olmo 3 32B | 4 | 8k | 1302.62 |
| Olmo 3 32B | 8 | 32k | 928.45 |

Training H100 baseline for the main acceptance run:

| Model | Learner TPS | Actor TPS |
| --- | ---: | ---: |
| Olmo 3 32B Thinker | ~19k | ~3.5k |

The PDF expects roughly:

- inference: `~2.4x` faster on GB200 vs H100
- training: `~2.5x` faster on GB200 vs H100

## CUDA And Torch Caveat

Current repo image/dependency stack:

- `Dockerfile` base image: `nvidia/cuda:12.8.1-devel-ubuntu22.04`
- Linux x86_64 Torch from lockfile: `torch==2.10.0+cu128`
- Linux aarch64 Torch from lockfile: `torch==2.10.0+cu130`
- x86_64 FlashAttention wheels are pinned to CUDA 12.8 / Torch 2.10 variants

Holmes machines reportedly use CUDA 13.0 and Torch 2.12. This branch does not yet update the image or lockfile to that stack. Treat the single-GPU benchmark as a compatibility probe. If it fails in CUDA/Torch/vLLM/FlashAttention wheel setup, the next step is a dedicated Holmes image/dependency update rather than changing workload launch arguments.

## Notes

- The branch includes only launch-script copies plus this handoff doc.
- No PR has been opened yet, so `CHANGELOG.md` has not been updated.
- The launch wrapper requires a clean git tree before launching.
