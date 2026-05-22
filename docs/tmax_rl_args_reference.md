# SWERL tmax RL Training Arguments Reference

Script: `scripts/tmax/4b/qwen35_4b_base_tmax_10k_8_podman_services.sh`  
Trainer: `open_instruct/grpo_fast.py`  
Model: `Qwen/Qwen3.5-4B`  
Dataset: `hamishivi/swerl-tmax-10k`

Related sibling: `scripts/tmax/4b/qwen35_4b_base_tmax_10k.sh`  
Only difference between the two: `--sequence_parallel_size 2` (8_podman_services) vs `--sequence_parallel_size 4` (base)

---

## Overview

This script trains Qwen3.5-4B as a **coding/terminal agent** that solves SWE-style tasks using a
bash shell inside an isolated Docker container. Each task in `swerl-tmax-10k` is a software
engineering problem with:
- `instruction.md` — the task description
- `tests/test.sh` — a test suite that grades the solution
- seed files and a Docker image to execute in

The agent calls a single `bash` tool and submits by echoing
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Reward is binary: tests pass or fail.

This script is much larger-scale than the DR-TULU script:
- **4 nodes × 8 GPUs = 32 GPUs** (vs 1 node / 8 GPUs for DR-TULU)
- **beta = 0.0** — pure reward maximization with no KL penalty
- **32768 token responses** — much longer than DR-TULU's 10240 (bash history accumulates)
- **Pool size 1024** — 1024 Docker container slots vs 8 tool actors
- **Extensive Podman infrastructure** — 8 sharded Podman daemons for parallel container execution

---

## Mason / Beaker Infrastructure Arguments

| Argument | Value | Explanation |
|---|---|---|
| `--cluster` | `ai2/jupiter` | Beaker cluster |
| `--workspace` | `ai2/dr-tulu-ablations` | Workspace for experiment grouping (DR-TULU ablations workspace, even for standalone tmax) |
| `--priority` | `urgent` | Queue priority |
| `--preemptible` | flag | Job can be preempted |
| `--num_nodes` | `4` | 4 machines |
| `--gpus` | `8` | 8 GPUs per node = 32 total |
| `--budget` | `ai2/oe-adapt` | Billing account (oe-adapt, not oe-omai as in DR-TULU) |
| `--max_retries` | `0` | Do not retry if the job fails. Coding evals can leave dirty container state, making retries unsafe |
| `--pure_docker_mode` | flag | Run inside Docker |
| `--mount_docker_socket` | flag | Mount the host's Docker socket into the container. Required so Podman can create sandbox containers for code execution. **Critical for swerl_sandbox** |
| `--no_auto_dataset_cache` | flag | Skip local dataset caching |

### Environment Variables

#### Container/Sandbox Identity

| `--env` | Value | Explanation |
|---|---|---|
| `REPO_PATH` | `/stage` | Path where the open-instruct repo is mounted inside the Docker container |
| `GIT_COMMIT` | `$(git rev-parse --short HEAD)` | Short commit SHA baked into the Beaker job for reproducibility tracking |
| `DOCKERHUB_USERNAME` | `hamishi740` | Docker Hub username for pulling sandbox images (hamishi = Hamish Ivison) |

#### vLLM Flags

| `--env` | Value | Explanation |
|---|---|---|
| `VLLM_USE_V1` | `1` | Use vLLM v1 engine architecture (newer, more performant — required for some features like prefix caching with hybrid models) |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | Allow vLLM to serialize/deserialize model weights via pickle. Required for IPC weight transfer between learner and vLLM processes |
| `VLLM_DISABLE_COMPILE_CACHE` | `1` | Disable torch.compile() cache. Avoids cache invalidation bugs when the training image changes between runs |

#### Podman / Docker Sandbox Infrastructure

These env vars configure the sharded Podman daemon layer. Processed by `scripts/docker/docker_login.sh` which runs before training starts.

| `--env` | Value | Explanation |
|---|---|---|
| `BEAKER_ALLOW_SUBCONTAINERS` | `1` | Tell Beaker to grant this task permission to spawn sub-containers (nested container support). Without this, `podman` fails with "cannot clone: Operation not permitted" |
| `BEAKER_SKIP_DOCKER_SOCKET` | `1` | Skip Beaker's own Docker socket injection — we manage our own Podman socket instead |
| `SWERL_PODMAN_SERVICE_COUNT` | `8` | Number of sharded Podman daemon instances to start. Each shard has its own socket, storage root, and run root. The `EnvironmentPool` distributes Docker SDK calls round-robin across all 8 sockets (via `SWERL_PODMAN_DOCKER_HOSTS`), parallelizing container lifecycle operations and reducing lock contention. 8 shards = 8× throughput for `docker start/stop/rm` calls |
| `SWERL_DOCKER_AUTO_REMOVE` | `1` | Automatically `docker rm` containers when they finish. Keeps disk usage bounded during long runs with 1024-pool tool actors |
| `SWERL_DOCKER_START_CONCURRENCY` | `128` | Maximum number of concurrent `docker start` calls (enforced via a file-slot semaphore). Prevents overwhelming the Podman daemon when many actors try to spin up containers simultaneously |
| `SWERL_SANDBOX_TIMING_LOGS` | `1` | Enable per-operation timing logs for sandbox operations (container start, exec, cleanup) |
| `SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S` | `1.0` | Only log timing for sandbox ops slower than 1 second. Filters noise while catching slowdowns |
| `MIRROR_URL` | `jupiter-cs-aus-150.reviz.ai2.in:5000` | Internal Docker Hub mirror on the Jupiter cluster network. Prevents Docker Hub rate-limiting when 32 nodes simultaneously pull the same `python:3.12-slim` sandbox image. Set up by `setup_dockerio_mirror` in the Docker image |
| `PODMAN_NUM_LOCKS` | `65536` | Number of file-based locks Podman allocates for container management. Default (8192) is too low when running 1024 concurrent containers — increases to 65536 to avoid "too many locks" errors |
| `CONTAINERS_STORAGE_CONF` | `/etc/containers/storage.conf` | Path to Podman container storage config. Needed so `setup_dockerio_mirror` knows where to write the overlay storage driver config |

### Secrets

| `--secret` | Secret name | Purpose |
|---|---|---|
| `DOCKER_PAT` | `hamishivi_DOCKER_PAT` | Docker Hub Personal Access Token for authenticating pulls. Written to `~/.docker/config.json` by `docker_login.sh` so all Ray workers pick it up |

---

## Pre-training Startup Sequence

Unlike DR-TULU, this script runs two setup steps before `grpo_fast.py`:

```
source scripts/docker/docker_login.sh && source configs/beaker_configs/ray_node_setup.sh && python open_instruct/grpo_fast.py
```

1. **`docker_login.sh`** — authenticates with Docker Hub, starts `SWERL_PODMAN_SERVICE_COUNT` (8) sharded Podman daemons, sets `SWERL_PODMAN_DOCKER_HOSTS` for the pool actors, sets up the `MIRROR_URL` registry mirror
2. **`ray_node_setup.sh`** — initializes the Ray cluster across all 4 nodes

---

## Training Script Arguments

### Core Identity

| Argument | Value | Explanation |
|---|---|---|
| `--exp_name` | `swerl_qwen35_4b_base_tmax_10k_verified_grpo_8_podman_services` | Experiment name for checkpointing |
| `--model_name_or_path` | `Qwen/Qwen3.5-4B` | Base model (not an instruction-tuned variant) |
| `--seed` | `42` | Global random seed |

### GRPO / RL Objective

| Argument | Value | Explanation |
|---|---|---|
| `--beta` | `0.0` | **No KL penalty.** Pure reward maximization — the policy is only constrained by the DAPO clipping, not by divergence from a reference policy. Appropriate for base-model RL where the starting point is far from the target behavior and strong exploration is needed |
| `--use_vllm_logprobs` | `true` | Use the logprobs returned by vLLM during generation as `old_logprobs` in the importance-sampling ratio, rather than running a separate forward pass with the learner. Saves memory and compute since there's no second pass over generated sequences. **Mutually exclusive with `truncated_importance_sampling_ratio_cap > 0`** (can't IS-correct if IS is the source) |
| `--truncated_importance_sampling_ratio_cap` | `0.0` | Disabled (`0.0`). When `use_vllm_logprobs=true`, the old and rollout logprobs are identical (π_old = π_vLLM), so the importance sampling ratio is always 1.0 — no correction is needed or meaningful |
| `--temperature` | `1.0` | No sampling sharpening |
| `--lr_scheduler_type` | `constant` | Flat LR |
| `--learning_rate` | `1e-6` | 2× higher than DR-TULU's 5e-7. Base model RL can tolerate more aggressive updates since the model starts farther from the target |

### Rollout / Batch Sizing

| Argument | Value | Explanation |
|---|---|---|
| `--num_unique_prompts_rollout` | `32` | 32 distinct prompts per rollout batch — 16× more than DR-TULU's 2. With `pool_size=1024` and 32 prompts × 8 samples, up to 256 Docker containers can run in parallel |
| `--num_samples_per_prompt_rollout` | `8` | 8 completions per prompt. More than DR-TULU's 4 → better GRPO advantage estimation, but higher memory cost |
| `--total_episodes` | `128000` | Long run: 128K episodes at 32 unique prompts × 8 samples = 256 rollouts per episode → ~32M total trajectories |
| `--num_epochs` | `1` | Train for 1 full pass through the dataset. With `total_episodes=128000` and a 10K-task dataset, this means roughly 12.8 passes through the data (episodes ÷ prompts-per-rollout = 128000/32 ≈ 4000 passes over the dataset if sampling uniformly, but with `active_sampling`, distribution is non-uniform) |
| `--max_steps` | `100` | Up to 100 bash tool calls per trajectory. Much higher than DR-TULU's 10 — coding tasks require many bash commands to iterate on a solution |
| `--per_device_train_batch_size` | `1` | 1 packed sequence per GPU per gradient step |
| `--async_steps` | `8` | 8 rollout batches prefetched ahead of training. Queue size = (8+1) × 32 = 288 prompt slots |
| `--verification_reward` | `1.0` | Maximum reward for passing all tests. Default is 10.0; reduced to 1.0 here to keep reward scale manageable. Partial credit is possible (fraction of tests passed) |
| `--advantage_normalization_type` | `centered` | Advantages = `scores - group_mean` (no std division). **`standard`** would be `(scores - mean) / std`, but std-normalization blows up when all 8 samples in a group get the same score (std → 0). `centered` is numerically safer for binary rewards (pass/fail) where many groups may have all-pass or all-fail |

### Sequence Length / Packing

| Argument | Value | Explanation |
|---|---|---|
| `--max_prompt_token_length` | `2048` | Max tokens in prompt |
| `--response_length` | `32768` | 32K tokens — ~3× longer than DR-TULU's 10240. Coding trajectories accumulate bash command history (commands + outputs) which grows very long over 100 steps |
| `--pack_length` | `35840` | Slightly larger than `response_length + max_prompt_token_length = 34816`. Packs ~1 trajectory per GPU slot at this length |

### Distributed Training

| Argument | Value | Explanation |
|---|---|---|
| `--deepspeed_stage` | `3` | Full ZeRO-3 parameter sharding |
| `--sequence_parallel_size` | `2` | **Ulysses sequence parallelism** — each pair of learner GPUs collaborates on a single sequence, splitting attention heads across them. With SP=2 on 8 learner GPUs per node: 4 data-parallel (DP) groups per node, 16 DP groups across 4 nodes. Required to fit 32K sequences in memory (sequence dim is halved per GPU). The sibling script `qwen35_4b_base_tmax_10k.sh` uses SP=4 instead |
| `--num_learners_per_node` | `8` | All 8 GPUs on each learner node are training GPUs (unlike DR-TULU which split 4/4). 8 × 4 nodes = 32 learner GPU processes total |
| `--vllm_num_engines` | `24` | 24 vLLM engines, each using 1 GPU (`vllm_tensor_parallel_size=1`). These are distributed across the 4 nodes. The GPU allocation between learners and vLLM is managed by Ray — vLLM engines run on GPUs shared with (but temporally separate from) learner processes |
| `--vllm_tensor_parallel_size` | `1` | Each vLLM engine uses 1 GPU. No tensor parallelism within a single engine — 24 independent single-GPU engines rather than fewer multi-GPU ones |
| `--gradient_checkpointing` | flag | Recompute activations to save memory |
| `--backend_timeout` | `1200` | 20-minute NCCL/DeepSpeed timeout (vs 30 min for DR-TULU; shorter because weight syncs are simpler with no rubric generation overhead) |

### Tool Use / Sandbox

| Argument | Value | Explanation |
|---|---|---|
| `--tools` | `swerl_sandbox` | The single tool: a bash shell inside a Docker container. Defined in `open_instruct/environments/swerl_sandbox.py`. The agent gets a `bash` function it can call repeatedly; submits by echoing `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` |
| `--tool_configs` | `{"task_data_hf_repo": "hamishivi/swerl-tmax-10k", "test_timeout": 120, "image": "python:3.12-slim"}` | Config for `SWERLSandboxEnv`: dataset repo to load task files from, 120s test timeout, default Docker image (overridable per-task via `image.txt`) |
| `--pool_size` | `1024` | **1024 Ray actors** for the swerl_sandbox tool pool — one actor per potential concurrent trajectory. With 32 unique prompts × 8 samples = 256 in-flight rollouts, having 1024 slots means 4× headroom for retries and slow containers without blocking the pool |
| `--tool_parser_type` | `vllm_qwen3_xml` | Parses Qwen3's XML-style tool call format |
| `--system_prompt_override_file` | `scripts/train/debug/envs/swerl_sandbox_system_prompt.txt` | System prompt describing the bash-only terminal agent interface |

### Checkpointing / Logging

| Argument | Value | Explanation |
|---|---|---|
| `--with_tracking` | flag | W&B tracking |
| `--save_traces` | flag | Save full rollout traces to `--rollouts_save_path` |
| `--rollouts_save_path` | `/output/rollouts` | Container path (mapped to Beaker /output mount) for rollout trace files |
| `--output_dir` | `/output` | Model checkpoint output directory (container path) |
| `--vllm_enable_prefix_caching` | flag | Cache KV states for shared prompt prefixes |
| `--vllm_gdn_prefill_backend` | `triton` | Use Triton kernels for prefilling in vLLM's GatedDeltaNet (Qwen3.5's hybrid attention) layers. The alternative is the default (CUDA) backend. Triton is faster for the GDN linear attention prefill on H100s |
| `--active_sampling` | flag | Prioritize prompts the model is still learning |
| `--inflight_updates` | `true` | Allow weight updates during rollout generation |
| `--checkpoint_state_freq` | `10` | Save full optimizer state every 10 steps (more frequent than DR-TULU's 100 — long runs need more recovery checkpoints) |
| `--local_eval_every` | `10` | Evaluate every 10 steps |
| `--save_freq` | `20` | Save model checkpoint every 20 steps |
| `--push_to_hub` | `false` | Don't push to HuggingFace Hub |
| `--try_launch_beaker_eval_jobs_on_weka` | `False` | Don't auto-launch eval jobs on Weka after checkpoints |

---

## Key Design Decisions / Tradeoffs

**Why `beta=0.0` (no KL penalty)?**  
This trains from the base model (`Qwen3.5-4B`, not an instruct variant). The base model has no
prior tendency to use bash tools or format code — the policy needs to move dramatically to learn
the task. Any KL penalty would slow this down. The DAPO clipping bounds still prevent single-step
gradient explosions.

**Why `use_vllm_logprobs=true` + `truncated_importance_sampling_ratio_cap=0.0`?**  
When the old policy is the same as the vLLM rollout policy (no weight updates happened between
generation and training), the IS ratio is exactly 1.0 — applying a cap or doing IS correction is
mathematically pointless. Using vLLM logprobs directly saves the cost of a second forward pass
through the learner for every generated token. Combined with `inflight_updates`, this is the
most throughput-efficient configuration.

**Why `advantage_normalization_type=centered` instead of `standard`?**  
Binary rewards (tests pass or fail) produce frequent all-pass or all-fail groups among the 8
samples for a prompt. `standard` normalization divides by group std, which → 0 in these cases,
creating NaN or exploding gradients. `centered` (subtract mean only) is numerically safe and still
centers the advantages around zero.

**Why `sequence_parallel_size=2` (not 4 as in the sibling script)?**  
SP=2 gives 16 DP groups across 4 nodes (4 per node). SP=4 gives 8 DP groups (2 per node).
More DP groups = larger effective batch size per step → more stable gradients. But SP=4 uses
less GPU memory per process (sequence split across 4 GPUs). The `8_podman_services` variant
chooses SP=2 to get a bigger effective batch; the base variant uses SP=4 for more memory headroom.

**Why `pool_size=1024` (vs 8 in DR-TULU)?**  
Each swerl_sandbox actor manages a Docker container lifecycle. With 32 prompts × 8 samples × up
to 100 bash steps, hundreds of containers may be active simultaneously. 1024 actors provides
4× headroom over the 256 in-flight trajectories, absorbing slow containers without stalling the
pipeline. DR-TULU's tools (HTTP APIs) are stateless and cheap; Docker containers are stateful
and slow to start — requiring a much larger pool.

**Why 8 sharded Podman services?**  
A single Podman daemon becomes a bottleneck with 1024+ concurrent Docker SDK calls (start, exec,
rm). Sharding across 8 independent Podman daemons (each with its own socket, storage, and lock
set) distributes this load. The `EnvironmentPool` assigns actors to shards round-robin, so
256 concurrent containers → ~32 per shard.

**Why `PODMAN_NUM_LOCKS=65536`?**  
Podman uses file locks for container state management. The default (8192) causes "too many locks"
errors when 200+ containers run simultaneously. 65536 locks provides 8× headroom.

**tmax vs standard SWE-bench:**  
`swerl-tmax-10k` ("terminal max") is a dataset of 10K programming tasks designed for RL training.
Unlike SWE-bench (which patches real GitHub repos), tmax tasks are self-contained with explicit
`tests/test.sh` scripts and Docker images. The reward is fully automated (no LLM judge needed).

---

## Relationship Between Scripts in `scripts/tmax/4b/`

| Script | Sequence Parallel | Notes |
|---|---|---|
| `qwen35_4b_base_tmax_10k.sh` | `4` | More memory efficient, smaller effective batch |
| `qwen35_4b_base_tmax_10k_8_podman_services.sh` | `2` | Larger effective batch, needs more GPU memory |
| `qwen35_4b_base_tmax_10k_last_step_warning.sh` | ? | Enables `last_step_warning` in swerl_sandbox (tells model it's on its last tool call) |

---

## Related Files

- `open_instruct/grpo_fast.py` — main training loop
- `open_instruct/environments/swerl_sandbox.py` — `SWERLSandboxEnv`, bash tool, Docker container management
- `open_instruct/environments/backends.py` — `DockerSandboxBackend` with semaphore-gated container ops; reads `SWERL_DOCKER_START_CONCURRENCY`, `SWERL_DOCKER_AUTO_REMOVE`
- `open_instruct/environments/pool.py` — `EnvironmentPool` Ray actor; reads `SWERL_PODMAN_DOCKER_HOSTS` for shard routing
- `open_instruct/grpo_utils.py` — `GRPOConfig` (truncated IS, use_vllm_logprobs, sequence_parallel_size); `compute_tis_weights()`
- `open_instruct/data_loader.py` — `StreamingConfig` (advantage_normalization_type, verification_reward, rollouts_save_path, vllm_gdn_prefill_backend)
- `open_instruct/vllm_utils.py` — vLLM engine initialization with `gdn_prefill_backend`
- `open_instruct/qwen3_5_packing_patch.py` — monkey-patch for GatedDeltaNet packing (sequence boundary tracking)
- `scripts/docker/docker_login.sh` — starts 8 Podman service shards, authenticates Docker Hub, sets `SWERL_PODMAN_DOCKER_HOSTS`
- `docker/podman/setup_dockerio_mirror` — configures Docker Hub mirror and Podman storage driver
