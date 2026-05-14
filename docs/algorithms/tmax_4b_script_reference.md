# Plain-Language Argument Reference: `qwen35_4b_base_tmax_10k_8_podman_services.sh`

This document walks through every argument in the tmax training script and explains what it does
and why it was set to its current value. Written for someone who understands RL concepts but is
new to this codebase and its infrastructure.

**Script:** `scripts/tmax/4b/qwen35_4b_base_tmax_10k_8_podman_services.sh`  
**Related overview:** [grpo_pipeline_overview.md](grpo_pipeline_overview.md)  
**Deep technical reference:** [tmax_rl_args_reference.md](../tmax_rl_args_reference.md)

---

## How the script is structured

The script has two layers:

1. **`mason.py` arguments** (lines starting with `--cluster`, `--num_nodes`, `--env`, etc.) — these
   tell Beaker (AI2's compute cluster) *how to allocate machines and configure the container*.
   Mason is just a launcher; it doesn't touch the training code.

2. **`grpo_fast.py` arguments** (everything after the `--`) — these are the actual RL training
   hyperparameters that control how learning happens.

---

## Part 1: Mason / Beaker Arguments

These control the infrastructure: which machines to use, how many GPUs, what environment variables
to set before training starts.

### Machine allocation

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--cluster` | `ai2/jupiter` | Which Beaker cluster to run on. Jupiter is AI2's H100 GPU cluster. |
| `--image` | `$BEAKER_IMAGE` | Which Docker image (pre-built container) to run the job in. Passed in as the first argument when you launch the script. |
| `--num_nodes` | `2` | Reserve 2 separate machines for this job. |
| `--gpus` | `8` | Each machine has 8 GPUs → 16 GPUs total across both machines. |
| `--priority` | `urgent` | Queue priority so the job starts sooner. |
| `--preemptible` | *(flag)* | Allow the job to be interrupted if higher-priority jobs need the machines. |
| `--max_retries` | `0` | Don't automatically restart if the job crashes. Set to 0 because a crash mid-way through container execution can leave inconsistent state that would corrupt a retry. |
| `--pure_docker_mode` | *(flag)* | Run the job inside Docker (as opposed to directly on the host). Required for isolation and reproducibility. |
| `--mount_docker_socket` | *(flag)* | Give the training container access to the host machine's Docker daemon. Required so the training code can spin up *nested* sandbox containers for code execution. Without this, the `swerl_sandbox` tool can't run any code. |
| `--workspace` | `ai2/general-tool-use` | Beaker workspace used for experiment grouping and billing visibility. |
| `--budget` | `ai2/oe-omai` | Internal billing account for compute costs. |
| `--no_auto_dataset_cache` | *(flag)* | Skip caching the dataset locally before launching. Required because local machines don't have vLLM installed, and the caching step would fail trying to import it. |
| `--description` | *(string)* | Human-readable label shown in the Beaker UI. |

---

### Environment variables

These are injected into the container before training starts. Think of them as global settings
the training process can read.

#### Basic identity

| Variable | Value | Plain-language meaning |
|---|---|---|
| `REPO_PATH` | `/stage` | Path inside the container where the open-instruct repo is mounted. The training scripts reference this to find files like system prompts. |
| `GIT_COMMIT` | *(auto)* | Short git hash of the current commit, baked into the job for reproducibility — so you can always look up which exact code version produced a given run. |

#### vLLM engine flags

| Variable | Value | Plain-language meaning |
|---|---|---|
| `VLLM_USE_V1` | `1` | Use the newer vLLM v1 engine. Required for some features like prefix caching on the Qwen3.5 hybrid model architecture. |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | Allow vLLM to pass model weights between processes using Python's pickle format. This is how the trainer ships updated weights to the inference engines after each training step. "Insecure" refers to the fact that pickle can theoretically execute arbitrary code — fine here since we control all processes. |
| `VLLM_DISABLE_COMPILE_CACHE` | `1` | Don't cache PyTorch compilation artifacts between runs. Avoids bugs where a stale cache from a previous Docker image causes the model to behave incorrectly. |

#### Docker / container infrastructure

These configure the system of nested containers used for running code (the "sandbox"). See the
pipeline overview for context on why this is needed.

| Variable | Value | Plain-language meaning |
|---|---|---|
| `BEAKER_ALLOW_SUBCONTAINERS` | `1` | Tell Beaker this job is allowed to spawn containers *inside* the job's own container. Without this, Podman fails immediately with a permissions error. |
| `BEAKER_SKIP_DOCKER_SOCKET` | `1` | Tell Beaker not to inject its own Docker socket — we manage our own Podman socket instead. |
| `SWERL_PODMAN_SERVICE_COUNT` | `8` | Start 8 separate Podman daemons (container managers), each with its own storage directory and network socket. We need multiple because a single daemon becomes a bottleneck when hundreds of containers are starting and stopping at the same time. 8 daemons = ~8× the throughput. The 128 sandbox actors are spread evenly across these 8 daemons. |
| `SWERL_DOCKER_AUTO_REMOVE` | `1` | Automatically delete containers once they finish executing. Prevents disk from filling up during a long run where thousands of containers are created and destroyed. |
| `SWERL_DOCKER_START_CONCURRENCY` | `128` | At most 128 containers can be *starting* simultaneously (enforced via a semaphore). Prevents overwhelming the Podman daemon with too many simultaneous `docker start` calls. This is set to match `pool_size`. |
| `SWERL_SANDBOX_TIMING_LOGS` | `1` | Enable timing logs for container operations (how long each start/exec/cleanup takes). Useful for diagnosing performance bottlenecks. |
| `SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S` | `1.0` | Only log timing for operations slower than 1 second. Filters out the noise of fast operations while still catching slow ones. |
| `PODMAN_NUM_LOCKS` | `65536` | Raise Podman's internal file-lock limit. The default (2048) is too low when running hundreds of containers simultaneously — you'd hit "too many locks" errors. 65536 is generous headroom. |
| `CONTAINERS_STORAGE_CONF` | `/etc/containers/storage.conf` | Path to the Podman storage config file. Needed for the Docker mirror setup step. |
| `MIRROR_URL` | *(internal AI2 URL)* | A local Docker Hub mirror on the Jupiter cluster. When 16 machines all try to pull the same `python:3.12-slim` image from Docker Hub at once, Docker Hub rate-limits them. The local mirror serves the image from within the cluster network instead. |
| `DOCKERHUB_USERNAME` | `shashankg209` | Docker Hub account used *with* the `DOCKER_PAT` secret to authenticate pulls and avoid rate limits. Only used for pulling, not pushing. |

#### Secret

| Secret name | Variable it becomes | Plain-language meaning |
|---|---|---|
| `shashankg_DOCKER_PAT` | `DOCKER_PAT` | Personal Access Token for Shashank's Docker Hub account. Combined with `DOCKERHUB_USERNAME` to authenticate container image pulls. Stored securely in Beaker's secret store (not in the script itself). |

---

## Part 2: Training Arguments (`grpo_fast.py`)

These control the actual RL training. Grouped by what they affect.

### What to train and on what data

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--model_name_or_path` | `Qwen/Qwen3.5-4B` | The model to start from. This is the raw *base* model (not instruction-tuned), so it has no prior knowledge of tool use. Training starts from scratch on the target task. |
| `--dataset_mixer_list` | `hamishivi/swerl-tmax-10k 1.0` | The training dataset and what fraction of it to use (1.0 = all of it). `swerl-tmax-10k` is a collection of 10K software engineering problems, each with a task description and a test suite. |
| `--dataset_mixer_list_splits` | `train` | Use the `train` split of that dataset. |
| `--exp_name` | *(string)* | Name for this experiment. Used to name output directories and W&B runs. |
| `--seed` | `42` | Random seed for reproducibility. Controls weight initialization noise, data shuffling, etc. |

### How long to train

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--total_episodes` | `1280` | Total number of *training steps* (not individual samples). Each step processes a batch of 256 trajectories (32 prompts × 8 samples). So `1280 steps × 256 trajectories = ~327K trajectories` total seen during training. |
| `--num_epochs` | `1` | How many passes through the dataset per step. 1 means each batch of rollout data is used for exactly one gradient update, then discarded. Standard for on-policy RL. |

### How much data per training step (rollout batch size)

Think of a "rollout" as: pick some problems, generate attempts for each, score them, then train.

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--num_unique_prompts_rollout` | `32` | Pick 32 distinct problems per training step. |
| `--num_samples_per_prompt_rollout` | `8` | Generate 8 independent attempts for each problem. Total per step: 32 × 8 = **256 trajectories**. The 8 attempts per problem are what GRPO uses to compute "was this attempt better or worse than average?" |
| `--per_device_train_batch_size` | `1` | Process 1 packed sequence per GPU at a time during the gradient update. With sequence parallel (2 GPUs per sequence) and 4 DP groups, that's 4 sequences per gradient step across all learner GPUs. |

### How long sequences can be

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--max_prompt_token_length` | `2048` | The problem description can be at most 2048 tokens long. |
| `--response_length` | `32768` | The model's entire response (all bash commands and outputs combined) can be up to 32,768 tokens. This is long because coding agents accumulate a growing history — every bash command and its output gets appended to the context. |
| `--pack_length` | `35840` | The total "slot" size for one training example: prompt + response = 2048 + 32768 = 34816. Set slightly above that (35840) to leave a small buffer. |
| `--max_steps` | `100` | The agent can call bash at most 100 times per trajectory before being cut off. Controls the maximum length of an interaction. |

### The RL learning algorithm

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--beta` | `0.0` | **No KL penalty.** Normally RL training adds a term that penalizes the model for moving too far from its starting point (to prevent "forgetting" too much). Here it's set to 0 because the base model has no useful prior behavior for tool use — we *want* it to move far from its starting point. The only thing preventing instability is the clipping in the DAPO objective. |
| `--advantage_normalization_type` | `centered` | How to turn raw scores into advantages for each attempt. With `centered`, each attempt's advantage = its score minus the average score for that problem. So if all 8 attempts for a problem pass the tests (score=1.0), all advantages are 0 and there's no gradient signal (nothing to learn from). If `standard` were used instead, it would also divide by the standard deviation — but that blows up to infinity when all scores are identical (std=0), which is common with binary pass/fail rewards. |
| `--verification_reward` | `1.0` | Maximum reward for passing all tests. Passing = 1.0, partial credit for partial passes, failing = 0.0. (The default in the code is 10.0; using 1.0 here keeps the reward scale small and stable with `beta=0`.) |
| `--temperature` | `1.0` | Sampling temperature for generation. 1.0 means no sharpening — the model samples proportionally to its raw output probabilities. Lower values make it more deterministic (less exploration). |
| `--learning_rate` | `1e-6` | How large each gradient step is. Small to avoid destabilizing the model. Standard for RL fine-tuning; slightly larger than some other configs here because base models need more aggressive updates. |
| `--lr_scheduler_type` | `constant` | Keep the learning rate constant throughout training. No warmup or decay. |
| `--use_vllm_logprobs` | `true` | Use the log-probabilities recorded during generation (by vLLM) as the "old policy" probabilities in the importance-sampling ratio, rather than running a second forward pass. Saves compute. **Only valid when beta=0** — when there's no KL term, the two policies are always identical, so IS correction is trivially 1.0. |
| `--truncated_importance_sampling_ratio_cap` | `0.0` | Disabled (`0.0`). This would normally cap the importance sampling ratio to prevent large corrections, but since `use_vllm_logprobs=true` makes the ratio always exactly 1.0, capping it is pointless. |

### How GPUs are used

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--num_learners_per_node` | `8` | All 8 GPUs on the training node (node 0) are used as learner (training) processes. Total: 8 training GPUs. |
| `--vllm_num_engines` | `8` | 8 separate inference engines for generation, one per GPU on node 1. Each engine independently generates trajectories. More engines = more parallel generation = faster rollout collection. |
| `--vllm_tensor_parallel_size` | `1` | Each vLLM engine uses exactly 1 GPU (no splitting a single model across multiple GPUs). With an 8-engine setup on 8 GPUs, this is the natural choice. |
| `--deepspeed_stage` | `3` | Use DeepSpeed ZeRO Stage 3: the model weights, gradients, and optimizer states are all sharded (split) across all 8 training GPUs. Each GPU holds only 1/8 of everything, dramatically reducing per-GPU memory. Necessary to fit a 4B model with 32K sequences on H100s. |
| `--sequence_parallel_size` | `2` | Each individual training sequence is further split across 2 GPUs (Ulysses sequence parallelism). With 8 training GPUs and SP=2, you get 4 "data parallel groups" — 4 sequences processed in parallel per gradient step. Required because 35K-token sequences don't fit on a single GPU even with ZeRO-3. |
| `--gradient_checkpointing` | *(flag)* | Don't store intermediate activations during the forward pass — recompute them during the backward pass instead. Halves memory at the cost of ~30% extra compute. Standard practice for training on long sequences. |

### Pipeline speed

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--async_steps` | `8` | The inference engines (generation) run 8 steps *ahead* of the trainer (gradient update). While the trainer is updating on batch N, the inference engines are already generating batches N+1 through N+8. This keeps all GPUs busy continuously instead of alternating between idle training GPUs and idle inference GPUs. |
| `--inflight_updates` | `true` | After a training step finishes, immediately resume generation even before the weight broadcast to the inference engines completes. The inference engines generate one batch with slightly stale weights, but overall throughput is higher. |
| `--active_sampling` | *(flag)* | Prioritize problems the model is still actively learning (i.e., not already solving 100% of the time or failing 100% of the time). Skips "easy" problems where the model has already converged and "impossible" problems where it never succeeds — focusing compute on the productive middle ground. |

### The tool / sandbox

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--tools` | `swerl_sandbox` | The tool the model is trained to use. This is a bash terminal inside an isolated Docker container. The model can call `bash(command)` repeatedly. To submit a final answer, the model echoes a special string. |
| `--tool_configs` | *(JSON)* | Settings for the sandbox: which dataset to pull task files from, a 120-second timeout for test execution, and the default Docker image (`python:3.12-slim`) to run code in (individual tasks can override this). |
| `--pool_size` | `128` | Number of sandbox actors (container slots) to pre-create. Think of these as "worker slots" — each running trajectory needs one. With 256 trajectories being collected at once, 128 slots means each slot handles 2 trajectories sequentially. The queue is managed automatically. |
| `--tool_parser_type` | `vllm_qwen3_xml` | How to parse the model's output into tool calls. Qwen3 uses XML-style tags to denote tool calls; this parser reads those tags and routes them to the bash executor. |
| `--system_prompt_override_file` | *(path)* | Path to the system prompt file that instructs the model on its role (it's a terminal agent, it has a bash tool, how to submit). Overrides any default system prompt. |
| `--backend_timeout` | `1200` | Kill any trajectory that takes more than 20 minutes. Prevents a single stuck container from blocking a pool slot forever. |

### vLLM generation settings

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--vllm_enable_prefix_caching` | *(flag)* | Cache the computed attention state for the system prompt and task description, which are the same across all 8 attempts for a given problem. Avoids recomputing them 8 times. Can give a meaningful speedup on long prompts. |
| `--vllm_gdn_prefill_backend` | `triton` | Qwen3.5 uses a hybrid architecture with some "linear attention" layers (GatedDeltaNet). These have a custom prefill implementation; `triton` uses Triton GPU kernels which are faster than the default CUDA implementation on H100s. |

### Saving and logging

| Argument | Value | Plain-language meaning |
|---|---|---|
| `--with_tracking` | *(flag)* | Enable Weights & Biases (W&B) logging. Tracks metrics like reward, loss, KL divergence, and solve rate over time. |
| `--save_traces` | *(flag)* | Save the full trajectory of each rollout (every bash command and its output) to disk. Useful for debugging what the model is actually doing. |
| `--rollouts_save_path` | `/output/rollouts` | Where on the container filesystem to save those trajectory files. Mapped to Beaker's persistent `/output` mount. |
| `--output_dir` | `/output` | Where to save model checkpoints. Also mapped to Beaker's `/output` mount. |
| `--checkpoint_state_freq` | `10` | Save a full checkpoint (model weights + optimizer state, so training can resume from here) every 10 steps. More frequent than saving just the model weights because long runs need recovery points. |
| `--save_freq` | `20` | Save just the model weights (without optimizer state) every 20 steps. Lighter than a full checkpoint. |
| `--local_eval_every` | `10` | Run a local evaluation pass every 10 steps to track solve rate during training. |
| `--push_to_hub` | `false` | Don't push model checkpoints to HuggingFace Hub. |
| `--try_launch_beaker_eval_jobs_on_weka` | `False` | Don't automatically launch external evaluation jobs after each checkpoint. |

---

## At a glance: the numbers that define this run

| What | Value | Why |
|---|---|---|
| **Total GPUs** | 16 (2 nodes × 8) | Fits on 2 Jupiter machines |
| **Training GPUs** | 8 (node 0) | DeepSpeed ZeRO-3 across all 8 |
| **Inference GPUs** | 8 (node 1) | 8 vLLM engines, 1 GPU each |
| **Batch size** | 256 (32 problems × 8 attempts) | Enough diversity for stable GRPO signal |
| **Max sequence length** | ~35K tokens | Long bash histories need this |
| **Max tool calls per trajectory** | 100 | Coding tasks require many iterations |
| **Sandbox slots** | 128 | Pre-warmed container workers |
| **Podman daemons** | 8 | Spread container load across 8 managers |
| **Prefetch depth** | 8 steps | Keeps both nodes busy continuously |

---

## Common "why not...?" questions

**Why not use more samples per prompt (e.g. 16 instead of 8)?**  
Each sample requires running a full trajectory (up to 100 bash steps, up to 32K tokens). Memory
and container slots scale linearly with this. 8 is a practical balance — enough for a reliable
signal, not so many that you can't fit everything in memory and pool slots.

**Why not a larger pool (e.g. 1024 instead of 128)?**  
Each pool actor is a Ray process and corresponds to potential container load. With only 16 GPUs
and 256 trajectories per batch, 128 slots is enough to keep the pipeline moving with minimal
queueing. Larger pools would cost more Ray memory for minimal throughput gain at this scale.

**Why not larger `async_steps`?**  
8 means up to 8 × 256 = 2048 trajectories are "in flight" at once (some being generated, some
waiting in the queue). More prefetch = more memory for queued trajectories and more staleness
between the training model and the generating model. 8 is a practical ceiling for this setup.
