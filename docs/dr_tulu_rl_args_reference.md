# DR-TULU RL Training Arguments Reference

Script: `scripts/train/dr-tulu/rl_qwen35_4b_drtulu.sh`  
Trainer: `open_instruct/grpo_fast.py`  
Model: `Qwen/Qwen3.5-4B`  
Dataset: `rl-research/dr-tulu-rl-data`

---

## Overview

This script launches a DR-TULU reinforcement learning job training Qwen3.5-4B as a multi-tool
research agent. The agent has access to three tools (Google Search via Serper, web page reader via
Jina, Semantic Scholar search) and learns via GRPO with two reward signals:
1. **Verifiable reward** — rule-based exact-match or code-execution checks.
2. **Evolving rubric reward** — GPT-4.1 generates and refines natural-language rubrics per query
   as training progresses (the DR-TULU innovation). Rather than a fixed reward function, rubrics
   adapt to what the model has and hasn't mastered.

The infrastructure uses a 4+4 GPU split on one node: 4 learner GPUs (ZeRO-3 training) + 4 vLLM
engines (rollout generation), with an async pipeline that prefetches rollouts ahead of gradient
updates.

---

## Mason / Beaker Infrastructure Arguments

These configure the Beaker cluster job before the `--` separator. They are consumed by `mason.py`,
not by the training script.

| Argument | Value | Explanation |
|---|---|---|
| `--task_name` | `dr_tulu_qwen35_4b` | Beaker experiment/task name, used for UI display and grouping |
| `--description` | `${RUN_NAME}` | Human-readable label; includes timestamp for uniqueness |
| `--cluster` | `ai2/jupiter` | Which Beaker cluster to schedule on |
| `--workspace` | `ai2/general-tool-use` | Beaker workspace for organizing experiments |
| `--priority` | `urgent` (default) | Queue priority; `urgent` > `normal` > `low` |
| `--pure_docker_mode` | flag | Run entirely inside Docker, not on the host environment |
| `--image` | `${BEAKER_IMAGE}` | Docker image; defaults to `${BEAKER_USER}/open-instruct-integration-test` |
| `--preemptible` | flag | Job can be preempted by higher-priority work; reduces cost but risks interruption |
| `--num_nodes` | `1` | Single machine |
| `--gpus` | `8` | 8 GPUs on that node (H100s on Jupiter) |
| `--budget` | `ai2/oe-omai` | Billing account for resource accounting |
| `--no_auto_dataset_cache` | flag | Skip caching the dataset locally before launching; required because `import vllm` fails locally on macOS/non-GPU machines |

### Environment Variables (injected into container)

| `--env` | Value | Purpose |
|---|---|---|
| `RUBRIC_JUDGE_MODEL` | `gpt-4.1` | OpenAI model used by the evolving rubric system to *score* rubrics — evaluates whether a model response satisfies a given rubric criterion |
| `RUBRIC_GENERATION_MODEL` | `gpt-4.1` | OpenAI model used to *generate* new rubrics — proposes new natural-language criteria based on model failures |

### Secrets (injected from Beaker secret store)

| `--secret` | Secret name | Purpose |
|---|---|---|
| `SERPER_API_KEY` | `shashankg_SERPER_API_KEY` | Google Search via Serper API |
| `S2_API_KEY` | `shashankg_S2_API_KEY` | Semantic Scholar paper search |
| `JINA_API_KEY` | `shashankg_JINA_API_KEY` | Jina web page reader/reader API |
| `OPENAI_API_KEY` | `shashankg_OPENAI_API_KEY` | OpenAI calls for rubric generation/judging |

---

## Training Script Arguments (`open_instruct/grpo_fast.py`)

### Core Identity

| Argument | Value | Explanation |
|---|---|---|
| `--run_name` | `${RUN_NAME}` | Logged to W&B and checkpoints; timestamped for uniqueness |
| `--exp_name` | `dr_tulu_qwen35_4b` | Experiment name used for checkpoint directory naming |
| `--model_name_or_path` | `Qwen/Qwen3.5-4B` | Base model loaded from HuggingFace Hub |
| `--seed` | `1` | Global random seed for reproducibility |

### GRPO / RL Objective

| Argument | Value | Explanation |
|---|---|---|
| `--beta` | `0.001` | KL penalty coefficient. Scales the KL divergence penalty between the current policy and reference policy in the GRPO loss. Very small value (default is often ~0.04) — allows aggressive exploration with only light KL constraint |
| `--load_ref_policy` | `True` | Load a separate frozen copy of the base model as the KL reference policy. Without this, GRPO has no KL anchor. Required when `beta > 0`. |
| `--kl_estimator` | `3` | Which of 4 KL estimators to use (defined in `model_utils.estimate_kl`): `0`=linear (`log_prob_diff`), `1`=quadratic (`log_prob_diff²/2`), `2`=numerically stable (`expm1(-lpd) + lpd`, the typical default), `3`=importance-weighted (`ratio * log_prob_diff`). Estimator 3 is IS-weighted — it accounts for the distribution shift between old and current policy during a gradient step, making the KL estimate more accurate under off-policy updates |
| `--non_stop_penalty` | `False` | If `True`, penalizes responses that don't emit a stop token (useful for forcing concise answers). Set to `False` here because long-horizon tool-use chains legitimately don't always end cleanly |
| `--temperature` | `1.0` | Sampling temperature for rollout generation. 1.0 = no sharpening; preserves diversity in rollouts which is important for GRPO variance reduction |
| `--lr_scheduler_type` | `constant` | No learning rate decay — flat LR throughout. Common for short RL fine-tuning runs |
| `--learning_rate` | `5e-7` | Very small LR (~10x smaller than typical SFT). RL fine-tuning is more sensitive to large updates; too-high LR causes reward hacking or collapse |

### Batch / Rollout Sizing

| Argument | Value | Explanation |
|---|---|---|
| `--num_unique_prompts_rollout` | `2` | Number of distinct prompts sampled per rollout batch. The "group" size in GRPO is defined per-prompt; having 2 unique prompts means advantages are normalized within each group of 4 completions |
| `--num_samples_per_prompt_rollout` | `4` | Number of independent completions generated per prompt. GRPO computes advantages by comparing these 4 completions against each other (mean-centered, std-normalized). More samples = better advantage estimates but more generation cost |
| `--per_device_train_batch_size` | `1` | Number of packed sequences per GPU per gradient step. With ZeRO-3 and gradient checkpointing, 1 is typically the memory limit |
| `--num_mini_batches` | `1` | Number of gradient accumulation steps within one rollout batch (analogous to PPO epochs over the same data). `1` means no reuse of rollouts — each batch is trained on once |
| `--total_episodes` | `100` | Total training episodes before the job stops. With `num_unique_prompts=2` and `num_samples=4`, each episode = 8 rollout trajectories |
| `--max_steps` | `10` | Max tool-use steps (turns) per episode. The agent gets at most 10 tool calls before generation is cut off. Prevents infinite loops in agentic settings |

### Sequence Length / Packing

| Argument | Value | Explanation |
|---|---|---|
| `--max_prompt_token_length` | `2048` | Prompts longer than 2048 tokens are truncated. Keeps the context window budget mostly for the response |
| `--response_length` | `10240` | Maximum tokens the model can generate per trajectory. Very large (10K) to accommodate multi-step tool-use chains: think tags + tool calls + results + final answer |
| `--pack_length` | `18500` | Target length when packing multiple shorter sequences into a single GPU batch. With `prompt=2048 + response=10240 ≈ 12K` per trajectory, packing allows ~1.5 sequences per GPU slot. Must be ≤ model max context. The Qwen3.5 GatedDeltaNet architecture requires explicit sequence boundary tracking (cu_seq_lens) during packing |

### Async Pipeline / Throughput

| Argument | Value | Explanation |
|---|---|---|
| `--async_steps` | `4` | Depth of the async rollout prefetch queue. While the learner is training on step N, vLLM is already generating rollouts for steps N+1 through N+4. Queue size = `(async_steps + 1) * num_unique_prompts = 5 * 2 = 10` prompt slots. Larger = better GPU utilization but more memory for buffered rollouts |
| `--active_sampling` | flag | Enable dynamic prioritization of prompts — the dataloader re-samples "active" prompts (ones the model is still actively learning from, based on reward variance) rather than cycling through the dataset uniformly |
| `--inflight_updates` | flag | Allow weight updates (from learners → vLLM) to propagate while rollouts are still in flight. Trades a small amount of rollout staleness for significantly higher throughput. Without this, vLLM must fully quiesce before each weight sync |

### Dataset

| Argument | Value | Explanation |
|---|---|---|
| `--dataset_mixer_list` | `rl-research/dr-tulu-rl-data 0.05` | Training dataset with 5% sampling fraction per epoch. The low fraction + `total_episodes=100` controls total data exposure |
| `--dataset_mixer_list_splits` | `train` | Use the `train` split |
| `--dataset_mixer_eval_list` | `rl-research/dr-tulu-rl-data 8` | Evaluation dataset — 8 samples drawn from the same dataset for periodic local eval |
| `--dataset_mixer_eval_list_splits` | `train` | Eval from `train` split (no held-out eval set; eval is a small sample of training data to track reward trends) |
| `--ground_truths_key` | `ground_truth` | Column name in the HuggingFace dataset containing the ground truth answer/reference used for reward computation |
| `--sft_messages_key` | `messages` | Column name containing the prompt messages (in chat format) |

### Distributed Training

| Argument | Value | Explanation |
|---|---|---|
| `--deepspeed_stage` | `3` | ZeRO-3: model parameters, gradients, and optimizer states are all fully sharded across the 4 learner GPUs. Maximum memory efficiency — required to fit Qwen3.5-4B with ZeRO-3 optimizer states alongside vLLM on the same node |
| `--num_learners_per_node` | `4` | 4 of the 8 GPUs are "learner" GPUs responsible for gradient computation and weight updates. The other 4 are reserved for vLLM |
| `--vllm_num_engines` | `4` | 4 independent vLLM engines, each on its own GPU, for parallel rollout generation. Each engine handles its own prompt slice in a given rollout batch |
| `--gradient_checkpointing` | flag | Recompute activations during the backward pass rather than storing them. Trades compute for memory — necessary for fitting ZeRO-3 + long sequences |
| `--backend_timeout` | `1800` | NCCL/DeepSpeed distributed initialization and weight-sync timeout in seconds (30 minutes). Long because weight synchronization from ZeRO-3 sharded learners to vLLM can be slow under network variability |

### Reward / Verifiers

| Argument | Value | Explanation |
|---|---|---|
| `--apply_verifiable_reward` | `true` | Enable rule-based verifiable rewards (exact match, code execution checks). These are fast and don't require LLM calls |
| `--apply_evolving_rubric_reward` | `true` | **Core DR-TULU feature.** Instead of a fixed reward function, GPT-4.1 generates natural-language rubrics for each query. After each rollout, the rubric buffer is updated based on what the model succeeded and failed at. The rubrics evolve to probe harder criteria as easier ones are mastered |
| `--max_active_rubrics` | `5` | Cap the rubric buffer at 5 simultaneously active rubrics per query. When a 6th rubric would be added, the lowest-variance (least informative) rubric is evicted (FIFO). Prevents the rubric space from exploding as training progresses |
| `--remap_verifier` | `general_rubric=rubric` | Alias the `general_rubric` verifier name to `rubric` in the verifier registry. The dataset uses `general_rubric` as the verifier key, but the internal class is registered as `rubric`. This arg bridges that naming gap without requiring a dataset rewrite |

### Tool Use

| Argument | Value | Explanation |
|---|---|---|
| `--tool_parser_type` | `vllm_qwen3_xml` | Specifies how to parse tool calls from model output. Qwen3 uses XML-style tags (e.g., `<tool_call>...</tool_call>`) rather than JSON function-call format |
| `--tools` | `serper_search jina_browse s2_search` | Tool class names from `TOOL_REGISTRY`. Three tools: (1) `serper_search` — Google Search via Serper API; (2) `jina_browse` — fetches and extracts content from web pages; (3) `s2_search` — Semantic Scholar for academic paper search |
| `--tool_call_names` | `google_search browse_webpage snippet_search` | Function names the *model emits* in its output when calling each tool. Maps 1:1 with `--tools`. The model says `google_search(...)` in its output; the framework routes this to the `serper_search` pool |
| `--tool_configs` | `'{}' '{}' '{}'` | Per-tool configuration JSON dicts (one per tool). All empty here — using default configs for all three tools |
| `--pool_size` | `8` | Number of Ray actors per tool pool. Set to `num_unique_prompts * num_samples = 2 * 4 = 8` so every in-flight trajectory can make a tool call simultaneously without blocking |
| `--system_prompt_override_file` | `scripts/train/dr-tulu/dr_tulu_adjusted.txt` | Replaces the default system prompt with a research assistant persona that instructs the model to: use `<think>` tags, call tools iteratively, only give `<answer>` when ready, and cite all claims with snippet IDs |

### Checkpointing / Logging

| Argument | Value | Explanation |
|---|---|---|
| `--with_tracking` | flag | Enable W&B experiment tracking |
| `--save_traces` | flag | Save full rollout traces (prompt + all tool calls + tool results + final response) to disk. Critical for DR-TULU debugging — you can inspect exactly what the agent searched for and how it reasoned |
| `--local_eval_every` | `100` | Run local evaluation (on the 8-sample eval set) every 100 training steps |
| `--save_freq` | `100` | Save model weights checkpoint every 100 steps |
| `--checkpoint_state_freq` | `100` | Save full training state (optimizer, scheduler, step count) every 100 steps — allows resuming from a checkpoint |
| `--keep_last_n_checkpoints` | `-1` | Keep **all** checkpoints indefinitely. `-1` = no deletion. Reasonable for a short 100-episode run |
| `--vllm_enable_prefix_caching` | flag | Cache KV states for shared prompt prefixes in vLLM. Since all rollouts share the same system prompt (from `--system_prompt_override_file`), prefix caching avoids recomputing those attention states for every generation — significant speedup |
| `--push_to_hub` | `False` | Don't push model checkpoints to HuggingFace Hub during training |

---

## Key Design Decisions / Tradeoffs

**Why `beta=0.001` (tiny KL)?**  
Tool-use RL requires more policy movement than math RL — the model needs to learn entirely new
behavior patterns (when/how to call tools, how to synthesize search results). A larger KL penalty
would over-constrain this. The tradeoff is less stability.

**Why `kl_estimator=3` (IS-weighted)?**  
With `inflight_updates` on, the rollouts used for training were generated by a slightly stale
policy (the policy before the last weight sync). IS-weighting corrects for this distribution
shift, making the KL estimate more accurate under off-policy conditions.

**Why 4+4 GPU split (not 8 learners or 8 vLLM)?**  
Tool-use generation is slow (multiple sequential tool calls per trajectory). Dedicating 4 GPUs
to vLLM keeps rollout throughput high. The 4 learner GPUs are sufficient for ZeRO-3 training on
a 4B model. Skewing more toward vLLM would be appropriate if tool latency increases further.

**Why `pack_length=18500` with `response_length=10240`?**  
The Qwen3.5 architecture (GatedDeltaNet hybrid) requires explicit sequence boundary tracking
(`cu_seq_lens`) when packing. The pack length is tuned to fit roughly 1.5 trajectories per GPU
slot while staying within the model's context window.

**Why `active_sampling`?**  
With only 2 unique prompts per rollout and a very small dataset (5% sampling), uniform random
sampling can get stuck repeating prompts the model has already solved. Active sampling prioritizes
prompts with high reward variance — ones the model is uncertain about — for more informative
gradient signal.

---

## Related Files

- `open_instruct/grpo_fast.py` — main training loop
- `open_instruct/rubrics/evolving_rubric_step.py` — rubric buffer management and generation
- `open_instruct/rubrics/metrics.py` — `filter_rubric_buffer`, rubric capping logic
- `open_instruct/ground_truth_utils.py` — verifier registry, `remap_verifier` logic
- `open_instruct/grpo_utils.py` — GRPO loss, KL estimators
- `open_instruct/model_utils.py` — `estimate_kl()` function (4 estimators)
- `open_instruct/data_loader.py` — `StreamingConfig` dataclass (most training args live here)
- `scripts/train/dr-tulu/dr_tulu_adjusted.txt` — system prompt for the research agent
- `configs/beaker_configs/ray_node_setup.sh` — Ray cluster init before training starts
