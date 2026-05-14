# GRPO Training Pipeline: Plain-Language Overview

A high-level guide to the tmax/swerl-style RL training pipeline for someone familiar with RL concepts but new to this codebase's infrastructure.

---

## What we're training and why

We take a base language model (e.g. Qwen3.5-4B) and teach it to be a **coding agent** — solving programming tasks by writing and running bash commands in a terminal. The reward signal is fully automated: did the code pass the test suite? No human labeling needed.

The algorithm is **GRPO** (Group Relative Policy Optimization): for each problem, generate multiple attempts, compare them against each other, and train the model to make successful attempts more likely relative to failed ones.

---

## The core RL loop

The pipeline repeats this cycle:

1. **Generate** — give the model a coding problem, let it attempt a solution by calling bash tools iteratively
2. **Score** — run the test suite inside an isolated container, get pass/fail
3. **Update** — push model weights toward making successful solutions more likely

---

## Hardware split: two jobs, two sets of GPUs

With 2 nodes (16 GPUs total), the GPUs are divided into two crews with different jobs:

**Node 0 — The Trainer (8 GPUs)**
Holds the model weights and runs gradient updates. Uses DeepSpeed ZeRO-3 to shard the model across all 8 GPUs. This is the "slow" expensive part.

**Node 1 — The Generator (8 GPUs)**
Runs 8 independent copies of the model just for text generation (vLLM engines, one per GPU). These are optimized purely for fast inference — no gradients needed. After each training step, the trainer broadcasts updated weights to these engines so they stay in sync.

These two jobs run **asynchronously**: while the trainer is updating weights on batch N, the generators are already collecting batch N+8. This keeps all GPUs busy continuously.

---

## The sandbox: how code execution works

Each coding attempt runs inside an **isolated container** (a disposable Linux box). The agent loop is:

```
model generates bash command
  → runs in container
  → output appended to context
  → model generates next command
  → repeat up to max_steps times
  → test suite runs → pass/fail reward
```

**The pool** is a fixed number of these containers pre-warmed and waiting. When a generation request arrives, it grabs a free container, uses it, and returns it. If all containers are busy, new requests wait in a queue.

**Podman services** are independent container-management daemons. Multiple daemons are used because a single daemon bottlenecks when hundreds of containers start/stop simultaneously — sharding the pool across N daemons gives N× the throughput.

---

## Key parameters explained

### Rollout size — "how much data per training step"

| Parameter | Typical value | Meaning |
|---|---|---|
| `num_unique_prompts_rollout` | 32 | Distinct coding problems per batch |
| `num_samples_per_prompt_rollout` | 8 | Attempts per problem |
| → **total per step** | 256 | Trajectories collected before one weight update |

GRPO needs multiple attempts per problem to compute a relative signal. With 8 attempts, even 6 failures + 2 successes gives a usable gradient.

### Sequence length — "how long can solutions be"

| Parameter | Typical value | Meaning |
|---|---|---|
| `max_prompt_token_length` | 2048 | Input problem description |
| `response_length` | 32768 | Up to 32K tokens of bash history |
| `max_steps` | 100 | Max bash tool calls per trajectory |

Long because coding tasks accumulate history — each tool call's output is appended to context before the next command is generated.

### Training dynamics — "how the update works"

| Parameter | Typical value | Meaning |
|---|---|---|
| `beta` | 0.0 | No KL penalty. The model is free to move as far from its starting point as needed — important for a base model learning a new behavior (tool use) from scratch. |
| `advantage_normalization_type` | centered | Each attempt's advantage = its score − mean score across all 8 attempts for that problem. No std division because binary pass/fail rewards can give std=0, which causes NaN with standard normalization. |
| `learning_rate` | 1e-6 | Small, as typical for RL fine-tuning |
| `total_episodes` | 1280 | Total distinct problems seen during training |

### Pipeline efficiency — "keeping GPUs busy"

| Parameter | Typical value | Meaning |
|---|---|---|
| `async_steps` | 8 | Generators stay 8 batches ahead of the trainer. Neither side idles waiting for the other. |
| `inflight_updates` | true | Resume generation immediately after a training step finishes, even before the weight broadcast completes. Slightly stale weights for one short window, but faster overall. |
| `sequence_parallel_size` | 2 | Each long sequence (35K tokens) is split across 2 GPUs to reduce per-GPU memory. |
| `vllm_num_engines` | 8 | Parallel generators. Each handles rollout_batch / num_engines sequences per step. |

### Pool and sandbox infrastructure

| Parameter | Typical value | Meaning |
|---|---|---|
| `pool_size` | 128 | Containers available for execution. Can be smaller than total rollout size (256) since requests queue. |
| `SWERL_PODMAN_SERVICE_COUNT` | 4 | Container daemons for throughput. Pool actors spread evenly across daemons. |
| `PODMAN_NUM_LOCKS` | 65536 | Raises Podman's internal file-lock limit (default 2048 is too low for hundreds of concurrent containers). |
| `backend_timeout` | 1200s | Kill a trajectory after 20 min to prevent hung containers blocking the pool. |
| `SWERL_DOCKER_START_CONCURRENCY` | 128 | Max simultaneous `docker start` calls. Should match or exceed `pool_size`. |

---

## Full pipeline in one picture

```
Training problems (32 each step)
         │
         ▼
  vLLM generators (8 GPUs, Node 1)
  ┌──────────────────────────────┐
  │  Generate bash commands      │◄─── receives updated weights
  │  → run in sandbox containers │     after each training step
  │  → collect pass/fail reward  │
  └──────────────────────────────┘
         │ 256 scored trajectories
         ▼
  Compute advantages
  (compare 8 attempts per problem,
   subtract group mean)
         │
         ▼
  Trainer (8 GPUs, Node 0)
  ┌──────────────────────────────┐
  │  Gradient update             │──── broadcasts new weights
  │  (DeepSpeed ZeRO-3,          │     to generators
  │   sharded across 8 GPUs)     │
  └──────────────────────────────┘
```

The pipeline always runs both sides in parallel — generators collect the *next* batch while the trainer updates on the *current* batch.

---

## How parameters affect each other (common tuning decisions)

**Increasing `num_samples_per_prompt_rollout`** gives richer relative signal per problem but multiplies compute proportionally.

**Increasing `async_steps`** increases GPU utilization but also increases memory for queued trajectories and means the model trains on slightly older rollouts.

**Increasing `pool_size`** reduces queueing latency for tool calls but costs more Ray memory and more concurrent containers (watch `PODMAN_NUM_LOCKS`).

**Increasing `SWERL_PODMAN_SERVICE_COUNT`** increases container throughput proportionally, but each daemon needs its own storage root — check disk space.

**`sequence_parallel_size`** trades communication overhead for memory. SP=2 gives larger effective batch size; SP=4 gives more memory headroom per GPU.

---

## Related docs

- [tmax_4b_script_reference.md](tmax_4b_script_reference.md) — plain-language walkthrough of every argument in the tmax 4B script
- [grpo_fast_internals.md](grpo_fast_internals.md) — deep dive into Ray actor architecture, weight sync sequence, loss computation
- [rl_with_environments.md](rl_with_environments.md) — how to add new environments and tool integrations
- [tmax_rl_args_reference.md](../tmax_rl_args_reference.md) — technical argument reference for the tmax training scripts
- [dr_tulu_rl_args_reference.md](../dr_tulu_rl_args_reference.md) — argument reference for the DR-TULU web research scripts
