# Training Scripts Guide

This guide covers the four main areas of training scripts used for GRPO/SFT experiments:

| Folder | Purpose | When to use |
|---|---|---|
| `debug/` | Integration tests and quick smoke tests for all training modes | Validating a code change works before a full run |
| `debug/envs/` | Debug scripts for specific RL environments | Testing a specific environment or new env feature |
| `qwen/` | Production GRPO/SFT runs on Qwen models | Launching real experiments |
| `../../tmax/` | Production runs specifically on tmax (tool-use agentic) tasks | Launching tmax-specific experiments |

All scripts run `open_instruct/grpo_fast.py` (or `grpo_fast.py` via mason for Beaker). Local scripts use `--single_gpu_mode` + `--vllm_sync_backend gloo`. Beaker scripts are wrapped with `mason.py` and launched via `build_image_and_launch.sh`.

---

## `debug/` — Integration tests and smoke tests

**Purpose:** Quickly verify that a code change doesn't break training. These run faster than production scripts — fewer steps, smaller models, smaller batch sizes.

**When to use:** Always run the appropriate debug script before launching a full production run. Use local scripts for fast iteration; use Beaker scripts when you need GPU memory closer to production.

### Core test scripts

| Script | What it tests | Scale | Duration |
|---|---|---|---|
| `single_gpu_on_beaker.sh` | Minimal GRPO smoke test | 1 GPU, Beaker | ~8 min |
| `grpo_fast.sh` | GRPO fast local run (math, GSM8K) | 1 GPU, local | fast |
| `grpo_fast_3_gpu.sh` | Same but multi-GPU | 3 GPU, local | fast |
| `large_test_script.sh` | Full multi-node GRPO integration test | 2×8 GPU, Beaker | ~32 min |
| `large_test_script_hybrid.sh` | Same but with OLMo-Hybrid model | 2×8 GPU, Beaker | ~32 min |

### DPO (`dpo/`)

| Script | Scale | Notes |
|---|---|---|
| `dpo/local.sh` | 1 GPU, local | No Beaker needed |
| `dpo/single_gpu.sh` | 1 GPU, Beaker | |
| `dpo/multi_node.sh` | 2×8 GPU, Beaker | |
| `dpo/multi_node_cache.sh` | 2×8 GPU, Beaker | With dataset caching |
| `dpo/multi_node_hybrid.sh` | 2×8 GPU, Beaker | OLMo-Hybrid model |

### Tool-use (`tools/`)

Scripts for debugging the tool-use training pipeline specifically:

| Script | What it tests |
|---|---|
| `tools/olmo_3_parser_multigpu.sh` | Multi-GPU GRPO with tool use (OLMo-3 parser) |
| `tools/qwen3_vllm_hermes_parser_debug.sh` | Qwen3 + vllm_hermes parser |
| `tools/dr_tulu_parser_debug.sh` | DrTulu parser |
| `tools/legacy_parser_debug.sh` | Legacy parser |
| `tools/mcp_weather_debug.sh` | MCP tool integration |
| `tools/tool_regression_beaker.sh` | Full tool regression test on Beaker |

### Other debug scripts

- `sft_integration_test.sh` / `finetune.sh` — SFT training tests
- `reward_modeling.sh` — reward model training
- `test_resume_weight_sync.sh` / `test_resume_weight_sync_zero3.sh` / `beaker_test_resume_weight_sync.sh` — checkpoint resume and vLLM weight sync tests
- `evolving_rubric_mini_test.sh` — evolving rubric reward
- `ppo.sh` — PPO training debug
- `judge.sh` — LLM-as-judge reward debug
- `qwen35_finetune_hybrid_test.sh` / `qwen35_4b_dapo_math.sh` — Qwen3.5-specific tests

---

## `debug/envs/` — RL environment-specific debug scripts

**Purpose:** Test GRPO training against a specific RL environment in isolation. Each script is self-contained and targets one environment.

**When to use:** When adding a new environment, debugging environment-specific behavior, or verifying that an environment integration still works after code changes. See [debug/envs/README.md](debug/envs/README.md) for the full per-script breakdown.

### Environment overview

| Environment | Scripts | Docker/Apptainer | Key trait |
|---|---|---|---|
| CounterEnv | `counter_1gpu.sh` | none | Simplest possible env, no tools |
| GuessNumberEnv | `guess_number_1gpu.sh`, `guess_number_beaker.sh` | none | Simple number guessing, no tools |
| SandboxLM | `sandbox_lm_1gpu.sh`, `sandbox_lm_8gpu.sh` | Docker | Math via code execution |
| SWERL Sandbox | `swerl_sandbox_*.sh` | Docker or Apptainer | Agentic tasks with bash/editor/submit tools |
| Wordle | `wordle_8gpu.sh`, `wordle_sandbox_lm_8gpu.sh` | optional | Text-based word game |

### Local vs Beaker

- **Local scripts** (`*_1gpu.sh`): run directly on your machine, no mason/Beaker. Use `--single_gpu_mode`. Fast to iterate.
- **Beaker scripts** (`*_8gpu.sh`, `*_4node.sh`, `*_beaker.sh`): wrapped with `mason.py`. Require a committed image via `build_image_and_launch.sh`.

### Scale progression for SWERL Sandbox

The swerl_sandbox scripts form a natural progression:

```
swerl_sandbox_1gpu.sh          — baseline, Qwen3-0.6B, Docker, minimal config
swerl_sandbox_1gpu_qwen35.sh   — same but Qwen3-4B, longer max_steps
swerl_sandbox_openthoughts_1gpu.sh — Apptainer, Qwen3.5-0.8B, production-faithful config
swerl_sandbox_8gpu.sh          — Beaker 8 GPU, Qwen3-4B, inflight updates, active sampling
swerl_sandbox_4node.sh         — Beaker 32 GPU, base instruct model, production scale
swerl_sandbox_4node_sft.sh     — same but SFT warm-start from tmax checkpoint
```

**Use `openthoughts_1gpu.sh` as your 1-GPU production mirror** — it mirrors the 4-node config most faithfully (active_sampling, inflight_updates, centered advantage normalization, Apptainer backend).

---

## `qwen/` — Production Qwen training scripts

**Purpose:** Full production training runs for Qwen models on real datasets. Launched to Beaker via `mason.py`.

**When to use:** Running actual experiments, not debugging. Always run the appropriate `debug/` script first to verify the code works.

### GRPO scripts

| Script | Model | Scale | Task |
|---|---|---|---|
| `grpo_fast_3b_single_node.sh` | Qwen2.5-3B | 1 node / 8 GPU | Math (GSM8K-style) |
| `grpo_fast_7b.sh` | Qwen2.5-7B | 2 nodes / 16 GPU | Math |
| `grpo_fast_7b_code.sh` | Qwen2.5-7B | 2 nodes / 16 GPU | Code |
| `grpo_fast_7b_orz.sh` | Qwen2.5-7B | 2 nodes / 16 GPU | Math (ORZ dataset) |
| `grpo_fast_32b.sh` | Qwen2.5-32B | multi-node | Math |
| `qwen3_4b_dapo_math_32k.sh` | Qwen3-4B | multi-node | Math (DAPO, 32k ctx) |

### SFT scripts

| Script | Model | Scale | Dataset |
|---|---|---|---|
| `finetune_7b.sh` | Qwen2.5-7B | single node | general SFT |
| `sft_qwen35_9b_tmax_sft.sh` | Qwen3.5-9B | 4 nodes / 32 GPU | hamishivi/tmax-sft-full |
| `sft_qwen35_9b_tmax_sft_w_incomplete.sh` | Qwen3.5-9B | 4 nodes / 32 GPU | tmax + incomplete trajectories |
| `sft_qwen3_4b_tmax.sh` | Qwen3-4B | 4 nodes / 32 GPU | hamishivi/tmax-sft |

`math_system_prompt.txt` is a shared system prompt used by math GRPO runs.

---

## `../../tmax/` — Tmax agentic task production scripts

**Purpose:** Production GRPO training on tmax tasks — long-horizon agentic tasks using the SWERL sandbox (Docker containers per sample). These are the main scripts for the tmax research line.

**When to use:** Launching tmax experiments. The `debug/envs/swerl_sandbox_4node*.sh` scripts are the debug counterparts for these.

### Main training scripts

| Script | Model | Scale | Notes |
|---|---|---|---|
| `qwen3_4b_sft_tmax_10k.sh` | hamishivi/sft_qwen3_4b_tmax (SFT init) | 4 nodes / 32 GPU | GRPO on tmax-10k |
| `qwen35_9b_sft_tmax_10k.sh` | hamishivi/qwen3.5-sftv1-9b (SFT init) | 4 nodes / 32 GPU | GRPO on tmax-10k |

### Ablation scripts (`qwen_35_base_500step/`)

500-step ablation runs starting from the Qwen3.5-9B base (not SFT), used to test training config variants:

| Script | What it ablates |
|---|---|
| `qwen35_9b_base_tmax_10k.sh` | Baseline |
| `qwen35_9b_base_tmax_10k_beta_0p1.sh` | Higher KL penalty (beta=0.1) |
| `qwen35_9b_base_tmax_10k_mask_no_submit.sh` | Mask turns with no submit call |
| `qwen35_9b_base_tmax_10k_mask_no_submit_10pct.sh` | Same but 10% masking |
| `qwen35_9b_base_tmax_10k_mask_overlong.sh` | Mask overlong sequences |
| `qwen35_9b_base_tmax_10k_concave_length_penalty.sh` | Concave length penalty |
| `qwen35_9b_base_tmax_10k_turncap35.sh` | Cap at 35 turns |
| `qwen35_9b_base_tmax_10k_verified.sh` | Verified rewards only |
| `qwen35_9b_base_tmax_10k_verified_mask_no_submit_10pct.sh` | Verified + 10% no-submit masking |
| `qwen36_27b_base_tmax_10k_tp2.sh` | Scale-up: Qwen3.6-27B with TP=2 |

Agent-task dataset variants (for testing different training data sources):

| Script | Dataset |
|---|---|
| `qwen35_9b_base_agent_task_endless_terminals.sh` | agent-task with endless terminals |
| `qwen35_9b_base_agent_task_openthoughts.sh` | agent-task-openthoughts |
| `qwen35_9b_base_agent_task_r2e_gym.sh` | R2E-Gym tasks |
| `qwen35_9b_base_agent_task_termigen.sh` | Termigen tasks |
| `qwen35_9b_base_agent_task_terminal_traj.sh` | Terminal trajectory tasks |

### Diagnostic tools (`debug/`)

- `count_truncations.py` / `count_truncations.sh` — count how many sequences are being truncated in a run
- `recreate_blowup_mask.sh` / `recreate_blowup_mask_64k.sh` — reproduce gradient/loss blowup conditions for debugging

---

## Decision guide: which script to use?

```
Want to test a code change?
  → Use debug/ scripts

  Quick local check (no Beaker needed)?
    → debug/grpo_fast.sh  (GRPO)
    → debug/dpo/local.sh  (DPO)

  Full integration test on Beaker?
    → debug/single_gpu_on_beaker.sh    (~8 min, cheap)
    → debug/large_test_script.sh       (~32 min, multi-node)
    → debug/tools/tool_regression_beaker.sh  (tool use)

Testing a specific RL environment?
  → Use debug/envs/ scripts
  → See debug/envs/README.md for the full breakdown

Running a real experiment?
  On a Qwen model for math/code?
    → qwen/ scripts

  On tmax agentic tasks?
    → tmax/ scripts (4-node runs)
    → tmax/qwen_35_base_500step/ for ablations

  Want to run a smaller version of a tmax production run first?
    → debug/envs/swerl_sandbox_4node.sh   (production scale)
    → debug/envs/swerl_sandbox_8gpu.sh    (cheaper, 1 node)
    → debug/envs/swerl_sandbox_openthoughts_1gpu.sh  (1 GPU, production-faithful config)
```

---

## Key config axes across all scripts

| Axis | Values and meaning |
|---|---|
| **Container backend** | `none` (toy envs), `docker` (swerl/sandbox), `apptainer` (openthoughts on Slurm) |
| **Tool parser** | `vllm_hermes` (most scripts), `vllm_qwen3_xml` / `vllm_qwen3xml` (Qwen3 XML-format tools) |
| **Beta** | `0.0` = no KL penalty; `0.01` = light KL regularization toward reference model |
| **Active sampling** | Production and multi-GPU runs use it; simple debug runs don't. Prioritizes harder samples. |
| **Inflight updates** | Enabled at 8+ GPU scale. Updates weights while rollouts are still running. |
| **Advantage normalization** | `default` (normalize per batch) vs `centered` (subtract mean, used in production) |
| **Model warm-start** | Base instruct checkpoint vs SFT checkpoint (tmax SFT init improves stability) |
| **DeepSpeed stage** | Stage 2 (smaller runs) vs Stage 3 + sequence parallelism (large multi-node runs) |
