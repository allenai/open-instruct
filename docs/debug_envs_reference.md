# scripts/train/debug/envs/ — Script Reference

All scripts run `open_instruct/grpo_fast.py`. Local scripts use `--single_gpu_mode` + `--vllm_sync_backend gloo`. Beaker scripts are wrapped with `mason.py` and launched via `build_image_and_launch.sh`.

---

## Toy environments (no Docker, no tools)

Simple built-in Python envs. All use Qwen3-0.6B, no external dependencies.

### `counter_1gpu.sh`
- **Env:** CounterEnv — increment a counter to reach a target, then submit
- **Execution:** local, 1 GPU
- Dataset: `hamishivi/rlenv-counter-nothink`
- episodes: 80, max_steps: 20, response_length: 1024, beta: 0.01, temp: 1.0, samples/prompt: 4

### `guess_number_1gpu.sh`
- **Env:** GuessNumberEnv — guess a secret number between 1 and 100
- **Execution:** local, 1 GPU
- Dataset: `hamishivi/rlenv-guess-number`
- episodes: 48, max_steps: 10, response_length: 1024, beta: 0.01, temp: 0.7, samples/prompt: 2

### `guess_number_beaker.sh`
- Same as above but **on Beaker** (1 GPU, any of jupiter/saturn/ceres)
- Dataset: `hamishivi/rlenv-guess-number-nothink` (`-nothink` variant suppresses CoT)
- episodes: 400, max_steps: 10, timeout: 30m, adds `--with_tracking`

---

## SandboxLM — math via Docker code execution

Dataset: `allenai/Dolci-RLZero-Math-7B`. Tools: `generic_sandbox`. beta=0.0.
The model writes and executes Python code inside Docker to answer math questions.
System prompt: `sandbox_lm_system_prompt.txt`.

### `sandbox_lm_1gpu.sh`
- **Execution:** local, 1 GPU
- Model: Qwen3-0.6B
- response_length: 8192, pack_length: 16384
- 1 learner / 1 engine, LR: 1e-6, pool_size: 4
- No inflight updates, no prefix caching

### `sandbox_lm_8gpu.sh`
- **Execution:** Beaker, 1 node / 8 GPUs (ai2/jupiter)
- Model: Qwen3-4B-Instruct-2507
- response_length: 30720, pack_length: 32768
- 4 learners / 4 engines, LR: 5e-7, pool_size: 16
- `inflight_updates: True`, async_steps: 4, prefix caching enabled

---

## SWERL Sandbox — agentic tasks with per-sample Docker containers

Dataset: `hamishivi/agent-task-combined` (except the openthoughts variant).
Tools: `swerl_sandbox`. Each task sample runs in its own isolated container.
Provides `execute_bash`, `str_replace_editor`, and `submit` tools.

### 1-GPU local variants

#### `swerl_sandbox_1gpu.sh`
- Model: Qwen3-0.6B — minimal resource baseline
- Backend: Docker
- response_length: 4096, pack_length: 8192, beta: 0.01, LR: 3e-7
- episodes: 48, max_steps: 10, pool_size: 8
- tool_parser: `vllm_hermes`
- No active_sampling, no inflight_updates, no advantage normalization override

#### `swerl_sandbox_1gpu_qwen35.sh`
- Model: Qwen3-4B-Instruct-2507 — larger model on 1 GPU
- Backend: Docker
- response_length: 4096, pack_length: 8192, beta: 0.01, LR: 3e-7
- episodes: 16, max_steps: **100** (longer debug run), pool_size: 4
- tool_parser: `vllm_hermes`
- `no_resampling_pass_rate: 0.875`
- system_prompt: `swerl_sandbox_system_prompt.txt`

#### `swerl_sandbox_openthoughts_1gpu.sh`
- Model: Qwen3.5-0.8B
- Dataset: `hamishivi/agent-task-openthoughts` (different dataset)
- Backend: **Apptainer** (not Docker) — requires `apptainer` on PATH
- response_length: 8192, pack_length: 10240, beta: 0.0, LR: 1e-6
- tool_parser: **`vllm_qwen3_xml`** (not hermes)
- pool_size: 8, episodes: 64, max_steps: 10
- `active_sampling: true`, `inflight_updates: true`
- `advantage_normalization_type: centered`
- `verification_reward: 1.0`, `truncated_importance_sampling_ratio_cap: 0.0`
- **Most production-faithful 1-GPU script** — mirrors the 4-node config as closely as possible

### Multi-GPU Beaker variants

#### `swerl_sandbox_8gpu.sh`
- **Scale:** 1 node / 8 GPUs (ai2/jupiter), workspace: `ai2/open-instruct-dev`
- Model: Qwen3-4B-Instruct-2507
- response_length: 30720, pack_length: 32768
- DeepSpeed stage 3, sequence_parallel_size: 4
- 4 learners / 4 engines, LR: 3e-7, beta: 0.01
- async_steps: 8, inflight_updates: true
- active_sampling: true, no_resampling_pass_rate: 0.875
- pool_size: 128
- tool_parser: `vllm_hermes`
- Requires Docker socket + Docker login via `hamishivi_DOCKER_PAT` secret

#### `swerl_sandbox_4node.sh`
- **Scale:** 4 nodes / 32 GPUs (ai2/jupiter), workspace: `ai2/olmo-instruct`
- Model: Qwen3-4B-Instruct-2507 (base instruct, no SFT warm-start)
- response_length: 32768, pack_length: 35840
- DeepSpeed stage 3, sequence_parallel_size: 4
- 8 learners / 24 engines, LR: 1e-6, lr_scheduler: constant, beta: 0.0
- async_steps: 8, inflight_updates: true
- active_sampling: true, no_resampling_pass_rate: 0.875
- pool_size: 128, backend_timeout: 1200, checkpoint_state_freq: 50
- `advantage_normalization_type: centered`
- `truncated_importance_sampling_ratio_cap: 2.0`
- total_episodes: 100000000 (unlimited, runs until stopped or preempted)

#### `swerl_sandbox_4node_sft.sh`
- Identical to `4node.sh` except:
  - Model: `hamishivi/sft_qwen3_4b_tmax_4node2203` (**SFT warm-start** on tmax data)
  - async_steps: **4** (half of 4node)
  - pool_size: **256** (double of 4node)
  - exp_name: `swerl_sandbox_qwen3_4b_sft_tmax_4node_grpo`

---

## Wordle

Both use: `PrimeIntellect/Qwen3-1.7B-Wordle-SFT`, dataset `hamishivi/rlenv-wordle-nothink`,
`reward_aggregator: last`, `advantage_normalization_type: centered`, beta: 0.0, LR: 1e-6,
`inflight_updates: True`, DeepSpeed stage 2, 4 learners / 4 engines, Beaker 8 GPU.

### `wordle_8gpu.sh`
- No tools — pure text guessing via `<guess>` tags, no Docker required
- response_length: 8192, pack_length: 16384
- prompts/rollout: 64, samples/prompt: 16, per_turn_max_tokens: 1024
- async_steps: 1, truncated_IS_ratio_cap: 5.0
- tool_parser: `vllm_hermes`

### `wordle_sandbox_lm_8gpu.sh`
- Tools: `wordle` + `generic_sandbox` — model can use Docker code execution while playing Wordle
- Requires Docker socket
- response_length: **4096** (shorter), pack_length: 16384
- prompts/rollout: 16, samples/prompt: 4, per_turn_max_tokens: **512**
- tool_parser: **`vllm_qwen3xml`** (different from wordle_8gpu)
- pool_size: 16, `no_filter_zero_std_samples`
- system_prompt: `wordle_sandbox_lm_system_prompt.txt`

---

## Utility scripts

### `download_swerl_data.sh`
Downloads `hamishivi/agent-task-combined` from HuggingFace and extracts `task-data.tar.gz`. Source this before local swerl runs to pre-cache the task data.

### `swerl_sandbox_apptainer_smoke.sh`
Not a training script. Runs `scripts/debug/apptainer_backend_smoke.py` to verify Apptainer works on the host (instance start, fakeroot, file I/O, exec, timeout). Use this on Slurm nodes (`salloc`) before running `openthoughts_1gpu.sh`. Override image with `$APPTAINER_TEST_IMAGE`.

---

## Quick comparison axes

| Axis | Values |
|---|---|
| Container backend | Docker, Apptainer, none |
| Tool parser | `vllm_hermes`, `vllm_qwen3_xml`, `vllm_qwen3xml` |
| Scale | local 1 GPU → Beaker 8 GPU → Beaker 32 GPU (4-node) |
| Beta | 0.0 (no KL) or 0.01 |
| Active sampling | production runs use it; simple debug runs don't |
| Inflight updates | enabled at 8+ GPU scale and in openthoughts_1gpu |
| Advantage normalization | default or `centered` |
| Model warm-start | base instruct vs SFT checkpoint (`sft_qwen3_4b_tmax`) |
