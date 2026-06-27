# AppWorld RL (code-as-action)

Train a policy on [AppWorld](https://appworld.dev) tasks: the model writes Python that
drives a simulated world of apps (Amazon, Spotify, Venmo, phone, file system, …) via a
single `execute_python` tool, and is rewarded by AppWorld's own programmatic evaluation
(fraction of unit tests passed).

## Execution model: per-rollout container + HTTP (mirrors swerl)

AppWorld pins **pydantic <2**; open-instruct/openenv need **pydantic ≥2** — they can't
share a process. So we use AppWorld's HTTP *environment server* and mirror the Terminal
RL (swerl) podman/docker workflow:

- One AppWorld container per rollout (`ghcr.io/stonybrooknlp/appworld:latest`), running
  `appworld serve environment`. The server holds one task world at a time → one container
  per concurrent rollout, reused across resets (re-`/initialize` per task).
- `AppWorldEnv` is a thin **HTTP client** (`/initialize`, `/execute`, `/task_completed`,
  `/evaluate`, `/close`) that **never imports `appworld`** — the trainer stays pydantic-2-clean.
- Unlike swerl's stateless `docker exec bash`, AppWorld is a stateful Python REPL (agent
  variables persist across turns), so the container runs a long-lived server and we talk
  HTTP, rather than exec-per-turn.
- Networking: locally the env reaches the container on its **bridge IP** + in-container
  port; for a remote `docker_host` it uses the published host port. (Same sibling-container
  caveat as the harbor/tmax evals.)
- The trainer needs a docker/podman socket, exactly like swerl.

## Pieces

| Path | What |
|------|------|
| [open_instruct/environments/appworld_env.py](../../../open_instruct/environments/appworld_env.py) | `AppWorldEnv` (config_name `appworld`): container lifecycle + HTTP client + prompt builder. |
| [scripts/data/convert_appworld_to_rl.py](../../data/convert_appworld_to_rl.py) | Build the RLVR dataset (reads `specs.json` off disk; no `appworld` import). |
| [smoke_test_appworld_env.py](smoke_test_appworld_env.py) | End-to-end env check (starts a real container; reset → execute → evaluate → reward). |
| [rl/qwen35_4b_appworld.sh](rl/qwen35_4b_appworld.sh) | GRPO launch script. |

## Setup (image + data)

```bash
docker pull ghcr.io/stonybrooknlp/appworld:latest

# Get the task data (run once, in a throwaway pydantic-1 venv — NOT the trainer venv):
pip install appworld && appworld install && appworld download data --root $APPWORLD_ROOT
```

The **docker daemon of each node** must be able to provide the data to the container. Two
options:

- **Bind-mount (default)**: stage `$APPWORLD_ROOT` (contains `data/` and
  `experiments/outputs/`) on a path the daemon sees (e.g. weka) and pass `APPWORLD_ROOT`.
  The env mounts `data/`→`/run/data` (ro) and `experiments/outputs/`→`/run/experiments/outputs` (rw).
- **Data-baked image**: build a derivative and set `APPWORLD_ROOT=""` (no mount). Useful
  where the daemon has no shared FS with the host:
  ```dockerfile
  FROM ghcr.io/stonybrooknlp/appworld:latest
  COPY data /run/data
  RUN mkdir -p /run/experiments/outputs
  ```
  `docker build -t <org>/appworld-data:latest .` then set `APPWORLD_IMAGE` to it.

> `appworld` is intentionally NOT a dependency of this repo (pydantic conflict) — it lives
> only inside the container image.

## Build the dataset

```bash
uv run python scripts/data/convert_appworld_to_rl.py \
    --data_root $APPWORLD_ROOT --splits train,dev,test_normal,test_challenge --private \
    --push_to_hub <org>/appworld-train-rl
```

Each row: `messages` (system with supervisor creds + user instruction), `ground_truth`
(task_id), `dataset="passthrough"` (env emits the reward), `tools=["appworld"]`,
`env_config` (per-row `task_id` + `max_steps`).

## Smoke-test the env (needs Docker + image + data)

```bash
uv run python scripts/general_agent/appworld/smoke_test_appworld_env.py --data_root $APPWORLD_ROOT
# or against a data-baked image:
#   ... --image <org>/appworld-data:latest --data_root ""   # (edit the script's data_root)
```

## Launch RL

```bash
APPWORLD_DATASET=<org>/appworld-train-rl APPWORLD_ROOT=/weka/.../appworld_root \
  ./scripts/train/build_image_and_launch.sh scripts/general_agent/appworld/rl/qwen35_4b_appworld.sh
```

## Design notes

- **Reward**: `/evaluate` returns `passes`/`failures`/`num_tests`/`success`. `dense_reward=true`
  uses `len(passes)/num_tests`; else binary all-pass (Task Goal Completion). Returned as the
  env step reward.
- **Completion**: the agent calls `apis.supervisor.complete_task()`; the env detects it via
  `/task_completed`. On hitting `max_steps` the env evaluates the partial state.
- **Isolation**: one container per rollout; unique `experiment_name` per rollout; container
  torn down on close. Model code runs inside the container, away from the trainer.
- **Pool sizing**: `pool_size` = max concurrent rollouts (one container each), same as swerl.
