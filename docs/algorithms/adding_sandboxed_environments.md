# Adding a sandboxed RL environment to open-instruct

A field guide for wiring a new **containerized / sandboxed** environment into the GRPO
RL pipeline, written from the end-to-end AppWorld integration. Read this before adding
the next one (terminal variants, web sandboxes, OS-agent worlds, MCP-backed apps, …) —
it captures the architecture, the exact dataset/launch contract, the two integration
styles, every trap that cost time, and a copy-paste checklist.

Companion docs: [rl_with_environments.md](rl_with_environments.md) (basic tool/env intro),
[multi_task_rl.md](multi_task_rl.md) (per-example routing, dataset columns),
[rollout_loop_internals.md](rollout_loop_internals.md) (token-level rollout),
[tmax_4b_script_reference.md](tmax_4b_script_reference.md) (annotated swerl launch).
Worked example: [scripts/general_agent/appworld/](../../scripts/general_agent/appworld/).

> **How to read this doc.** Part 1 is a plain-language mental model — start here if you're
> new to how this RL pipeline works. Part 2 onward is the engineering reference (exact files,
> the dataset contract, the traps). You can act on Part 2 alone once Part 1 clicks.

---

# Part 1 — Plain-language mental model (start here)

## What we're actually doing

We're training a model to *do tasks*, not just answer questions. The model takes actions
(run a shell command, call an API, write code), something **responds**, the model acts again,
and at the end we **score** how well it did. We then nudge the model toward the behavior that
scored well. That's reinforcement learning (RL); the specific algorithm here is **GRPO**, but
you don't need its math to add an environment.

One **attempt at one task** is called a **rollout**: prompt → model acts → world responds →
… → done → a reward number. Training runs thousands of rollouts and learns from them.

## The "environment" and why it's a sandbox

The **environment** is the little world the model acts in. For agentic tasks that world is
usually a **sandbox**: an isolated computer (a **container**) created fresh for the task.
We use containers for three reasons:

1. **Safety** — the model writes arbitrary code/commands; run them in a throwaway box, not on
   the training machine.
2. **Reproducibility** — every task starts from an identical clean state.
3. **Dependency isolation** — the task's software (and its library versions) stay inside the
   container and can't clash with the trainer's. (This one turned out to matter a lot for AppWorld.)

The environment's job each turn: take the model's action, run it in the container, hand back
what happened (the "observation"), and — when the task is finished — compute the **reward**.

## The cast of characters (all running at once)

Think of a training step as a small factory, with these workers (each is a separate process;
**Ray** is the glue that runs and connects them, possibly across several machines):

- **The learner** — holds the model's weights and does the actual learning (gradient updates).
  Lives on GPUs.
- **The generator (vLLM)** — runs the current model *fast* to produce the next action/text.
  Also on GPUs. (Generating and learning are split for speed.)
- **The environment pool** — a stash of ready-to-go environment instances. Because we run many
  rollouts in parallel, we keep a *pool* of, say, 128 environments warm; each rollout borrows
  one, uses it, returns it. For a container env, **one warm environment ≈ one running container**,
  so the pool size is also how many containers run at once (watch the RAM).
- **A dataset** — a list of tasks. Each **row** is one task and carries: the prompt, *which*
  tool/environment to use, the task id, and how to score it.

## How one rollout flows (the narrated version)

1. The pipeline picks a task row and **borrows an environment from the pool**, which **starts
   (or re-initializes) the container** for that task.
2. The **generator** produces the model's first action — for a tool-using agent, that's a
   **tool call** like `execute_python({"code": "..."})`.
3. The environment **runs that action in the container** and returns an **observation** (stdout,
   an API response, an error). The observation is appended to the conversation.
4. Steps 2–3 repeat (multi-turn) until the model signals it's done, or a step limit is hit.
5. The environment **computes the reward** (e.g. "fraction of the task's tests that pass").
6. The reward + the conversation go back to the **learner**, which updates the weights.

A few non-obvious facts that follow from this flow (and bite people):

- The **prompt lives in the dataset row**, not in the environment. The environment's "welcome
  message" from step 1 is actually thrown away by the pipeline — so the task instructions must
  be written into the dataset.
- The **environment supplies the tool's definition** (its name + arguments). The dataset just
  says *which* tool name is allowed; the schema the model sees comes from the environment.
- The **reward comes from the environment** (we set the dataset's separate "verifier" to a
  no-op called `passthrough` so it doesn't double-count).

## Where the container actually runs (local vs. the cluster)

- **On your dev box / locally:** a normal Docker daemon. Easy.
- **On the Beaker cluster:** each node runs a fleet of **podman** services (podman = a
  Docker-compatible container engine). The trainer container spawns the task containers as
  **siblings** next to it. There's also a **registry mirror** (a local cache of Docker Hub) so
  pulling images is fast. There's a standard recipe of environment variables + a startup script
  that turns all this on — you copy it from an existing script; you don't invent it.

## The two ways an environment can work

Both kinds of environment are *stateful* — the difference is **where the state lives** and
whether a brand-new process can still see it:

- **Style A — "run a command, get output."** State lives in the container's **files**. Each turn
  is a fresh shell, but it sees everything previous turns wrote to disk, so a simple "run this
  command in the box" model works (the repo's ready-made container helper does this). The only
  thing a fresh shell forgets is the current folder and environment variables, which the terminal
  envs quietly save to a file and reload each turn. Example: the terminal/coding sandboxes (`swerl`).
- **Style B — "talk to a long-running program."** State lives in a **live program's memory** — a
  Python session with variables, a loaded "world", open connections, a browser. You can't save
  that to a file and reload it the way you can a folder path, so the container has to keep **one
  long-running program** alive; our environment is a thin **client** that sends it requests over
  HTTP. Example: **AppWorld**.

AppWorld is style B for two reasons: it's a Python session where the agent's variables must
persist across turns, **and** its software needs an old library version that clashes with the
trainer's — so it *has* to live in its own container, talked to over HTTP, never imported into
the trainer. The container is both the sandbox and the "keep its dependencies away from ours" wall.

(And don't worry that style B is heavy: the container is started **once and reused** for many
tasks — not per attempt — so the setup cost is amortized, just like the terminal sandboxes.)

## What "adding an environment" therefore means

1. Write a small Python class that knows how to **start the container, send the model's action
   to it, read back the observation, and compute the reward** (`reset` / `step` / `close`).
2. **Register** that class under a name.
3. Build a **dataset** of task rows in the exact format the pipeline expects.
4. Write a **launch script** that turns on the container machinery and points at your env + dataset.
5. **Test it in three stages**: the env alone → a tiny local training run → the real cluster run.

Part 2 is the precise, file-by-file version of those five steps, plus every trap we hit.

## Mini-glossary

- **Policy / model** — the thing being trained.
- **Rollout** — one attempt at one task; ends in a reward.
- **GRPO** — the RL training algorithm used here (details not needed to add an env).
- **RLVR data** — the dataset format for RL ("RL with verifiable rewards"): rows of task +
  how-to-score, *not* example answers.
- **Environment / env** — the world the model acts in (here, usually a container sandbox).
- **Tool** — an action the model can take, with a name + arguments (e.g. `execute_python`).
- **Observation** — what the environment returns after an action.
- **Reward** — the score for a rollout, produced by the environment.
- **Pool** — a set of warm environment instances for running rollouts in parallel.
- **Container / image** — an isolated mini-computer / its frozen template.
- **Ray** — runs the many pieces (learner, generator, envs) as parallel processes ("actors").
- **vLLM** — the fast model-serving engine used to generate actions.
- **Beaker** — Ai2's compute cluster. **podman** — the container engine on cluster nodes.
- **Registry mirror** — a local cache of Docker Hub on the cluster, for fast image pulls.

---

# Part 2 — Engineering reference

## 0. TL;DR — the two integration styles

There are two ways a sandboxed env runs, and which one you pick decides everything else:

| | **A. Filesystem-state (swerl model)** | **B. Live-session server (AppWorld model)** |
|---|---|---|
| Interaction | `docker exec bash -c <cmd>` per turn | long-lived server in the container; env is an HTTP client |
| State between turns | container **filesystem** (+ cwd/env serialized to files) | a **live process**: REPL vars, imports, in-memory world/handles |
| Reuses `backends.py` | **Yes** (`DockerBackend.run_command`) | **No** — needs network reachability to the container |
| In-process possible? | n/a (always container) | only if the env's deps don't conflict with the trainer's |
| Examples | `swerl_sandbox`, `swerl_vanillux_sandbox`, `generic_sandbox` | `appworld` |

**"Stateful vs stateless" is the wrong axis — both are stateful. The real question is *where the
state lives* and whether a fresh process can see it:**

- **Style A:** state is the container **filesystem**, which every new `docker exec` shell sees.
  The only per-shell state a fresh `exec` loses is **cwd + env vars**, and swerl explicitly
  serializes those to files and reloads them each call (see the bash wrapper in
  `swerl_vanillux_sandbox.py`). Everything serializable → stateless-per-call `exec` works, no
  network, no ports.
- **Style B:** state is a **live in-memory session** — Python REPL variables, imported modules,
  an `apis` object with open DB handles. You cannot serialize/reload that between fresh
  processes the way you can cwd/env, so you need **one long-lived process** holding it = a
  server, reached by a client. (AppWorld persists its *world DB* to disk each step, but the
  agent's REPL namespace — `x = apis.amazon.search(...)` reused next turn — only lives in the
  process.)

Decision rule: filesystem (or trivially serializable) state → **A**; live interpreter/browser/
session state → **B**.

> AppWorld forced style B for two reasons: (1) it's a Python REPL where the agent's in-memory
> variables persist across turns, and (2) AppWorld pins `pydantic<2` while open-instruct/openenv
> need `pydantic>=2`, so it **cannot be imported in the trainer process at all**. The container
> is both the sandbox *and* the dependency quarantine.

**Does style B scale?** Yes — the per-container cost people worry about isn't per-rollout. The
container (and its port) is created **once per pool actor** and reused across all tasks (each
`reset()` just re-`/initialize`s over HTTP; each turn is one HTTP call) — exactly how swerl
reuses its `sleep infinity` container. Allocating a free port is a single `bind(0)`/`getsockname`
syscall done ~once per actor (a few hundred times over a multi-hour run), and HTTP-over-localhost
per turn costs about the same as a `docker exec`. The real per-node costs (container RAM, podman
create throughput) are the same shape as swerl; AppWorld's only extra is its server loading
apps/DBs at startup. Ephemeral ports (~28k) vastly exceed containers-per-node, and the tiny
bind→start race self-heals via reset retry.

---

## 1. The environment/tool system (what the repo gives you)

### 1.1 The `RLEnvironment` ABC — `open_instruct/environments/base.py`

Every env subclasses `RLEnvironment` (or `TextRLEnvironment`). The contract:

```python
class RLEnvironment(ABC):
    config_name: str = ""            # registry key, e.g. "appworld"
    response_role: str = "tool"      # role used to inject observations back ("tool" or "user")

    async def setup(self) -> None: ...                 # once per actor at startup (clients, downloads)
    @classmethod
    def get_tool_definitions(cls) -> list[dict]: ...   # OpenAI-format tool schemas the model sees
    async def reset(self, **kwargs) -> tuple[StepResult, list[dict]]: ...  # start episode -> (obs, tools)
    async def step(self, call: EnvCall) -> StepResult: ...                 # one action -> obs + reward + done
    def state(self) -> State: ...
    def get_metrics(self) -> dict[str, float]: ...     # custom wandb metrics
    async def close(self) / shutdown(self): ...        # teardown
```

- **`EnvCall`** = a parsed tool call (`id`, `name`, `args`). For tool-style envs the rollout
  loop parses the model's tool call and dispatches `step(call)`. `call.name` is the **tool
  function name** (e.g. `execute_python`), not the config_name.
- **`StepResult`** = `result: str` (the observation text injected back), `reward: float`,
  `done: bool`, `metadata: dict`.
- **`TextRLEnvironment`** (subclass) is for envs that consume the model's *entire* generation
  as text (no parsed tool call) — implement `text_step(text)` and `_reset()`. `swerl_vanillux`
  is tool-style; `wordle_text_env` is text-style. Tool-style is usually the right default for a
  sandbox (it composes with the tool-calling rollout machinery and the qwen3_xml parser).
- **`BaseEnvConfig`** (dataclass) pairs a config with the env via `tool_class: ClassVar`.
  Config fields become `__init__` kwargs of the env (the pool expands the dataclass).

### 1.2 Registration — `open_instruct/environments/tools/tools.py`

Add one import + one `TOOL_REGISTRY` entry (`~line 652`):

```python
TOOL_REGISTRY[MyEnvConfig.tool_class.config_name] = MyEnvConfig
```

`tools/tools.py` is imported at trainer startup, so **the env module must import cleanly
without the env's heavy/optional deps** — import them lazily inside `setup()` (see §6 trap 6).

### 1.3 The pool — `open_instruct/environments/pool.py`

`EnvironmentPool` is a Ray actor holding `pool_size` env-actor instances. Key behavior:

- **One actor per concurrent rollout.** `pool_size` should be ≥ `num_unique_prompts_rollout
  * num_samples_per_prompt_rollout` (the max concurrent rollouts). For heavyweight container
  envs this is also the max concurrent containers — size accordingly (RAM!).
- **`acquire_reset(reset_kwargs)`** acquires a free actor and calls `actor.reset(**kwargs)`,
  returning `(actor, tool_defs)`. **The reset `StepResult` observation is dropped** for tool
  envs — the prompt comes entirely from the dataset `messages` (see §2).
- **Podman host rotation:** the pool injects a `docker_host` into `reset_kwargs` by leasing
  from `SWERL_PODMAN_DOCKER_HOSTS` — but **only when the env config has `backend == "docker"`**
  (`pool.py` ~line 74). Your config dataclass **must** have a `backend: str = "docker"` field
  or your containers never land on the beaker podman hosts. The env's `reset` should read
  `kwargs.get("docker_host")` and (re)connect to it.
- Reset is retried with backoff; host-connectivity failures rotate to the next podman host.

### 1.4 Backends — `open_instruct/environments/backends.py`

`DockerBackend` / `ApptainerBackend` implement `start / run_command(bash) / write_file /
read_file / put_archive / close`, with `docker_host` support and concurrency semaphores
(`SWERL_DOCKER_START_CONCURRENCY`, `SWERL_DOCKER_EXEC_CONCURRENCY`). **Reuse these for style A**
(bash-exec). For style B you typically don't use the backend's `run_command` (you make HTTP
calls), but you can still mirror its container *lifecycle* (start/stop, `docker_host`).

### 1.5 How envs reach the rollout — `open_instruct/grpo_fast.py`

- `create_tool_pools(parsed_tools, pool_size)` (`~1962`): builds one `EnvironmentPool` per
  `--tools` entry, **keyed by call name** (`pools[call_name] = ...`; call name = `--tool_call_names`
  or the config_name if not remapped).
- `_discover_tools_from_datasets(...)` (`~3050`): scans the dataset `tools` column **and**
  `env_config.env_name`s and **auto-creates a pool** for any registry name not already
  configured. ← this is where the duplicate-pool trap lives (§6 trap 3).
- `initialize_tools_and_envs(...)` (`~3085`): calls the above, collects
  `tool_definitions = [d for pool in pools for d in pool.get_tool_definitions()]`, returns
  `(pools, tool_definitions, stop_sequences)`.
- `_validate_and_log_dataset_tools(...)` (`~1876`): asserts the dataset `tools` names ⊆
  configured call names. A mismatch is a hard error here.

### 1.6 The rollout loop — `open_instruct/vllm_utils.py`

- `process_request` → `_acquire_and_reset_pools(...)` (`~1042`) runs **before generation**:
  it acquires + resets one actor from *every configured pool* (so a heavyweight container
  starts at the top of every rollout — see the "over-acquire" note in
  [multi_task_rl.md §1.2](multi_task_rl.md)).
- `allowed_tools = configured_tools(pool keys) & active_tools(dataset tools column)`, then the
  env's tool function names are added to `allowed_tools` at register time. Tool calls are
  filtered to `allowed_tools` (`~1231`).
- Each `step()` result's `.result` is injected back as an observation with the env's
  `response_role`; `.reward` is appended to `rollout.rewards` (`~1285`). On `done`, the rollout
  ends.
- **Hard limit:** only **one text env** per rollout (`~1090`). Tool envs are unrestricted.

### 1.7 Reward flow — `open_instruct/ground_truth_utils.py`

- The env emits reward via `StepResult.reward`. To make the env's reward the *only* reward,
  set the dataset `dataset` column to **`passthrough`** (the `passthrough` verifier returns 0,
  `~line 1037`), so total reward = env reward + 0. This is the swerl/appworld pattern.
- `apply_verifiable_reward` (`~1253`) selects the verifier per-row by the `dataset` column.

---

## 2. The dataset (RLVR) contract — get this exactly right

One row per task. Columns (`open_instruct/dataset_transformation.py`):

| Column | Value for a sandbox env | Notes |
|--------|------------------------|-------|
| `messages` | `[{system, …}, {user, instruction}]` | The **prompt**. The env reset observation is **dropped**, so the task instruction must live here. Bake the system prompt in (or use `--system_prompt_override_file`). |
| `ground_truth` | usually the **task id** (a string) | The env verifies internally; this is just an identifier. |
| `dataset` | `"passthrough"` | Env emits the reward; no extra verifier. |
| `tools` | `["<tool FUNCTION name>"]` e.g. `["execute_python"]` | **NOT the config_name, NOT schema dicts.** See below. |
| `env_config` | `{"env_configs": [{"env_name": "<POOL KEY>", "task_id": …}], "max_steps": N}` | `env_name` must equal the **pool key = call name** (§6 trap 3). |

### The `tools` column is the single most error-prone field

There are **two** tool representations in the codebase and they are NOT interchangeable:

- **RL (RLVR) data → name strings.** `rlvr_tokenize_v1` (`dataset_transformation.py:~1539`)
  does `active_tool_names = set(row["tools"])` then filters the env-provided global
  `tool_definitions` by `t["function"]["name"] in active_tool_names`. So the column must hold
  the tool **function name(s)** that match `get_tool_definitions()`. A schema dict here fails
  the name match → **no tool is injected** and the model silently never gets the tool.
- **SFT data → schema dicts.** `_normalize_tools_for_chat_template` (`:~976`) expects the
  column to *be* OpenAI-style schema dicts. That's the SFT path; don't use it for RL.

Verify empirically (cheap, do this for every new env):

```python
import open_instruct.dataset_transformation as dt
from datasets import load_dataset
out = dt.rlvr_tokenize_v1(dict(load_dataset("parquet", data_files=PARQUET, split="train")[0]),
                          tokenizer, tool_definitions=[MY_TOOL_SCHEMA])
assert "my_tool_name" in tokenizer.decode(out[dt.INPUT_IDS_PROMPT_KEY])  # schema actually injected
```

### Launch flags that pair with the dataset

```
--tools <config_name>            # e.g. appworld   (registry key; creates the pool)
--tool_call_names <function>     # e.g. execute_python  (pool is keyed by THIS)
--tool_configs '{"image": "...", ...}'   # run-level env kwargs (one JSON per --tools entry)
```

Keep the names consistent: **pool key = call name = dataset `tools` entry = `env_config.env_name`**.
If `get_tool_definitions()`'s function name differs from the config_name (like AppWorld:
config_name `appworld`, function `execute_python`), you *must* remap via `--tool_call_names`
and set `env_config.env_name` to the **call name**, not the config_name (§6 trap 3).

Local datasets: `dataset_mixer_list` accepts a local `.parquet`/`.jsonl` path (train split only)
— great for debug runs with no HF push.

---

## 3. Containers on Beaker (the podman recipe)

Sandboxes don't run a local docker daemon on the trainer; they use a fleet of **podman
services** on each node. The canonical recipe is
[scripts/general_agent/terminal/rl/qwen35_4b_base_tmax_10k.sh](../../scripts/general_agent/terminal/rl/qwen35_4b_base_tmax_10k.sh).
Copy its mason block. The load-bearing pieces:

- `--env BEAKER_ALLOW_SUBCONTAINERS=1` and `--env BEAKER_SKIP_DOCKER_SOCKET=1` — allow the
  trainer container to spawn sibling containers.
- `--env SWERL_PODMAN_SERVICE_COUNT=8` — number of podman service instances (the "docker hosts").
- `--env MIRROR_URL=jupiter-cs-aus-193.reviz.ai2.in:5000` — the cluster **registry mirror /
  pull-through cache**. It's consumed by the host podman config (not the Python code), so plain
  `docker.io/...` image refs are pulled through it automatically — **no prefix needed** in your
  `--tool_configs` image.
- `--env DOCKERHUB_USERNAME=<user>` + `--secret DOCKER_PAT=<user>_DOCKER_PAT` — docker login
  (avoids rate limits; required to pull private images).
- `--env CONTAINERS_STORAGE_CONF=...`, `PODMAN_NUM_LOCKS=...`, `SWERL_DOCKER_AUTO_REMOVE=1`,
  `SWERL_RESET_FAILURE_ZERO_REWARD=1` (reset failure → zero reward instead of crashing — keep this on),
  image-janitor envs (cleanup).
- **Entry command:** `source scripts/docker/docker_login.sh && source configs/beaker_configs/ray_node_setup.sh && python open_instruct/grpo_fast.py …`
  - **`scripts/docker/docker_login.sh`** is the one that **starts the podman services and sets
    `SWERL_PODMAN_DOCKER_HOSTS`** (despite the name; `ray_node_setup.sh` does *not*). The pool
    then injects those hosts as `docker_host` into env resets (requires the `backend="docker"`
    config field — §1.3).

Launch with **`scripts/train/build_image_and_launch_dirty.sh <script>`** when the working tree
is dirty (the non-`_dirty` builder refuses uncommitted changes). It builds the trainer image
and passes it to the script as `$1` (`BEAKER_IMAGE`).

### Networking to the container (style B only) — the big one

**Beaker's podman services run every container on the _host_ network.** Verified by probe:
`NetworkSettings.Networks = {"host": …}`, `IPAddress=""`, `Ports={}` even when you ask to
publish. So:

- There is **no bridge IP** and **port publishing is a no-op** — `docker exec` (style A) doesn't
  care, but an HTTP client (style B) has nothing to connect to via those paths.
- The server **is** reachable at **`127.0.0.1:<port>`** from the trainer process, because the
  podman services run inside the trainer's beaker container, so its host netns *is* the
  trainer's netns. (Probe: `GET http://127.0.0.1:<port>/ → 200`.)
- Because all containers share that one netns, **they collide on a single port** → each
  container must bind a **unique port**. Allocate a free port per container in the trainer
  process (same netns) and pass it as the server's `--port`.

So for style B on Beaker: **don't publish; give each container a unique free port; reach it at
`127.0.0.1:<that port>`.** Keep the container **bridge IP** as a fallback candidate for *local
docker* (default bridge network, where `127.0.0.1` won't reach it but the bridge IP will). See
`AppWorldEnv._pick_free_port` / `_candidate_base_urls` / `_wait_for_server`.

This was the cause of a 4h "hang": the env only probed the (empty) bridge IP and published
port, never `127.0.0.1`, so every reset failed → zero-reward no-ops → idle GPUs. Diagnose with
[scripts/general_agent/appworld/debug/probe_container.py](../../scripts/general_agent/appworld/debug/probe_container.py)
(a no-training job that starts the container and prints status/exit/logs/`NetworkSettings`/reachability),
and **keep `auto_remove=False` + capture container logs on early exit** — `auto_remove=True`
deletes the evidence and you go blind.

---

## 4. Validation ladder (always climb it in order)

1. **Pin the env's real API** in a throwaway venv: `python -m venv /tmp/x && /tmp/x/bin/pip
   install <pkg>`, then read its source for exact signatures. Never guess an external API.
2. **Standalone env smoke** (no GRPO): instantiate the env, `setup()`, `reset()`, a few
   `step()`s, confirm reward. Catches the env logic + container lifecycle + reward path.
3. **Local 2-GPU GRPO** ([scripts/general_agent/appworld/rl/local_rl_2gpu.sh](../../scripts/general_agent/appworld/rl/local_rl_2gpu.sh)):
   Qwen3-0.6B, local parquet, data-baked image, no HF/Beaker. Catches the dataset contract,
   pool wiring, prompt enqueue, the duplicate-pool trap, tokenization. **This is where most
   integration bugs surface** — run it before ever touching Beaker.
4. **Beaker** via `build_image_and_launch_dirty.sh`. Catches podman-host injection, image pull
   via mirror, and (style B) container networking.

A tiny debug model won't reliably emit tool calls in-loop — that's expected; use a capable
model (4B+) to actually exercise the tool, but the small model still validates the *plumbing*.

---

## 5. AppWorld, concretely (the worked example)

- **What it is:** an interactive coding benchmark; the agent writes Python that calls
  `apis.<app>.<endpoint>(...)` against a stateful simulated world and submits via
  `apis.supervisor.complete_task()`. Reward = AppWorld's own unit-test pass fraction.
- **API (pinned against `appworld==0.1.3.post1`):** `AppWorld(task_id, experiment_name, …)` →
  `.execute(code)->str`, `.task_completed()->bool`, `.evaluate(suppress_errors=True)->TestTracker`
  (`.success`, `.pass_count`, `.num_tests`; the HTTP server returns `passes`/`failures` lists +
  `num_tests`). Per-app login passwords come from `apis.supervisor.show_account_passwords()` at
  **runtime** (not the prompt). Supervisor name/email/phone are synthetic.
- **Why container+HTTP:** `pydantic<2` pin (can't share the trainer process) **and** REPL state
  must persist. AppWorld ships an HTTP **environment server** (`appworld serve environment`,
  image `ghcr.io/stonybrooknlp/appworld:latest`) built for exactly this. Endpoints:
  `/initialize`, `/execute`, `/task_completed`, `/evaluate`, `/close`. **One world per server →
  one container per concurrent rollout** (reused across resets via re-`/initialize`).
- **Data:** `appworld install` + `appworld download data` (~193 MB). Either bind-mount a weka
  data root (`data_root=APPWORLD_ROOT`) or bake it into the image (`data_root=""`). The data is
  the "protected portion" — Apache-2.0 **but public redistribution must be encrypted**, so keep
  the dataset and any data-baked image **private**.
- **Implementation:** [open_instruct/environments/appworld_env.py](../../open_instruct/environments/appworld_env.py)
  (env = thin HTTP client, never imports `appworld`),
  [scripts/data/convert_appworld_to_rl.py](../../scripts/data/convert_appworld_to_rl.py)
  (reads `data/tasks/<id>/specs.json` off disk; no `appworld` import),
  [scripts/general_agent/appworld/](../../scripts/general_agent/appworld/) (launch + smoke + README).

---

## 6. Traps that cost time (check every one for a new env)

1. **`tools` column = function names, not config_name, not schema dicts.** Wrong value →
   model silently gets no tool. Verify with the tokenize snippet in §2.
2. **Reset observation is dropped** for tool envs — the prompt must be in the dataset
   `messages`, not returned from `reset()`.
3. **`env_config.env_name` must equal the pool key (= call name).** If it's the config_name
   while the pool is keyed by a different call name, `_discover_tools_from_datasets`
   **auto-creates a duplicate pool** and per-row `task_id` routing misses → the rollout
   **hangs at prompt enqueue** (vLLM waits for prompts, data-prep waits for results, env actors
   idle). Symptom check: log says `Initialized 2 tool pools` for one logical env.
4. **`backend = "docker"` field on the config** is required for the pool to inject
   `SWERL_PODMAN_DOCKER_HOSTS` → without it, no `docker_host` on beaker, containers can't start.
5. **Dependency conflicts** (e.g. `pydantic` major) → you cannot import the env in the trainer
   process. Go container + HTTP; the env module must not import the conflicting package at all.
6. **Lazy-import optional deps.** `tools/tools.py` imports your env module at startup; importing
   a heavy/optional dep at module top breaks every install. Import inside `setup()` with `# noqa: PLC0415`.
7. **Local `.parquet`/`.jsonl` in `dataset_mixer_list`** now works in
   `_discover_tools_from_datasets` too (it previously only worked in `DatasetConfig` and threw
   `FileNotFoundError` in discovery — fixed). Keep both paths in mind if you add more loaders.
8. **Container networking** (style B): Beaker podman = **host network** (no bridge IP, publish is
   a no-op). Give each container a **unique free port** and reach it at **`127.0.0.1:<port>`**;
   keep the bridge IP as a local-docker fallback. (And `auto_remove=False` so you can read a
   dead container's logs.)
9. **Dev-box docker has no shared FS with the daemon sometimes** (e.g. this sandbox: daemon root
   `/media/8TBNVME`, no `/tmp` sharing) → bind mounts silently mount empty dirs. Validate by
   `docker run --rm -v <src>:/x alpine ls /x`; if empty, **bake data into the image** instead.
10. **`build_image_and_launch.sh` refuses a dirty tree** ("Uncommitted changes detected") →
    use `build_image_and_launch_dirty.sh`.
11. **One container per concurrent rollout.** `pool_size = num_unique_prompts_rollout *
    num_samples_per_prompt_rollout` (cap concurrency; don't over-provision heavyweight containers).
12. **Pin the external API in a throwaway venv** before writing code — don't guess signatures.

---

## 7. Copy-paste checklist for a new sandboxed env `myenv`

```
[ ] Decide style A (bash-exec, reuse DockerBackend) vs B (stateful server + HTTP client).
[ ] Throwaway venv: install the env's package, read its source, pin exact API.
[ ] Dependency conflict with pydantic>=2 / openenv? If yes -> style B, container quarantine.
[ ] open_instruct/environments/myenv_env.py:
      class MyEnv(RLEnvironment): config_name="myenv"; backend field via config; lazy import deps in setup()
      get_tool_definitions() -> [ { "type":"function","function":{"name":"<fn>", ...} } ]
      reset(task_id, docker_host=..., max_steps=...): start/reuse container, init episode, return (StepResult(""), tools)
      step(call): dispatch by call.name, detect done, compute reward; truncate long observations
      close()/shutdown(): tear down container
    @dataclass MyEnvConfig(BaseEnvConfig): tool_class=MyEnv; backend="docker"; image=...; <params>
[ ] Register in tools/tools.py TOOL_REGISTRY (import + entry).
[ ] Data converter -> rows: messages(system+user instruction), ground_truth(task_id),
      dataset="passthrough", tools=["<fn>"], env_config={env_configs:[{env_name:"<fn>",task_id}],max_steps}
[ ] Verify the tool schema injects (rlvr_tokenize_v1 snippet, §2).
[ ] tests/test_environments.py: prompt builder, reward mapping, registry, get_tool_definitions, reset-needs-id.
[ ] local_rl_2gpu.sh: Qwen3-0.6B, local parquet, data-baked image, run end-to-end.
[ ] Beaker script: copy the podman recipe (§3), set --tools myenv --tool_call_names <fn> --tool_configs {...}.
[ ] Data/image private if license-restricted.
[ ] make style && uv run pytest tests/test_environments.py -k MyEnv
```

---

## 8. Key file/line index (jump straight here)

| Concern | File |
|---|---|
| Env ABC, TextRLEnvironment, BaseEnvConfig | `open_instruct/environments/base.py` |
| Registry | `open_instruct/environments/tools/tools.py` (`TOOL_REGISTRY`, ~652) |
| Pool, podman host rotation, `backend=="docker"` gate | `open_instruct/environments/pool.py` (~74) |
| Docker/Apptainer backends (style A) | `open_instruct/environments/backends.py` |
| Tool discovery, pool creation, tool_definitions | `open_instruct/grpo_fast.py` (`_discover_tools_from_datasets` ~3050, `create_tool_pools` ~1962, `initialize_tools_and_envs` ~3085) |
| Dataset tools validation | `open_instruct/grpo_fast.py:_validate_and_log_dataset_tools` (~1876) |
| `tools` column semantics (names vs schemas) | `open_instruct/dataset_transformation.py` (`rlvr_tokenize_v1` ~1518/1539, `_normalize_tools_for_chat_template` ~976, local-file loader ~1793) |
| Rollout: reset-before-gen, allowed_tools, reward collect, one-text-env | `open_instruct/vllm_utils.py` (`_acquire_and_reset_pools` ~1042, ~1090, ~1231, ~1285) |
| passthrough verifier, reward selection | `open_instruct/ground_truth_utils.py` (~1037, ~1253) |
| Podman services + `SWERL_PODMAN_DOCKER_HOSTS` | `scripts/docker/docker_login.sh` |
| Ray + node setup | `configs/beaker_configs/ray_node_setup.sh` |
| Dirty-tree launcher | `scripts/train/build_image_and_launch_dirty.sh` |
| Podman recipe reference | `scripts/general_agent/terminal/rl/qwen35_4b_base_tmax_10k.sh` |
| Style-B worked example | `open_instruct/environments/appworld_env.py` + `scripts/general_agent/appworld/` |
