# Sandbox execution: Podman (current) vs. Modal

This doc summarizes how open-instruct runs agent/code sandboxes today, what a managed
service like [Modal](https://modal.com) would change, whether it would improve training
efficiency, and roughly what it would cost. It is a decision aid, not a migration plan.

## Background: what Podman does here

Podman runs **containers** — isolated mini-environments that bundle an OS image,
dependencies, and code. It is command-compatible with Docker (`podman run ...` works like
`docker run ...`) but is **rootless** (no privileged background daemon), which is what lets
it run *inside* another container.

During RL (GRPO) training, the model emits actions (bash/code) that must execute in a
throwaway sandbox. Those sandboxes are containers. The training job is itself already a
container on Beaker, where a normal Docker daemon isn't available — so Podman runs the
nested sandbox containers.

How it's wired:

- **Installed into the training image** (`Dockerfile`): Podman 5.6.2 + crun compiled from
  source, configured for rootless/nested use, and `docker` is symlinked to `podman`.
- **Runs as a Docker-API server**: at job startup `scripts/docker/docker_login.sh` starts
  one or more `podman system service` processes, each exposing a Unix socket
  (e.g. `/tmp/podman.sock`) that speaks the Docker API.
- **Consumed via the Docker SDK**: `open_instruct/environments/backends.py` (`DockerBackend`)
  connects the Python Docker SDK to that socket — it thinks it's talking to Docker.
- **Sharded for throughput**: `SWERL_PODMAN_SERVICE_COUNT` (4–8) independent daemons, each
  with its own socket/storage/locks (`PODMAN_NUM_LOCKS=65536`); a Ray actor pool
  (`open_instruct/environments/pool.py`) load-balances sandboxes across them round-robin
  via `SWERL_PODMAN_DOCKER_HOSTS`.
- **Requires Beaker nesting perms**: `BEAKER_ALLOW_SUBCONTAINERS=1`,
  `BEAKER_SKIP_DOCKER_SOCKET=1`. Without them Podman fails with
  `cannot clone: Operation not permitted`.

Podman is used **only to run sandbox containers** — there is no `podman build`/`push`/`login`
in the repo. Marginal cost is ~$0 because sandboxes run on spare CPU/RAM of GPU nodes already
rented. (`scripts/docker/start_dind.sh` is an alternate Docker-in-Docker path behind the same
abstraction.)

## How Modal differs

Both run the sandbox in a container. The difference is **where the container lives and who
operates it**.

| | Podman (current) | Modal |
|---|---|---|
| Location | Container on the same Beaker GPU node as training | Container in Modal's cloud, off-node |
| Operator | You (compile, start, shard, set locks, run janitors) | Modal (fully managed, serverless) |
| Scaling | You shard across N daemons | Modal autoscales |
| Cost | ~$0 marginal (uses spare node CPU/RAM) | Paid per container-second |
| Latency | Localhost Unix socket → microseconds/command | Network RPC → tens of ms/command + rate limits |
| Requirements | Beaker subcontainer perms, rootless kernel features | Outbound internet egress + Modal API tokens |
| Idle billing | Free (your node) | **You pay wall-clock while the container is alive, busy or idle** |

The trade is **self-hosted-on-node vs. managed-remote-service**.

## What the codebase change would look like

The `SandboxBackend` ABC (`backends.py:130`) is the whole interface — six methods: `start`,
`run_command`, `write_file`, `read_file`, `put_archive`, `close`. `ApptainerBackend` already
proves a second backend fits behind it. So Modal is fundamentally **"add a third backend,"**
not a rewrite.

1. **New `ModalBackend(SandboxBackend)`** in `backends.py` — `start()` → `modal.Sandbox.create()`,
   `run_command()` → `sandbox.exec(...)`, file I/O via Modal file handles or the same
   tar-over-exec trick `ApptainerBackend` uses, `close()` → `sandbox.terminate()`.
2. **Register it** in `create_backend()` (`backends.py:783`) — one `if` branch.
3. **Sharding/pool layer largely disappears** for Modal: the round-robin over
   `SWERL_PODMAN_DOCKER_HOSTS`, per-host cooldown/health tracking, and the
   `_FileSlotSemaphore` start/exec caps (`backends.py:175-176`) exist to protect a *local*
   daemon. Modal autoscales; the throttle becomes Modal's rate limits / account concurrency.
4. **Dockerfile gets much simpler** — drop the Podman/crun build, the `docker`→`podman`
   symlink, the containers.conf/policy.json/subuid/subgid setup, the registry-mirror warmer.
   Just `pip install modal`.
5. **Startup glue changes** — `docker_login.sh` / `start_dind.sh` are replaced by credential
   setup (`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` as Beaker secrets) + network egress.
6. **Beaker launch flags change** — `--mount_docker_socket`, `BEAKER_ALLOW_SUBCONTAINERS=1`,
   `BEAKER_SKIP_DOCKER_SOCKET=1` become unnecessary; **outbound network becomes a hard
   requirement** (some secured clusters disallow it — the main feasibility question).
7. **Error handling rewritten** — the Docker-specific OOM/409/NotFound retry logic
   (`backends.py:281-319`) becomes Modal-specific exception + network-retry handling.

## Would it improve training efficiency?

Not automatically. It depends on the current bottleneck.

**Per-call sandbox latency is already hidden two ways**, so Modal's added network latency is
mostly masked:

- **Across rollouts**: vLLM runs as one `AsyncLLMEngine` with hundreds of `process_request`
  coroutines on one event loop (`vllm_utils.py:537`). When rollout A awaits its sandbox
  (`vllm_utils.py:1280`), the GPU keeps generating tokens for rollouts B, C, D…
- **Across steps**: generation runs `async_steps` (default 8) ahead of the trainer
  (`grpo_fast.py:3289`), so sandbox time on future batches overlaps the current update.

**The governing metric is `time/trainer_idle_waiting_for_inference`** (see
`docs/algorithms/monitoring_and_debugging_runs.md:295`). The docs call its spiking — generation
being the bottleneck — "the normal state for agentic RL." When the trainer starves, the lever
is **aggregate generation throughput**, not per-call latency.

Modal **could help** if:
- Sandboxes are heavy (compiling, test suites) and are starving vLLM/Ray of CPU/RAM on the
  GPU node — offloading frees those resources for generation. *(Strongest argument.)*
- You've hit a local-daemon concurrency ceiling you can't shard past (the existence of the
  4–8 daemon sharding + raised lock limits suggests you're near it).

Modal **could hurt** if:
- Tail latency / stragglers: a batch finishes with its slowest rollout; network RPC has a fat
  tail (jitter, rate limits, cold starts) that localhost sockets don't, and sequential
  per-turn tool calls compound it.
- You're token-generation-bound or the node has CPU/RAM headroom — then Modal adds latency
  risk and cost for no gain.

**Before switching, measure**: (1) is `time/trainer_idle_waiting_for_inference` spiking? If not,
not generation-bound — Modal won't help. (2) If it is, is the node CPU/RAM-contended (sandboxes
starving vLLM) vs. token-bound? Only the former is fixed by offloading. (3) Turn on
`SWERL_SANDBOX_TIMING_LOGS` (`backends.py:177`) to see how much rollout wall-clock is sandbox
vs. generation.

## What it would cost

In this codebase **one sandbox lives for one rollout, and total rollouts over a run =
`total_episodes`**. So:

> Total sandbox cost ≈ `total_episodes` × `avg_container_lifetime_sec` × per-second rate

**Modal rate (Sandbox tier, CPU-only, verified 2026-06-18 at modal.com/pricing):**
$0.00003942 / core·sec + $0.00000672 / GiB·sec (a "core" = 1 physical core = 2 vCPU). Sandboxes
are billed ~3× the Function tier. With `mem_limit="4g"` and 1 core: **$0.0000663/sec ≈ $0.239/hr**.

**The dominant cost driver is idle billing.** The container is created at episode reset and
stays alive for the entire multi-turn rollout — including all the time the model spends
*generating* the next action. The docs note "a rollout might be minutes of sandbox execution,"
and the container is alive-but-idle for most of it. On Podman that idle time is free; on Modal
you pay wall-clock for every second it exists. So `avg_container_lifetime` ≈ full rollout
wall-clock, which is the biggest uncertainty — ranged below.

Cost per run (1-core / 4-GiB profile, $0.0000663/sec):

| Run | total_episodes | avg rollout life | sandbox-hours | Modal cost |
|---|---|---|---|---|
| Debug | ~400 | 1 min | ~7 | ~$0.50 |
| **Typical tmax run** | 128,000 | 3 min | ~6,400 | **~$1,500** |
| same, 5 min/rollout | 128,000 | 5 min | ~10,700 | ~$2,500 |
| Large instruct RL | ~1,000,000 | 3 min | ~50,000 | ~$12,000 |
| Frontier (32B think RL) | 10,000,000 | 3 min | ~500,000 | ~$120,000 |

So a **typical sandbox RL run lands ~$1,500–$2,500**. Halve it for ~90s rollouts or a
0.5-core/2-GiB profile; double it for ~10-min rollouts.

Caveats that move the number:
- **Team plan required** ($250/mo base): production `pool_size` is 128–1024 concurrent
  sandboxes; Modal's free Starter plan caps at 100 containers (Team allows 1000).
- **Region pinning** adds 1.5–1.75× if you must pin a region.
- **The 3× Sandbox-vs-Function premium** is baked into the rates above; restructuring as
  Functions could cut compute to a third but Sandboxes are the right primitive for arbitrary
  agent shell commands.

## Bottom line

- Code change is small and localized (one backend class + one factory line); the bigger shifts
  are operational (delete the self-hosting infra) and a new hard requirement: **the job must
  have internet egress and Modal credentials** — which the current air-gapped-friendly design
  avoids. Verify your Beaker cluster allows egress before anything else.
- Efficiency gain is **conditional**, not automatic: worth it mainly if sandboxes are starving
  the GPU node of CPU/RAM, or you've hit a local concurrency ceiling.
- Cost is **purely additive** vs. Podman's ~$0 marginal cost: ~$1.5–2.5k for a typical run,
  ~$12k for a large one. The idle-during-generation billing makes Modal a structurally poor fit
  for this specific workload, where containers sit idle waiting on the LLM.
