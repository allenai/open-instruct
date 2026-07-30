# Sandbox management

During RL (GRPO) training, the model emits actions (bash/code) that must execute in a
throwaway sandbox. This doc describes the sandbox backends open-instruct ships, how each
is wired, and the efficiency/cost trade-offs between running sandboxes on-node (Podman)
and off-node (Modal, OpenSandbox).

## The backend abstraction

`SandboxBackend` (`open_instruct/environments/backends.py`) is the whole interface — six
methods: `start`, `run_command`, `write_file`, `read_file`, `put_archive`, `close`.
`create_backend()` is the factory; environments select a backend with
`"backend": "<type>"` in `--tool_configs`. Four implementations exist:

| Backend | Where sandboxes run | Use when |
|---|---|---|
| `docker` (default) | Containers on the training node, via Podman or Docker | Beaker GPU jobs; ~$0 marginal cost |
| `apptainer` | Apptainer instances on the training node | HPC/Slurm hosts without a container daemon |
| `modal` | Containers in [Modal](https://modal.com)'s cloud | Off-node execution, managed by a vendor |
| `opensandbox` | Pods on a self-hosted [OpenSandbox](https://open-sandbox.ai) service (e.g. GKE Autopilot) | Off-node execution, operated by you |

## Podman (default): on-node sandboxes

Podman runs **containers** — isolated mini-environments that bundle an OS image,
dependencies, and code. It is command-compatible with Docker (`podman run ...` works like
`docker run ...`) but is **rootless** (no privileged background daemon), which is what lets
it run *inside* another container. The training job is itself already a container on
Beaker, where a normal Docker daemon isn't available — so Podman runs the nested sandbox
containers.

How it's wired:

- **Installed into the training image** (`Dockerfile`): Podman 5.6.2 + crun compiled from
  source, configured for rootless/nested use, and `docker` is symlinked to `podman`.
- **Runs as a Docker-API server**: at job startup `scripts/docker/docker_login.sh` starts
  one or more `podman system service` processes, each exposing a Unix socket
  (e.g. `/tmp/podman.sock`) that speaks the Docker API.
- **Consumed via the Docker SDK**: `DockerBackend` connects the Python Docker SDK to that
  socket — it thinks it's talking to Docker.
- **Sharded for throughput**: `SWERL_PODMAN_SERVICE_COUNT` (4–8) independent daemons, each
  with its own socket/storage/locks (`PODMAN_NUM_LOCKS=65536`); a Ray actor pool
  (`open_instruct/environments/pool.py`) load-balances sandboxes across them round-robin
  via `SWERL_PODMAN_DOCKER_HOSTS`. The `_FileSlotSemaphore` start/exec caps in
  `backends.py` protect the local daemons from stampedes.
- **Requires Beaker nesting perms**: `BEAKER_ALLOW_SUBCONTAINERS=1`,
  `BEAKER_SKIP_DOCKER_SOCKET=1`. Without them Podman fails with
  `cannot clone: Operation not permitted`.

Podman is used **only to run sandbox containers** — there is no `podman build`/`push`/`login`
in the repo. Marginal cost is ~$0 because sandboxes run on spare CPU/RAM of GPU nodes already
rented. (`scripts/docker/start_dind.sh` is an alternate Docker-in-Docker path behind the same
abstraction.)

## Modal: managed off-node sandboxes

Both run the sandbox in a container. The difference is **where the container lives and who
operates it**.

| | Podman (default) | Modal |
|---|---|---|
| Location | Container on the same Beaker GPU node as training | Container in Modal's cloud, off-node |
| Operator | You (compile, start, shard, set locks, run janitors) | Modal (fully managed, serverless) |
| Scaling | You shard across N daemons | Modal autoscales |
| Cost | ~$0 marginal (uses spare node CPU/RAM) | Paid per container-second |
| Latency | Localhost Unix socket → microseconds/command | Network RPC → tens of ms/command + rate limits |
| Requirements | Beaker subcontainer perms, rootless kernel features | Outbound internet egress + Modal API tokens |
| Idle billing | Free (your node) | **You pay wall-clock while the container is alive, busy or idle** |
| Image fidelity | Runs your image byte-for-byte | **Rebuilds the image before running it** — build requires Python + pip inside the image, else a standalone Python is injected |

The trade is **self-hosted-on-node vs. managed-remote-service**.

The image-fidelity row is subtle but bit us in practice: Modal's `Image.from_registry`
transforms the image rather than running it as built, so image-dependent behavior (PATH
resolution, tools under `/usr/local`, anything a task's `setup.sh` baked in) can silently
differ from Podman. Concretely, the injected standalone Python shadowed the tmax images'
own `/usr/bin/python3`, so `python3 -m pytest` verification lost its packages and every
rollout scored 0. `ModalBackend` repairs the `python3` shadowing after a fallback start,
but any new image-dependent oddity on Modal should be checked against this row first.

### How the Modal backend is wired

- **`ModalBackend`** (`backends.py`) drives `modal.Sandbox`: `start()` →
  `modal.Sandbox.create()` (with an `add_python` fallback for images without Python/pip,
  followed by the python3-shadowing repair), `run_command()` → `sandbox.exec(...)`, file
  I/O piped over exec, `close()` → `sandbox.terminate()`. Selected with
  `"backend": "modal"` in `--tool_configs`.
- **Config**: `SWERL_MODAL_APP_NAME`, `SWERL_MODAL_ENVIRONMENT`, `SWERL_MODAL_CPU`, and
  `SWERL_MODAL_SANDBOX_LIFETIME_S` env vars; credentials via `MODAL_TOKEN_ID` /
  `MODAL_TOKEN_SECRET` (Beaker secrets).
- **The Podman sharding/pool layer doesn't apply**: the round-robin over
  `SWERL_PODMAN_DOCKER_HOSTS`, per-host cooldown/health tracking, and the file-slot
  semaphores exist to protect a *local* daemon. Modal autoscales; the throttle is Modal's
  rate limits / account concurrency.
- **Leak protections** (a leaked sandbox bills wall-clock until reaped): every sandbox is
  created with a hard lifetime cap (`SWERL_MODAL_SANDBOX_LIFETIME_S`, default 1h),
  `close()` retries a failed terminate and logs loudly, an `atexit` reaper terminates
  live sandboxes on abrupt process exit, and
  `scripts/modal/cleanup_modal_sandboxes.sh` is the end-of-job janitor for sandboxes a
  killed job couldn't clean up. These fixes took the first 1,280-episode toy run from
  $1,219 to ~$65 (see the cost section).
- **Feasibility gate**: `scripts/modal/check_modal_egress.sh` verifies the training
  cluster can reach api.modal.com and drive the full sandbox lifecycle. **Outbound
  network is a hard requirement** — some secured clusters disallow it.
- **Launch example**:
  `scripts/general_agent/terminal/rl/qwen35_4b_base_tmax_10k_modal_2node_toy.sh` (no
  `--mount_docker_socket`, no `BEAKER_ALLOW_SUBCONTAINERS` / `BEAKER_SKIP_DOCKER_SOCKET`).

## OpenSandbox: self-hosted off-node sandboxes on GKE Autopilot

[OpenSandbox](https://open-sandbox.ai) is an open-source (Alibaba, Apache-2.0) sandbox
service you host yourself: a control plane that turns each sandbox into a Kubernetes pod,
plus a Python SDK that drives it over HTTP. `OpenSandboxBackend` (`backends.py`) sits
behind the same `SandboxBackend` ABC; select it with `"backend": "opensandbox"` in
`--tool_configs`.

It keeps Modal's execution model — sandboxes off-node, behind a network API, no nested
containers or podman services on the GPU node — but changes who operates the fleet:

| | Modal | OpenSandbox on GKE Autopilot |
|---|---|---|
| Operator | Modal (managed) | You (deploy/upgrade the service, own the cluster) |
| Image fidelity | **Rebuilds the image** (the `add_python` / python3-shadowing problem above) | Kubernetes pulls the OCI image **as-is** — no rebuild, no fidelity gap |
| Rate (1 CPU + 4 GiB) | ~$0.239/hr (Sandbox tier premium) | ~$0.068/hr Autopilot list price (less with Spot pods) |
| Cold start | Seconds (warm pool) | Pod start is seconds if the cluster has headroom; **minutes if Autopilot must provision a node** — the backend defaults `ready_timeout` to 180s for this |
| Concurrency cap | Plan-based (100 free / 1000 Team) | Whatever your cluster/quota allows |
| Egress requirement | api.modal.com | Your GKE endpoint (`SWERL_OPENSANDBOX_DOMAIN`) |

The idle-during-generation billing structure is unchanged — a sandbox pod is alive (and
billing at Autopilot rates) for the whole rollout — but at roughly 3.5× lower unit cost,
and the image-fidelity failure mode that silently zeroed rewards on Modal does not exist.

How it's wired:

- **`OpenSandboxBackend`** uses the SDK's sync API (`SandboxSync`). Config comes from
  `SWERL_OPENSANDBOX_DOMAIN` / `SWERL_OPENSANDBOX_PROTOCOL` / `OPEN_SANDBOX_API_KEY`
  (Beaker secret), with `SWERL_OPENSANDBOX_CPU`, `SWERL_OPENSANDBOX_LIFETIME_S`,
  `SWERL_OPENSANDBOX_READY_TIMEOUT_S`, and `SWERL_OPENSANDBOX_APP_NAME` tuning knobs.
- **Creates are throttled per node** (`SWERL_OPENSANDBOX_START_CONCURRENCY`, default 64)
  with the same file-slot semaphore `DockerBackend` uses: a large pool (hundreds of env
  actors) otherwise stampedes the control plane at t=0 with concurrent creates that force
  mass Autopilot node provisioning — observed to collapse a pool-768 run into 504s and
  failed adoptions while a throttled ramp is absorbed fine. Running sandboxes hold no
  slot, so steady-state concurrency still reaches the full pool size. `ready_timeout`
  defaults to 600s so creates and 504-adoptions survive node-provisioning waves.
- **Command output is an SSE stream** whose HTTP read timeout is the connection's
  `request_timeout`; the backend sets it above the per-command timeout so long-quiet
  commands don't sever the stream. Exec output is text-only, so binary file reads go
  through the SDK's filesystem API instead.
- **The same leak protections as Modal**: hard sandbox lifetime at create time
  (`SWERL_OPENSANDBOX_LIFETIME_S`, default 1h), loud retry on failed kill, an atexit
  reaper, and an end-of-job janitor
  (`scripts/opensandbox/cleanup_opensandbox_sandboxes.sh`) keyed on the
  `open_instruct_app` metadata tag.
- **Feasibility gate**: `scripts/opensandbox/check_opensandbox_egress.sh` verifies
  endpoint reachability and the full create→exec→kill lifecycle from the training
  cluster, mirroring the Modal egress check.
- **Launch example**:
  `scripts/general_agent/terminal/rl/qwen35_4b_base_tmax_10k_opensandbox_2node_toy.sh`.

What you give up vs. Modal is the ops: the GKE cluster, the OpenSandbox deployment, API
key issuance, and capacity planning are yours, and there is no vendor autoscaler-with-SLA
behind the endpoint. Deployment of the OpenSandbox server itself is outside this repo.

**Cold-start 504 gotcha** (observed 2026-07-23 against the AI2 deployment): if a create
outlasts the GCLB ingress's upstream timeout (default 30s; cold image pull or Autopilot
node provisioning can take minutes), the client gets HTTP 504 — **but the sandbox still
comes up server-side**, sometimes more than once per timed-out request. Two layers of
defense exist:

- **Server side (primary)**: raise the load balancer's backend timeout well above
  worst-case pod startup (`timeoutSec` in a `BackendConfig` for GKE Ingress, or a
  `GCPBackendPolicy` for the Gateway API). After this fix, a 101s cold create completed
  normally on the AI2 deployment.
- **Client side (insurance)**: `OpenSandboxBackend.start()` tags every create with a
  unique `open_instruct_create_id` and, on a gateway-timeout error, polls the management
  API for a sandbox carrying that tag, adopts it via `SandboxSync.connect`, and kills
  any duplicates — so a 504 becomes a slower start instead of an error plus a leaked
  pod. If nothing appears within `ready_timeout`, the original error is re-raised and
  the janitor reclaims any late arrival by app tag.

Warm-path numbers from live verification: create ~5–7s, ~1.05s per exec — on par with
Modal.

## Do off-node sandboxes improve training efficiency?

Not automatically. It depends on the current bottleneck. (The analysis below is written
for Modal but applies equally to OpenSandbox — both replace a localhost socket with a
network RPC.)

**Per-call sandbox latency is already hidden two ways**, so the added network latency is
mostly masked:

- **Across rollouts**: vLLM runs as one `AsyncLLMEngine` with hundreds of `process_request`
  coroutines on one event loop (`vllm_utils.py`). When rollout A awaits its sandbox, the
  GPU keeps generating tokens for rollouts B, C, D…
- **Across steps**: generation runs `async_steps` (default 8) ahead of the trainer
  (`grpo_fast.py`), so sandbox time on future batches overlaps the current update.

**The governing metric is `time/trainer_idle_waiting_for_inference`** (see
`docs/algorithms/monitoring_and_debugging_runs.md`). The docs call its spiking — generation
being the bottleneck — "the normal state for agentic RL." When the trainer starves, the lever
is **aggregate generation throughput**, not per-call latency.

Off-node sandboxes **could help** if:
- Sandboxes are heavy (compiling, test suites) and are starving vLLM/Ray of CPU/RAM on the
  GPU node — offloading frees those resources for generation. *(Strongest argument.)*
- You've hit a local-daemon concurrency ceiling you can't shard past (the existence of the
  4–8 daemon sharding + raised lock limits suggests you're near it).

Off-node sandboxes **could hurt** if:
- Tail latency / stragglers: a batch finishes with its slowest rollout; network RPC has a fat
  tail (jitter, rate limits, cold starts) that localhost sockets don't, and sequential
  per-turn tool calls compound it.
- You're token-generation-bound or the node has CPU/RAM headroom — then off-node execution
  adds latency risk and cost for no gain.

**Before switching, measure**: (1) is `time/trainer_idle_waiting_for_inference` spiking? If not,
not generation-bound — offloading won't help. (2) If it is, is the node CPU/RAM-contended
(sandboxes starving vLLM) vs. token-bound? Only the former is fixed by offloading. (3) Turn on
`SWERL_SANDBOX_TIMING_LOGS` (`backends.py`) to see how much rollout wall-clock is sandbox
vs. generation.

## What Modal costs

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

**Measured:** the first 1,280-episode toy run (2026-07-17) billed **$1,219** vs ~$15 from this
model — leaked/idle sandboxes, which are free on Podman. With leak fixes (episode-end close,
1h lifetime cap, end-of-job janitor) the same run billed **~$65** (2026-07-18). The residual
~4× over the model is the active-sampling rollout multiplier plus Modal's sandbox rate premium,
so scale the table above ~4×: a typical 128k-episode run lands **~$6–7k**, not ~$1.5k.

Caveats that move the number:
- **Team plan required** ($250/mo base): production `pool_size` is 128–1024 concurrent
  sandboxes; Modal's free Starter plan caps at 100 containers (Team allows 1000).
- **Region pinning** adds 1.5–1.75× if you must pin a region.
- **The 3× Sandbox-vs-Function premium** is baked into the rates above; restructuring as
  Functions could cut compute to a third but Sandboxes are the right primitive for arbitrary
  agent shell commands.

For OpenSandbox, apply the same wall-clock model at GKE Autopilot rates (~$0.068/hr for
1 vCPU + 4 GiB, roughly 3.5× cheaper than Modal's sandbox tier, before Spot discounts),
plus the fixed cost of keeping the service/cluster running.

## Bottom line

- All backends sit behind the same six-method `SandboxBackend` interface, so switching is
  a `--tool_configs` change plus credentials/egress — no training-code changes.
- Off-node backends trade the self-hosting infra (nested Podman, subcontainer perms,
  daemon sharding) for a new hard requirement: **the job must have internet egress and
  service credentials** — which the air-gapped-friendly Podman design avoids. Run the
  relevant egress-check script on the target cluster before anything else.
- Efficiency gain is **conditional**, not automatic: worth it mainly if sandboxes are
  starving the GPU node of CPU/RAM, or you've hit a local concurrency ceiling.
- Cost is **purely additive** vs. Podman's ~$0 marginal cost, and the
  idle-during-generation billing makes per-second-billed sandboxes a structurally poor
  fit for this workload, where containers sit idle waiting on the LLM. Modal lands
  ~$6–7k for a typical 128k-episode run (measured-calibrated); OpenSandbox cuts the unit
  rate ~3.5× and removes the image-fidelity risk, at the price of operating the service
  yourself.
