# Podman in Terminal-RL jobs: how we configure and use it

How our agentic-RL training jobs stand up Podman inside a Beaker task and drive it to run
per-rollout sandbox containers — written as a handoff for anyone building a managed
container-execution service (e.g. a Redis-coordinated Podman fleet) that would replace the
in-job Podman we run today.

State as of 2026-08-07. Companion doc: [Terminal-RL container registry mirror](registry_mirror.md),
which covers the image-pull side. Existing arg-level reference:
[tmax RL args](tmax_rl_args_reference.md).

---

## 1. What the workload actually needs (the contract)

Strip away the implementation and this is all the training loop asks for. Anything that provides
these operations, at this concurrency, with this isolation, can replace Podman:

| Need | Detail |
|---|---|
| **Create a container from an arbitrary Docker Hub image** | ~14.5k distinct per-task images, `hamishi740/swerl-tmax-v3:<content-hash>`, ~306 MB each. Image is chosen **per rollout**, not per job. |
| **Long-lived idle container** | Created with `command="sleep infinity"`; it must stay up across many exec calls (one rollout = up to 64 agent turns). |
| **Exec a bash command, get (exit_code, stdout, stderr)** | Separate streams (`demux=True`). This is the agent's only tool. |
| **Push/pull files as tar streams** | `put_archive` / `get_archive` — used to inject task data and tests, and read results back. |
| **Destroy the container** | Reliably, including after the client process dies. |
| **Per-container memory limit** | `mem_limit` + `memswap_limit`, enforced, with OOM-kill *detectable* after the fact. |
| **Concurrency** | ~512 concurrent live containers per job (see §6), with sub-second exec latency. |
| **Lifecycle churn** | ~0.7 container creations/sec sustained per 4-node job, each with a ~300 MB image pull ~66–89% of the time. |

Everything below is *how we currently satisfy that*, and which parts are incidental vs. load-bearing.

---

## 2. Where Podman comes from: the training image

Built into the open-instruct training image (`Dockerfile`, ~lines 60–135). Not installed at runtime.

- **Podman 5.6.2**, built from source (`make BUILDTAGS="selinux seccomp" PREFIX=/usr`).
- **crun 1.14.3**, built from source. crun (not runc) is the configured OCI runtime.
- apt deps: `conmon`, `netavark`, `passt`, `uidmap`, `iptables`, `golang-github-containers-common`,
  plus the build toolchain.
- `/usr/local/bin/docker` is a **symlink to podman**. Any code or script calling `docker` gets
  Podman. (DinD paths that need the real Docker CLI call `/usr/bin/docker` explicitly.)
- subuid/subgid ranges for rootless: `root:10000:11165536` in `/etc/subuid` and `/etc/subgid`.
- Config files copied to `/etc/containers/` (see §3).

`scripts/docker/setup_podman.sh` reproduces this exact setup on an already-built machine/image
(same versions, same config files) — useful as an executable spec of the install.

### 2.1 `/etc/containers/containers.conf` — the isolation posture

This is the part most worth reading closely, because it is deliberately **permissive** and any
replacement has to make a conscious decision about each line:

```toml
[containers]
netns="host"              # sandbox shares the NODE's network namespace
userns="auto:size=1024"   # auto-allocated user namespace, 1024 uids each
ipcns="host"
utsns="host"
cgroupns="host"
cgroups="disabled"        # no per-container cgroup created
keyring=false
log_driver = "k8s-file"
volumes = ["/proc:/proc"]
default_sysctls = []

[engine]
cgroup_manager = "cgroupfs"
events_logger = "file"
num_locks = 8192          # overridden to 65536 at runtime, see §4
runtime = "crun"
```

Implications to carry over (or deliberately change):

- **`netns="host"` means the agent's bash has the node's network.** It can reach the vLLM engines,
  the registry mirror, weka mounts' network paths, and anything else on the cluster network. This
  was chosen for pull/throughput simplicity, not because the workload needs it. A managed service
  that gives each sandbox its own netns would be *safer*; we would need to verify nothing in the
  task images depends on host networking (some tasks start local services and talk to them on
  `localhost`, which still works inside a private netns).
- **`cgroups="disabled"` + `cgroup_manager="cgroupfs"`** — we do not get per-container cgroup
  accounting. `mem_limit` is still passed on create and OOM kills are still observed via
  `State.OOMKilled`, but there is no cgroup-level metric to scrape. If your service can give us
  real per-sandbox cgroup limits and usage, that is an upgrade, not a regression.
- **`userns="auto:size=1024"`** is what the `root:10000:11165536` subuid range feeds. 11165536 /
  1024 ≈ 10,900 concurrent user namespaces per node before exhaustion.

Also installed: `policy.json` set to `insecureAcceptAnything` (no signature verification), and
`registries.conf.d/10-unqualified-search-registries.conf` = `["docker.io"]`.

---

## 3. Beaker-side requirements

The training task runs Podman **inside** the Beaker task container — this is nested containers, not
Docker-in-Docker and not a mounted host socket.

| Setting | Why |
|---|---|
| `--env BEAKER_ALLOW_SUBCONTAINERS=1` | Grants the task permission to spawn sub-containers. Without it Podman fails at `podman system service` with **`cannot clone: Operation not permitted`**. |
| `--env BEAKER_SKIP_DOCKER_SOCKET=1` | Tells Beaker not to inject its own Docker socket; we manage our own. |
| `--pure_docker_mode` (mason.py) | Runs our image directly rather than Beaker's wrapper. |
| *(not used)* `--mount_docker_socket` | mason.py supports it for real DinD, but our Terminal-RL scripts do **not** use it. Older docs that call it "required" are stale. |

We do **not** require privileged mode or a host Docker daemon.

---

## 4. Runtime bring-up: `scripts/docker/docker_login.sh`

Sourced as the literal first command of every Terminal-RL training task, before Ray starts:

```
-- source scripts/docker/docker_login.sh && source configs/beaker_configs/ray_node_setup.sh \
   && python open_instruct/grpo_fast.py ...
```

It does five things:

**(a) Configures the registry mirror.** Runs `/usr/local/bin/setup_dockerio_mirror`, which writes
`/etc/containers/registries.conf` from `$MIRROR_URL`, forces `storage.driver = "overlay"` in
`/etc/containers/storage.conf`, and sets `[engine] num_locks` from `$PODMAN_NUM_LOCKS`
(default 8192; we run **65536** — the default is exhausted at our container counts). Details in
[registry_mirror.md](registry_mirror.md).

**(b) Starts N sharded Podman daemons.** `SWERL_PODMAN_SERVICE_COUNT` (we run 4–8) independent
`podman system service --time=0` processes, each with its **own storage**:

| Shard | Socket | graphroot | runroot | tmpdir |
|---|---|---|---|---|
| single-shard mode | `/tmp/podman.sock` | default | default | default |
| shard *i* (N>1) | `/tmp/podman-services/i/podman.sock` | `/var/lib/containers/storage/swerl-podman-shards/i` | `/run/containers/storage/swerl-podman-shards/i` | `/var/tmp/swerl-podman-shards/i` |

Then exports:
- `DOCKER_HOST=unix:///tmp/podman-services/0/podman.sock` (shard 0)
- `SWERL_PODMAN_DOCKER_HOSTS=` comma-separated list of all shard sockets — this is what the
  scheduler in §6 reads.

It waits up to ~5s per socket and dumps the service log if a socket never appears.

**Why sharded:** a single Podman daemon serialises on its bolt DB and lock pool and becomes the
bottleneck at our concurrency (symptoms: `database is locked`, `retrieving exec session`,
`timed out waiting for file`). Sharding gives N independent daemons + N independent lock pools.
**The cost:** each shard is a *separate image store*, so the same image is pulled once per shard per
node — this is the main driver of the pull volume in the registry-mirror doc (16 independent image
stores on a 4-node job).

**(c) Raises ulimits:** `ulimit -n 1048576`, `ulimit -u 1048576`.

**(d) Starts two janitor loops** (§7).

**(e) Writes Docker Hub credentials** to `~/.docker/config.json` from the `DOCKER_PAT` Beaker
secret. Note: **Podman reads `containers/auth.json`, not `~/.docker/config.json`**, when pulling via
its own socket — our eval path additionally runs `podman login` for this reason. A managed service
needs to get credentials into whichever store its Podman actually consults.

### Environment variables that configure this layer

| Var | Our value | Meaning |
|---|---|---|
| `SWERL_PODMAN_SERVICE_COUNT` | 4–8 | Number of Podman daemon shards per node |
| `SWERL_PODMAN_SERVICE_DIR` | `/tmp/podman-services` | Shard socket root |
| `SWERL_PODMAN_GRAPHROOT_BASE` | `/var/lib/containers/storage/swerl-podman-shards` | Per-shard image/container storage |
| `SWERL_PODMAN_RUNROOT_BASE` | `/run/containers/storage/swerl-podman-shards` | Per-shard runtime state |
| `SWERL_PODMAN_TMPDIR_BASE` | `/var/tmp/swerl-podman-shards` | Per-shard tmp |
| `PODMAN_NUM_LOCKS` | 65536 | Podman lock pool (default 8192 too small) |
| `CONTAINERS_STORAGE_CONF` | `/etc/containers/storage.conf` | Where the storage driver gets written |
| `SWERL_DOCKER_START_CONCURRENCY` | 64 | Node-wide cap on concurrent container starts |
| `SWERL_DOCKER_EXEC_CONCURRENCY` | 256 | Node-wide cap on concurrent execs |
| `SWERL_DOCKER_LOCK_DIR` | `/tmp/open_instruct_docker_locks` | Backing files for those caps |
| `SWERL_DOCKER_AUTO_REMOVE` | 1 | `auto_remove=True` on container create |
| `SWERL_DOCKER_JANITOR_*` | see §7 | Exited-container reaper |
| `SWERL_PODMAN_IMAGE_JANITOR_*` | see §7 | Image pruner |
| `SWERL_PODMAN_HOST_COOLDOWN_S` | 300 (+30s jitter) | How long a failing shard is benched |
| `SWERL_SANDBOX_TIMING_LOGS` | 1 | Emit per-phase lifecycle timings |

---

## 5. How the training code talks to Podman

**Entirely through the Python `docker` SDK (docker-py) against the Podman socket** — we never shell
out to the `podman` CLI from the training loop. Podman's Docker-compatible REST API is the
integration surface. `open_instruct/environments/backends.py::DockerBackend`.

```python
client = docker_sdk.DockerClient(base_url=docker_host, timeout=300)   # docker_host = unix://.../podman.sock
```

The complete set of API calls we make — **this is the porting checklist**:

| Phase | Call |
|---|---|
| start | `client.images.get(image)` → on `ImageNotFound` → `client.images.pull(image)` |
| start | `client.containers.create(image, command="sleep infinity", detach=True, auto_remove=<bool>, labels={"open_instruct": "swerl_sandbox"}, mem_limit=..., memswap_limit=...)` |
| start | `container.start()` |
| exec | `container.exec_run(["bash","-c", wrapped], demux=True)` |
| files | `container.put_archive(root, tar_bytes)` / `container.get_archive(path)` |
| health | `container.reload()`, `container.attrs["State"]["OOMKilled"]`, `container.logs()` |
| stop | `container.kill()`, falling back to `container.stop(timeout=3)` |
| janitor | `podman ps -aq --filter status=exited --filter label=open_instruct=swerl_sandbox`, `podman rm`, `podman image prune` (these two are CLI, not SDK) |

Every command is wrapped host-side before exec so a runaway command cannot hang the rollout:

```bash
timeout --signal=TERM --kill-after=10 <N> bash -c '<agent command>'
```

Output is truncated to 1 MB (`_MAX_OUTPUT_BYTES`). The agent's shell state (cwd + exported env) is
persisted between turns by a wrapper script inside the container that re-sources
`/tmp/.swerl_vanillux_env` and `/tmp/.swerl_vanillux_cwd` — i.e. **we fake a persistent shell over
stateless execs.** A service offering a genuinely persistent shell session would let us delete that.

### Per-rollout sequence

1. Pool leases an actor and a shard socket, calls `reset(task_id, docker_host=...)`.
2. Image resolved per task (from `env_config.image` or `image.txt` in the task data).
3. Old container `close()`d, new one created + started on the leased socket.
4. `mkdir -p /workspace /output /logs/verifier`.
5. Task data + tests injected via `put_archive`.
6. Up to 64 agent turns, each one `exec_run`.
7. Verifier script run inside the container (timeout ≥600s), reward extracted.
8. `close()` → `kill()`; `auto_remove` reaps it.

---

## 6. Concurrency and scheduling

- **`--pool_size 512`** → 512 Ray actors per tool, each owning at most one sandbox container at a
  time. So up to ~512 live containers per job, spread over the job's nodes.
- **`EnvironmentPool`** (`open_instruct/environments/pool.py`) leases a shard socket per reset,
  **least-inflight first with a round-robin cursor** among healthy shards.
- **Shard health:** if a reset fails with a connectivity-shaped error (`connection refused`,
  `read timed out`, `broken pipe`, `Error while fetching server API version`, …), that shard is
  marked unhealthy for **300s + up to 30s jitter** and the reset is retried on another shard. All
  shards exhausted → the reset raises.
- **Node-wide admission control:** two cross-process semaphores implemented as `flock` on files in
  `/tmp/open_instruct_docker_locks` — `docker-start` (64 slots) and `docker-exec` (256 slots). These
  exist purely to stop us from stampeding the local Podman daemons.
- **Reset retries:** 5 attempts (docker backend) with exponential backoff 1s→16s + jitter.

> **Important subtlety for anyone porting this.** `docker_host` today is a **unix socket path that
> exists identically on every node**. The pool balances across *shard indices*; which physical node
> a sandbox lands on is decided entirely by where Ray placed the actor, and the pool has no
> knowledge of it. Moving to a centrally managed fleet inverts this: `DOCKER_HOST` becomes a network
> endpoint, and sandbox placement decouples from Ray actor placement. That is a genuine improvement
> (real load balancing across the fleet) but it changes two assumptions in our code — that the
> socket is local and free to talk to, and that a shard string is meaningful on any node.

---

## 7. Resource management on the node

Sandbox containers are ~300 MB images churning at ~0.7 creations/sec; without active cleanup a node
fills its disk and Podman starts failing in confusing ways.

- **Container janitor** (`SWERL_DOCKER_JANITOR_ENABLED`, interval 60s, batch 20): lists exited
  containers by the `open_instruct=swerl_sandbox` label and removes them. Auto-enabled when
  `SWERL_DOCKER_AUTO_REMOVE=0`.
- **Image janitor** (`SWERL_PODMAN_IMAGE_JANITOR_ENABLED=1`, interval 60s,
  `until=10m`): `podman image prune -a --force --filter until=10m` per shard, logging `df -h` and
  `podman system df` before/after. **This is why our cache-miss rate is 66–89%** — we prune images
  faster than we revisit them, trading pull volume for disk headroom. A fleet with more disk per
  node, or content-addressed shared storage across shards, would cut our registry load
  substantially.
- We have taken down runs by filling `/tmp` on a node — Podman resets then hang rather than error,
  and the failure surfaces as a stalled data-preparation actor, not a clean exception.

---

## 8. Failure modes we handle (and would need handled)

These are all things the current code has explicit recovery paths for. Any replacement service will
meet the same conditions, so it is worth knowing which ones we absorb ourselves.

| Symptom | Cause | Our response |
|---|---|---|
| `cannot clone: Operation not permitted` at startup | `BEAKER_ALLOW_SUBCONTAINERS` not set | Fail fast with an explicit hint in the log |
| `database is locked`, `retrieving exec session`, `timed out waiting for file` | Podman daemon lock contention | Classified as *transient*: retry the same exec up to 5× with 0.5s→8s backoff |
| Exec returns 409 Conflict | Container not running (crash, OOM, external stop) | Probe `State.OOMKilled`; if OOM → raise `SandboxOOMError` and end the episode with reward 0 (retrying would just re-OOM). Otherwise recreate the container and retry once |
| Exec raises `NotFound` | Container vanished (janitor race, auto_remove) | Recreate + retry once |
| Socket-level connectivity errors | Podman shard wedged/dead | Bench that shard for 300s, retry the reset on another shard |
| Image pull fails / manifest corrupt | Registry mirror dead or serving bad data | Reset retried 5×; with `SWERL_RESET_FAILURE_ZERO_REWARD=1` the affected rollouts score 0 — a *fraction* of the batch, silently |
| Node disk full | Image churn outpacing the janitor | No graceful handling — resets hang. Detected by hand from Beaker logs |
| "too many locks" | `num_locks` default 8192 | Raised to 65536 |

The pattern worth noting: **almost everything degrades silently into lower reward rather than
failing the job.** For a managed service, the single most valuable property beyond correctness is
*telling us loudly when a sandbox couldn't be provided*, rather than letting it look like the model
got the answer wrong.

---

## 9. What a Redis-managed Podman fleet would need to provide

Mapped against the above, in rough priority order:

1. **The docker-py API surface in §5**, or an equivalent client we can swap `DockerBackend` for
   (it is one ~350-line class behind a 6-method `SandboxBackend` ABC — `start`, `run_command`,
   `write_file`, `read_file`, `put_archive`, `close` — so an alternate implementation is a
   contained change on our side; we already have a second implementation, `ApptainerBackend`, for
   Slurm/HPC).
2. **Arbitrary Docker Hub images pulled on demand**, ~14.5k distinct tags, ~306 MB each,
   at ~0.7 pulls/s sustained per job. A shared image cache across the fleet would be a large win
   over our per-shard duplication.
3. **~512 concurrent live containers per job**, multiple jobs at once, with exec round-trips fast
   enough not to dominate a 64-turn rollout. For calibration, our measured container-start cost
   today is 1.55s median without an image pull and ~4.2s median with one (p99 47s); exec cost we
   only log above a 1s threshold, so most execs are faster than that.
4. **Per-container memory limits with detectable OOM kills** — we branch on this to end episodes
   cleanly.
5. **Explicit, loud failures** when a sandbox can't be created, distinguishable from "the agent
   failed the task". Today a broken backend looks like a bad model.
6. **A decision on the isolation posture in §2.1** — host netns/ipc/uts, disabled cgroups. We are
   happy to tighten these; we just need to know what we're getting, since some task images assume
   they can bind ports and write anywhere.
7. **Lifetime ≥ one rollout** — minutes to ~1h: up to 64 agent turns at a 120s per-command
   timeout, plus a verifier run with a ≥600s timeout — with a hard guarantee the container is reaped when the client disappears — orphaned sandboxes at
   0.7/s add up fast.
8. **Credentials** for private/rate-limited registries reachable by whatever runs the pull.

### Open questions for you

- Where do the sandboxes physically run — on the training nodes (as now) or on a separate pool? If
  separate, `put_archive`/`get_archive` and every exec become network round-trips; we'd want to
  know the expected latency, since a 64-turn rollout is 64+ sequential round-trips.
- Does the service own image pulling and caching, or do we still point it at a mirror?
- Is there a per-sandbox filesystem quota, and what happens when a task fills it?
- What's the failure semantics when the Redis coordinator is unavailable mid-rollout — do live
  sandboxes survive?

---

## 10. Code map

| Path | What's there |
|---|---|
| `Dockerfile` (~60–135) | Podman/crun build, `/etc/containers` config, docker→podman symlink, subuid/subgid |
| `docker/podman/containers.conf` | Isolation posture (§2.1) |
| `docker/podman/policy.json` | `insecureAcceptAnything` |
| `docker/podman/10-unqualified-search-registries.conf` | `docker.io` |
| `docker/podman/setup_dockerio_mirror` | Writes `registries.conf`, storage driver, `num_locks` |
| `scripts/docker/setup_podman.sh` | Reproduce the whole install on an existing machine |
| `scripts/docker/docker_login.sh` | Runtime bring-up: shards, sockets, janitors, creds (§4) |
| `open_instruct/environments/backends.py` | `DockerBackend` (the docker-py client), `ApptainerBackend`, semaphores |
| `open_instruct/environments/pool.py` | Actor pool, shard leasing, health cooldown (§6) |
| `open_instruct/environments/swerl_vanillux_sandbox.py` | Per-rollout reset/step logic, persistent-shell wrapper |
| `scripts/general_agent/terminal/rl/*.sh` | Launch scripts — all the env vars in one place |
