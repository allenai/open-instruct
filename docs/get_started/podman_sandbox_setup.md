# Podman Sandbox Setup

This setup is for Beaker jobs that need sandbox tasks to create their own
containers, such as SWERL-style tool environments. Beaker does not expose the
host Docker socket when `BEAKER_SKIP_DOCKER_SOCKET=1`, so the training job starts
an in-task container runtime and points Docker SDK clients at it.

There are two supported modes:

- Podman service mode, via `source scripts/docker/docker_login.sh`.
- Docker-in-Docker mode, via `source scripts/docker/start_dind.sh`.

Podman service mode is the default path for high-concurrency sandbox rollouts.
DinD is useful when a workload requires a real Docker daemon instead of the
Podman Docker API compatibility layer.

## Image Setup

The repository `Dockerfile` installs Podman, `crun`, container config files under
`docker/podman/`, and `setup_dockerio_mirror`.

Podman mode also symlinks `docker` to `podman` in `/usr/local/bin` so command-line
Docker calls inside sandbox tasks go through Podman by default. DinD mode
prepends `DIND_PREFIX` to `PATH`, so the static Docker CLI installed there is
found first without needing to write into `/usr/local/bin`.

## Beaker Launch Requirements

Launch jobs that use subcontainers with:

```bash
--env BEAKER_ALLOW_SUBCONTAINERS=1 \
--env BEAKER_SKIP_DOCKER_SOCKET=1 \
--secret DOCKER_PAT=hamishivi_DOCKER_PAT
```

Source the runtime setup script before Ray starts so Ray workers inherit
`DOCKER_HOST` and related environment variables:

```bash
source scripts/docker/docker_login.sh && source configs/beaker_configs/ray_node_setup.sh && python open_instruct/grpo_fast.py ...
```

For DinD:

```bash
source scripts/docker/start_dind.sh && source configs/beaker_configs/ray_node_setup.sh && python open_instruct/grpo_fast.py ...
```

## Podman Service Mode

`scripts/docker/docker_login.sh` does four things:

- Runs `/usr/local/bin/setup_dockerio_mirror` to write Podman registry and storage
  config from environment variables.
- Starts one or more `podman system service` sockets for Docker SDK clients.
- Exports `DOCKER_HOST` to the first Podman socket and
  `SWERL_PODMAN_DOCKER_HOSTS` to the comma-separated list of all shard sockets.
- Writes Docker Hub credentials to `~/.docker/config.json` when `DOCKER_PAT` is
  available.

With `SWERL_PODMAN_SERVICE_COUNT > 1`, each Podman service gets separate storage
roots. The environment pool can use `SWERL_PODMAN_DOCKER_HOSTS` to spread
sandbox actors across those sockets.

Common Podman flags:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SWERL_PODMAN_SERVICE_COUNT` | `1` | Number of Podman API service shards to start. |
| `PODMAN_SOCKET_PATH` | `/tmp/podman.sock` | Socket path for single-service mode. |
| `SWERL_PODMAN_SERVICE_DIR` | `/tmp/podman-services` | Parent directory for sharded socket/log files. |
| `SWERL_PODMAN_GRAPHROOT_BASE` | `/var/lib/containers/storage/swerl-podman-shards` | Parent directory for per-shard Podman graph roots. |
| `SWERL_PODMAN_RUNROOT_BASE` | `/run/containers/storage/swerl-podman-shards` | Parent directory for per-shard Podman run roots. |
| `SWERL_PODMAN_TMPDIR_BASE` | `/var/tmp/swerl-podman-shards` | Parent directory for per-shard Podman temp dirs. |
| `PODMAN_NUM_LOCKS` | `8192` | Value written to `[engine].num_locks` in `/etc/containers/containers.conf`. Increase for many concurrent containers. |
| `CONTAINERS_STORAGE_CONF` | unset | Set to `/etc/containers/storage.conf` when scripts should force the generated storage config. |
| `DOCKERHUB_USERNAME` | `hamishivi` | Username paired with `DOCKER_PAT` for Docker Hub auth. |
| `DOCKER_PAT` | unset | Docker Hub token, usually provided by a Beaker secret. |

Cleanup flags:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SWERL_DOCKER_JANITOR_ENABLED` | unset | Starts a background cleanup loop for exited sandbox containers. If unset, it auto-enables when `SWERL_DOCKER_AUTO_REMOVE=0`. |
| `SWERL_DOCKER_JANITOR_INTERVAL_S` | `60` | Seconds between exited-container cleanup passes. |
| `SWERL_DOCKER_JANITOR_BATCH_SIZE` | `20` | Max exited containers removed per shard per pass. |
| `SWERL_PODMAN_IMAGE_JANITOR_ENABLED` | `0` | Starts a background `podman image prune` loop. |
| `SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S` | `300` | Seconds between image prune passes. |
| `SWERL_PODMAN_IMAGE_JANITOR_UNTIL` | `30m` | Age filter passed to `podman image prune --filter until=...`. |

## Registry Mirror

`setup_dockerio_mirror` reads `MIRROR_URL` and rewrites Podman config so pulls
from Docker Hub can use a local registry mirror.

```bash
--env MIRROR_URL=jupiter-cs-aus-218.reviz.ai2.in:5000
```

Multiple mirrors can be comma-separated. Mirrors are written as insecure
`docker.io` mirrors because Beaker-local registry mirrors are commonly served
over plain HTTP.

The helper also forces overlay storage and updates Podman lock count:

- `/etc/containers/storage.conf`: `driver = "overlay"`
- `/etc/containers/containers.conf`: `[engine].num_locks = $PODMAN_NUM_LOCKS`

## DinD Mode

`scripts/docker/start_dind.sh` starts a local Docker daemon inside a user
namespace and exports:

```bash
DOCKER_HOST=unix:///run/docker/docker.sock
SWERL_PODMAN_DOCKER_HOSTS=$DOCKER_HOST
```

It installs a static Docker release under `DIND_PREFIX`, starts `dockerd` with
networking through `slirp4netns`, and runs a smoke test by default.

Common DinD flags:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DIND_DOCKER_VERSION` | `27.3.1` | Static Docker release to install. |
| `DIND_PREFIX` | `/opt/docker` | Install location for Docker binaries and wrappers. |
| `DIND_RUN_DIR` | `/run/docker` | Runtime directory and default socket parent. |
| `DIND_DATA_ROOT` | `/var/lib/docker` | Docker data root. |
| `DIND_SOCKET` | `$DIND_RUN_DIR/docker.sock` | Docker daemon socket path. |
| `DIND_STORAGE_DRIVER` | `vfs` | Storage driver passed to `dockerd`. |
| `DIND_BRIDGE` | `docker0` | Bridge network passed to `dockerd`; set to `none` only for workloads that do not need network inside sandbox containers. |
| `DIND_IPTABLES` | `true` | Whether `dockerd` manages iptables for container NAT. |
| `DIND_IP_FORWARD` | `true` | Whether `dockerd` enables IP forwarding for bridge networking. |
| `DIND_SMOKE_TEST` | `1` | Run a tiny container after startup. |
| `DIND_SMOKE_IMAGE` | `python:3.12-slim` | Image used by the smoke test. |
| `DIND_SLIRP4NETNS_BIN` | auto-detected | Override the `slirp4netns` binary path. |

## Runtime Concurrency Flags

These flags are consumed by the sandbox backend changes in the companion
DockerBackend hardening PR. They are useful with Podman because too many
simultaneous container starts or execs can overwhelm the runtime.

| Variable | Typical value | Purpose |
| --- | --- | --- |
| `SWERL_DOCKER_START_CONCURRENCY` | `64` | Cross-process limit for concurrent container starts. |
| `SWERL_DOCKER_EXEC_CONCURRENCY` | `256` | Cross-process limit for concurrent container exec calls. |
| `SWERL_DOCKER_LOCK_DIR` | `/tmp/open_instruct_docker_locks_<uid>` | Directory for file-slot semaphore locks. |
| `SWERL_DOCKER_AUTO_REMOVE` | `1` | Whether sandbox containers auto-remove on exit. |
| `SWERL_SANDBOX_TIMING_LOGS` | `0` | Enables timing logs for sandbox runtime phases. |
| `SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S` | `1.0` | Only log timings above this threshold. |

## Troubleshooting

- Podman logs are written under `/output/tmp`:
  - `podman-system-service.log`
  - `podman-info.log`
  - `podman-janitor.log`
  - `podman-image-janitor.log`
- DinD logs are written under `/output/tmp` by default:
  - `dockerd.log`
  - `slirp.log`
- If Podman reports `cannot clone: Operation not permitted`, verify the Beaker
  job has `BEAKER_ALLOW_SUBCONTAINERS=1` and `BEAKER_SKIP_DOCKER_SOCKET=1`.
- If pulls are slow or rate-limited, verify `DOCKER_PAT`, `DOCKERHUB_USERNAME`,
  and `MIRROR_URL`.
