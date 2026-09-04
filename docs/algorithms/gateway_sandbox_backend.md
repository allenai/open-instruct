# Gateway sandbox backend (LiteRegistry Podman)

The `gateway` sandbox backend runs terminal-RL sandbox containers on an
**external fleet of Podman servers** behind a
[LiteRegistry](https://github.com/goncalorafaria/literegistry) gateway, instead
of podman `system service` daemons colocated inside the training job.

```text
training job (Ray env actors)                LiteRegistry deployment (separate Beaker experiment)
┌───────────────────────────────┐            ┌─────────────────────────────────────────────┐
│ EnvironmentPool               │   HTTP     │ gateway ──► podman replica 1..N (containers)│
│  └─ SWERLVanilluxSandboxEnv   ├───────────►│    │   ──► docker-mirror replica 1..M       │
│      └─ GatewayBackend        │            │    └──► Redis (service registry + affinity) │
└───────────────────────────────┘            └─────────────────────────────────────────────┘
```

Why: with colocated podman, a podman daemon crash or a full `/tmp` takes down
rollouts on that training node and there is no way to repair it without
restarting the job. With the gateway, the sandbox fleet is deployed, scaled,
and repaired independently of training: replicas heartbeat into Redis, the
gateway load-balances new sessions across live replicas, and a dead replica
just stops receiving handshakes.

## Using it

Set `"backend": "gateway"` and `"gateway_url"` in `--tool_configs`:

```bash
--tools swerl_vanillux_sandbox \
--tool_configs '{"backend": "gateway", "gateway_url": "http://GATEWAY_HOST:PORT", "task_data_hf_repo": "allenai/tmax-15k-open-instruct", "test_timeout": 120, "image": "python:3.12-slim"}'
```

(`$SWERL_GATEWAY_URL` is the env-var fallback when `gateway_url` is unset.)

Everything podman-related disappears from the launch script: no
`scripts/docker/docker_login.sh`, no `SWERL_PODMAN_SERVICE_COUNT` /
`SWERL_DOCKER_*` / `MIRROR_URL` / `PODMAN_NUM_LOCKS` env vars, no `DOCKER_PAT`
secret, no `BEAKER_ALLOW_SUBCONTAINERS`. Task images are pulled by the podman
replicas through the deployment's own docker.io pull-through mirrors.

Reference scripts: `scripts/general_agent/terminal/rl/gateway/`.

## Session lifecycle

- `reset()` → `POST /affinity/handshake {service, image}`: the gateway picks a
  replica, the replica `podman run`s a container from the task image, and the
  returned `affinity_id` (= container ID) is bound to that replica in Redis.
- every `bash` step → `POST /affinity/podman {affinity_id, command, stdin,
  timeout, workdir}`: the gateway routes to the bound replica, which
  `podman exec`s in the container. Each request refreshes the binding TTL.
- episode end → `POST /affinity/close {affinity_id}`: removes the container
  and releases the binding.

## How `GatewayBackend` works around the per-exec contract

The replicas enforce hard per-exec limits (60s timeout cap, 1MB stdin, and
they *abort* rather than truncate >1MB stdout / >256KB stderr, after which the
gateway retries the exec — re-running the command). `run_command()` therefore
never executes agent commands as a bare exec:

1. **launch** — one exec ships the command via stdin into
   `/tmp/.swerl_gateway/<token>/cmd.sh` and starts it detached under
   `timeout`, with stdout/stderr redirected to files and the exit code written
   to an `rc` file. A bare `mkdir <token>` acts as a lock so a gateway-level
   retry cannot double-launch the command.
2. **wait+read** — repeated execs block in-container (up to 45s each) until
   the `rc` file exists, then emit the exit code plus `head -c`-bounded
   stdout/stderr in the same response. Fast commands complete in one round
   trip; a 600s verifier costs one cheap request per ~45s.

File I/O (`write_file`, `read_file`, `put_archive` for task seeds/tests) moves
as base64 chunks sized under the stdin/stdout caps; every chunk write is
idempotent so gateway retries are safe.

A keepalive thread (default every 300s, `SWERL_GATEWAY_KEEPALIVE_S`, `0`
disables) runs a no-op exec while the container idles between agent turns, so
the affinity binding (15-minute TTL by default) cannot expire during a long
generation pause. If a binding is lost anyway (404), the backend re-handshakes
and retries the command once — the same restart-and-retry semantics
`DockerBackend` uses when a local container disappears.

## Failure modes vs colocated podman

| failure | colocated podman | gateway |
|---|---|---|
| podman daemon crash / wedge | rollouts on the node fail until job restart | replica drops out of Redis; new handshakes go elsewhere; in-flight episodes on it fail their reset retries and are retried |
| node `/tmp` or storage full | DataPreparationActor hangs (see `reference_tmax_disk_full_sandbox_hang`) | contained to one replica; fix/restart it independently |
| replica preempted/restarted | n/a | its containers are cleaned up on restart; episodes re-handshake |
| training job killed | containers die with the node | containers leak on replicas until the binding TTL expires and the replica restarts (see gaps below) |

## Known gaps (in the LiteRegistry deployment, as of 2026-08-31)

- **No per-container memory / pids limits**: colocated podman ran containers
  with `mem_limit=4g`; replica containers are currently unbounded, so one
  runaway rollout can pressure a whole replica node. Needs `--memory` support
  in the replica's `podman run`.
- **Oversized output wedges a session**: a bare exec producing >1MB stdout
  makes the replica return 413 and the gateway retry (re-running the command)
  up to 20 times, serializing behind the per-container lock. `GatewayBackend`
  never triggers this (all output is file-redirected), but any other client
  sharing the deployment can.
- **No idle-container janitor on replicas**: containers whose client vanished
  without `close()` persist until the replica restarts.
- **Handshake pull races the gateway timeout**: a cold image pull slower than
  the gateway's per-request timeout makes the gateway retry the handshake on
  another replica, orphaning the first container.
