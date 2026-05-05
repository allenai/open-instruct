"""Pool of Ray actors with acquire/release semantics."""

import asyncio
import os
import random
import time
from typing import Any

import ray

from open_instruct import logger_utils
from open_instruct.environments.backends import is_docker_host_connectivity_error

logger = logger_utils.setup_logger(__name__)

DEFAULT_ACQUIRE_TIMEOUT_S = 7200
ACQUIRE_CONCURRENCY = 1000
RELEASE_CONCURRENCY = 128
DEFAULT_PODMAN_HOST_COOLDOWN_S = 300.0
DEFAULT_PODMAN_HOST_COOLDOWN_JITTER_S = 30.0


def _podman_docker_hosts_from_env() -> list[str]:
    hosts = os.getenv("SWERL_PODMAN_DOCKER_HOSTS", "")
    return [host.strip() for host in hosts.split(",") if host.strip()]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, value, default)
        return default


def _actor_key(actor: ray.actor.ActorHandle) -> str:
    actor_id = getattr(actor, "_actor_id", None)
    if actor_id is not None:
        hex_fn = getattr(actor_id, "hex", None)
        if callable(hex_fn):
            return hex_fn()
        return str(actor_id)
    return str(actor)


def _is_podman_host_failure(error: BaseException) -> bool:
    return is_docker_host_connectivity_error(error)


def _actor_reusable_after_error(error: BaseException) -> bool:
    return not isinstance(error, ray.exceptions.RayActorError)


@ray.remote(concurrency_groups={"acquire": ACQUIRE_CONCURRENCY, "release": RELEASE_CONCURRENCY})
class EnvironmentPool:
    """Shared pool of RLEnvironment Ray actors for concurrent rollouts.

    This is an async Ray actor. acquire() blocks until an actor is available
    (no polling needed — release() wakes up waiting acquirers via asyncio.Queue).
    """

    def __init__(
        self,
        pool_size: int,
        actor_class: type,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT_S,
        **actor_kwargs: Any,
    ):
        self._acquire_timeout = acquire_timeout
        remote_class = ray.remote(actor_class)
        docker_hosts = _podman_docker_hosts_from_env()
        self._docker_hosts = (
            docker_hosts if actor_kwargs.get("backend") == "docker" and "docker_host" not in actor_kwargs else []
        )
        self._host_cursor = 0
        self._host_inflight = {host: 0 for host in self._docker_hosts}
        self._host_unhealthy_until: dict[str, float] = {}
        self._actor_host_leases: dict[str, str] = {}
        self._host_cooldown_s = _env_float("SWERL_PODMAN_HOST_COOLDOWN_S", DEFAULT_PODMAN_HOST_COOLDOWN_S)
        self._host_cooldown_jitter_s = _env_float(
            "SWERL_PODMAN_HOST_COOLDOWN_JITTER_S", DEFAULT_PODMAN_HOST_COOLDOWN_JITTER_S
        )

        logger.info(f"Creating pool of {pool_size} {actor_class.__name__} actors")
        if self._docker_hosts:
            logger.info(
                "Balancing %s %s actors across %s Podman Docker hosts at reset time",
                pool_size,
                actor_class.__name__,
                len(self._docker_hosts),
            )
        self._actors = [remote_class.remote(**actor_kwargs) for _ in range(pool_size)]

        setup_tasks = [actor.setup.remote() for actor in self._actors]
        ray.get(setup_tasks)

        self._available: asyncio.Queue[ray.actor.ActorHandle] = asyncio.Queue()
        for actor in self._actors:
            self._available.put_nowait(actor)
        logger.info(f"Pool ready: {pool_size} {actor_class.__name__} actors")

    async def _acquire_actor(self) -> ray.actor.ActorHandle:
        try:
            return await asyncio.wait_for(self._available.get(), timeout=self._acquire_timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Pool acquire timed out after {self._acquire_timeout}s. "
                f"Pool has {len(self._actors)} actors, {self._available.qsize()} available. "
                f"An actor may have crashed without being released."
            ) from e

    @ray.method(concurrency_group="acquire")
    async def acquire(self) -> ray.actor.ActorHandle:
        return await self._acquire_actor()

    @ray.method(concurrency_group="acquire")
    async def acquire_reset(self, reset_kwargs: dict[str, Any]) -> tuple[ray.actor.ActorHandle, list[dict]]:
        """Acquire an actor and reset it, rotating Podman hosts on host-level failures."""
        actor = await self._acquire_actor()
        try:
            target_tools = await self._reset_actor(actor, reset_kwargs)
        except Exception as e:
            if _actor_reusable_after_error(e):
                await self._release_actor(actor)
            else:
                logger.warning("Not returning crashed environment actor to pool after reset failure: %s", e)
            raise
        return actor, target_tools

    @ray.method(concurrency_group="release")
    async def release(self, actor: ray.actor.ActorHandle) -> None:
        await self._release_actor(actor)

    def size(self) -> int:
        return len(self._actors)

    async def _release_actor(self, actor: ray.actor.ActorHandle) -> None:
        actor_key = _actor_key(actor)
        host = self._actor_host_leases.pop(actor_key, None)
        if host is not None:
            self._release_host(host)
        await self._available.put(actor)

    async def _reset_actor(self, actor: ray.actor.ActorHandle, reset_kwargs: dict[str, Any]) -> list[dict]:
        if not self._docker_hosts:
            _, target_tools = await actor.reset.remote(**reset_kwargs)
            return target_tools

        attempted_hosts: set[str] = set()
        last_error: BaseException | None = None
        while len(attempted_hosts) < len(self._docker_hosts):
            host = self._lease_next_host(attempted_hosts)
            if host is None:
                break
            try:
                kwargs = dict(reset_kwargs)
                kwargs["docker_host"] = host
                _, target_tools = await actor.reset.remote(**kwargs)
                self._actor_host_leases[_actor_key(actor)] = host
                return target_tools
            except Exception as e:
                last_error = e
                self._release_host(host)
                attempted_hosts.add(host)
                if not _is_podman_host_failure(e):
                    raise
                self._mark_host_unhealthy(host, e)
                logger.warning(
                    "Reset failed on Podman Docker host %s; trying another host if available (%s/%s): %s",
                    host,
                    len(attempted_hosts),
                    len(self._docker_hosts),
                    e,
                )

        if last_error is not None:
            raise RuntimeError(
                f"Reset failed after trying {len(attempted_hosts)} Podman Docker hosts: {last_error}"
            ) from last_error
        raise RuntimeError("Reset failed before trying any Podman Docker hosts.")

    def _lease_next_host(self, exclude: set[str]) -> str | None:
        candidates = [host for host in self._healthy_hosts() if host not in exclude]
        if not candidates:
            candidates = [host for host in self._docker_hosts if host not in exclude]
        if not candidates:
            return None

        min_inflight = min(self._host_inflight.get(host, 0) for host in candidates)
        least_loaded = {host for host in candidates if self._host_inflight.get(host, 0) == min_inflight}
        ordered_hosts = self._docker_hosts[self._host_cursor :] + self._docker_hosts[: self._host_cursor]
        for host in ordered_hosts:
            if host in least_loaded:
                self._host_cursor = (self._docker_hosts.index(host) + 1) % len(self._docker_hosts)
                self._host_inflight[host] = self._host_inflight.get(host, 0) + 1
                return host
        return None

    def _healthy_hosts(self) -> list[str]:
        now = time.monotonic()
        return [host for host in self._docker_hosts if self._host_unhealthy_until.get(host, 0.0) <= now]

    def _release_host(self, host: str) -> None:
        self._host_inflight[host] = max(0, self._host_inflight.get(host, 0) - 1)

    def _mark_host_unhealthy(self, host: str, error: BaseException) -> None:
        cooldown = self._host_cooldown_s + random.uniform(0.0, self._host_cooldown_jitter_s)
        self._host_unhealthy_until[host] = time.monotonic() + cooldown
        logger.warning(
            "Temporarily disabling Podman Docker host %s for %.1fs after reset failure: %s", host, cooldown, error
        )
