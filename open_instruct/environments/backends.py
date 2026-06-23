"""Sandbox backend abstraction for code/command execution."""

import contextlib
import errno
import fcntl
import io
import os
import random
import shlex
import tarfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import docker as docker_sdk

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, value, default)
        return default


class _FileSlotSemaphore:
    """Small cross-process semaphore using advisory locks on per-node files."""

    def __init__(self, name: str, slots: int):
        self.name = name
        self.slots = max(0, slots)
        self.lock_dir = os.getenv("SWERL_DOCKER_LOCK_DIR", "/tmp/open_instruct_docker_locks")
        if self.slots > 0:
            os.makedirs(self.lock_dir, exist_ok=True)

    @contextlib.contextmanager
    def acquire(self):
        if self.slots <= 0:
            yield 0.0
            return

        start_time = time.perf_counter()
        handle = None
        while handle is None:
            for slot in range(self.slots):
                path = os.path.join(self.lock_dir, f"{self.name}.{slot}.lock")
                candidate = open(path, "a+")  # noqa: SIM115 - lock handle lives through the context manager
                try:
                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as e:
                    candidate.close()
                    if e.errno in (errno.EACCES, errno.EAGAIN):
                        continue
                    raise
                handle = candidate
                break
            if handle is None:
                time.sleep(0.05 + random.uniform(0.0, 0.05))

        wait_s = time.perf_counter() - start_time
        try:
            yield wait_s
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


@dataclass
class ExecutionResult:
    """Result from code or command execution."""

    stdout: str
    stderr: str
    exit_code: int


class SandboxOOMError(RuntimeError):
    """Raised when the sandbox container was killed by the OOM reaper.

    Callers should treat this as a terminal condition for the current
    episode (reward 0, done=True) rather than retrying, because the
    agent's next command will almost certainly trip the same limit.
    """


class SandboxBackend(ABC):
    """Abstract interface for code/command execution backends."""

    @abstractmethod
    def start(self) -> None:
        """Initialize the sandbox. Must be called before other operations."""

    @abstractmethod
    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        """Execute a shell command in the sandbox."""

    @abstractmethod
    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file to the sandbox filesystem."""

    @abstractmethod
    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        """Read a file from the sandbox filesystem."""

    @abstractmethod
    def close(self) -> None:
        """Cleanup sandbox resources."""


class DockerBackend(SandboxBackend):
    """Local Docker backend using the ``docker`` Python SDK.

    Runs code in a Docker container on the local machine.
    Requires Docker to be running and the ``docker`` pip package installed.
    """

    _MAX_OUTPUT_BYTES = 1_000_000
    _TRANSIENT_EXEC_API_ERROR_RETRIES = 5
    _TRANSIENT_EXEC_RETRY_BASE_DELAY_S = 0.5
    _TRANSIENT_EXEC_RETRY_MAX_DELAY_S = 8.0
    _TRANSIENT_EXEC_RETRY_JITTER_S = 0.5
    _TRANSIENT_EXEC_API_ERROR_MARKERS = ("database is locked", "retrieving exec session", "timed out waiting for file")
    _START_SEMAPHORE = _FileSlotSemaphore("docker-start", _env_int("SWERL_DOCKER_START_CONCURRENCY", 64))
    _EXEC_SEMAPHORE = _FileSlotSemaphore("docker-exec", _env_int("SWERL_DOCKER_EXEC_CONCURRENCY", 256))
    _TIMING_LOGS = _env_flag("SWERL_SANDBOX_TIMING_LOGS", False)
    _TIMING_LOG_THRESHOLD_S = _env_float("SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S", 1.0)

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: int = 1800,
        mem_limit: str = "4g",
        docker_host: str | None = None,
    ):
        """
        Args:
            image: Docker image to use (default: python:3.12-slim)
            timeout: Per-command timeout in seconds (default: 1800 / 30 min)
            mem_limit: Memory limit for the container (default: 4g)
            docker_host: Optional Docker API endpoint, e.g. ``unix:///tmp/podman.sock``.
        """
        self._image = image
        self._timeout = timeout
        self._mem_limit = mem_limit
        self._docker_host = docker_host
        self._auto_remove = _env_flag("SWERL_DOCKER_AUTO_REMOVE", True)
        self._container = None
        self._client = None

    def _create_client(self):
        if self._docker_host:
            return docker_sdk.DockerClient(base_url=self._docker_host, timeout=300)
        return docker_sdk.from_env(timeout=300)

    def start(self) -> None:
        previous_cid = self._container.short_id if self._container is not None else None
        logger.info(
            "Starting Docker container (image=%s, previous_container=%s, auto_remove=%s, docker_host=%s)",
            self._image,
            previous_cid,
            self._auto_remove,
            self._docker_host or "<from_env>",
        )
        if self._client is None:
            self._client = self._create_client()
        start_time = time.perf_counter()
        with self._START_SEMAPHORE.acquire() as semaphore_wait_s:
            lifecycle_start_time = time.perf_counter()
            phase_timings: dict[str, float] = {}

            phase_start_time = time.perf_counter()
            try:
                self._client.images.get(self._image)
                phase_timings["image_get"] = time.perf_counter() - phase_start_time
            except docker_sdk.errors.ImageNotFound:
                phase_timings["image_get"] = time.perf_counter() - phase_start_time
                phase_start_time = time.perf_counter()
                self._client.images.pull(self._image)
                phase_timings["image_pull"] = time.perf_counter() - phase_start_time

            phase_start_time = time.perf_counter()
            self._container = self._client.containers.create(
                self._image,
                command="sleep infinity",
                detach=True,
                auto_remove=self._auto_remove,
                labels={"open_instruct": "swerl_sandbox"},
                mem_limit=self._mem_limit,
                memswap_limit=self._mem_limit,
            )
            phase_timings["container_create"] = time.perf_counter() - phase_start_time

            phase_start_time = time.perf_counter()
            self._container.start()
            phase_timings["container_start"] = time.perf_counter() - phase_start_time
        elapsed_s = time.perf_counter() - start_time
        create_s = time.perf_counter() - lifecycle_start_time
        if self._TIMING_LOGS and elapsed_s >= self._TIMING_LOG_THRESHOLD_S:
            logger.info(
                "DockerBackend.start timing image=%s container=%s total=%.3fs semaphore_wait=%.3fs create_start=%.3fs phases=%s",
                self._image,
                self._container.short_id,
                elapsed_s,
                semaphore_wait_s,
                create_s,
                {key: round(value, 3) for key, value in phase_timings.items()},
            )
        logger.info(f"Docker container started: {self._container.short_id}")

    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")

        effective_timeout = self._timeout if timeout is None else timeout
        container_id = self._container.short_id
        logger.debug(
            "Docker exec start (container=%s, image=%s, timeout=%ss, command=%r)",
            container_id,
            self._image,
            effective_timeout,
            command,
        )
        wrapped = (
            f"timeout --signal=TERM --kill-after=10 {shlex.quote(str(effective_timeout))} "
            f"bash -c {shlex.quote(command)}"
        )
        try:
            exit_code, output = self._exec_run(wrapped)
        except docker_sdk.errors.NotFound:
            self._log_container_state("exec_not_found", container_id)
            logger.warning(
                "Docker container disappeared before exec (container=%s, image=%s). "
                "Restarting and retrying command once.",
                container_id,
                self._image,
            )
            exit_code, output = self._restart_and_retry_exec(wrapped, container_id)
        except docker_sdk.errors.APIError as e:
            # 409 Conflict is typically "container is not running" (OOM, crash,
            # external stop). Raise SandboxOOMError when OOM-killed so the
            # episode can terminate cleanly; otherwise restart + retry.
            self._log_container_state("exec_api_error", container_id)
            if getattr(e, "status_code", None) == 409:
                if self._container_was_oom_killed(container_id):
                    raise SandboxOOMError(
                        f"Sandbox container {container_id} (image={self._image}) was OOM-killed. Aborting episode."
                    ) from e
                logger.warning(
                    "Docker exec 409 Conflict (container=%s, image=%s): %s. Restarting and retrying command once.",
                    container_id,
                    self._image,
                    e,
                )
                exit_code, output = self._restart_and_retry_exec(wrapped, container_id)
            else:
                if self._is_transient_exec_api_error(e):
                    logger.warning(
                        "Transient Docker exec APIError (container=%s, image=%s): %s. "
                        "Retrying command on the same container.",
                        container_id,
                        self._image,
                        e,
                    )
                    exit_code, output = self._retry_exec_same_container(wrapped, container_id)
                else:
                    logger.warning("Docker exec APIError (container=%s, image=%s): %s", container_id, self._image, e)
                    raise
        stdout_raw = (output[0] or b"") if output else b""
        stderr_raw = (output[1] or b"") if output else b""
        stdout = stdout_raw[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = stderr_raw[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if exit_code == 124:
            stderr = f"Command timed out after {effective_timeout}s.\n" + stderr
        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    @classmethod
    def _is_transient_exec_api_error(cls, error: docker_sdk.errors.APIError) -> bool:
        message = str(error).lower()
        return any(marker in message for marker in cls._TRANSIENT_EXEC_API_ERROR_MARKERS)

    def _exec_run(self, wrapped: str):
        if self._container is None:
            raise RuntimeError("Container missing during Docker exec.")
        start_time = time.perf_counter()
        with self._EXEC_SEMAPHORE.acquire() as semaphore_wait_s:
            exec_start_time = time.perf_counter()
            result = self._container.exec_run(["bash", "-c", wrapped], demux=True)
        elapsed_s = time.perf_counter() - start_time
        exec_s = time.perf_counter() - exec_start_time
        if self._TIMING_LOGS and elapsed_s >= self._TIMING_LOG_THRESHOLD_S:
            logger.info(
                "DockerBackend.exec timing image=%s container=%s total=%.3fs semaphore_wait=%.3fs exec=%.3fs",
                self._image,
                self._container.short_id,
                elapsed_s,
                semaphore_wait_s,
                exec_s,
            )
        return result

    def _retry_exec_same_container(self, wrapped: str, container_id: str):
        """Retry an exec after a transient Docker daemon/storage error."""
        if self._container is None:
            raise RuntimeError("Container missing during Docker exec retry.")
        last_error = None
        for attempt in range(1, self._TRANSIENT_EXEC_API_ERROR_RETRIES + 1):
            delay = self._transient_exec_retry_delay(attempt)
            time.sleep(delay)
            with contextlib.suppress(Exception):
                self._container.reload()
            logger.info(
                "Retrying command after transient Docker exec APIError "
                "(container=%s, image=%s, attempt=%s/%s, delay=%.2fs)",
                container_id,
                self._image,
                attempt,
                self._TRANSIENT_EXEC_API_ERROR_RETRIES,
                delay,
            )
            try:
                return self._exec_run(wrapped)
            except docker_sdk.errors.APIError as e:
                if not self._is_transient_exec_api_error(e):
                    raise
                last_error = e
        if last_error is not None:
            raise last_error
        raise RuntimeError("Docker exec retry failed without capturing an error.")

    @classmethod
    def _transient_exec_retry_delay(cls, attempt: int) -> float:
        backoff = min(
            cls._TRANSIENT_EXEC_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)), cls._TRANSIENT_EXEC_RETRY_MAX_DELAY_S
        )
        return backoff + random.uniform(0.0, cls._TRANSIENT_EXEC_RETRY_JITTER_S)

    def _container_was_oom_killed(self, container_id: str) -> bool:
        """Best-effort probe for ``State.OOMKilled``. Returns False on any error."""
        if self._client is None:
            return False
        try:
            container = self._client.containers.get(container_id)
            with contextlib.suppress(Exception):
                container.reload()
            return bool(container.attrs.get("State", {}).get("OOMKilled"))
        except Exception:
            return False

    def _restart_and_retry_exec(self, wrapped: str, old_container_id: str):
        """Recreate the container and re-run a prepared bash command once.

        Shared between the NotFound and 409-Conflict paths. Returns
        ``(exit_code, output)`` from the retried ``exec_run``.
        """
        self.start()
        if self._container is None:
            raise RuntimeError("Failed to restart Docker container during exec retry.")
        logger.info(
            "Retrying command after container restart (old_container=%s, new_container=%s)",
            old_container_id,
            self._container.short_id,
        )
        return self._exec_run(wrapped)

    def _log_container_state(self, reason: str, container_id: str) -> None:
        """Best-effort container state diagnostics for flaky lifecycle issues."""
        if self._client is None:
            logger.warning(
                "Container state unavailable during %s (container=%s, image=%s): docker client is None",
                reason,
                container_id,
                self._image,
            )
            return

        try:
            container = self._client.containers.get(container_id)
        except docker_sdk.errors.NotFound:
            logger.warning(
                "Container state during %s (container=%s, image=%s): container not found in daemon",
                reason,
                container_id,
                self._image,
            )
            return
        except Exception as e:
            logger.warning(
                "Container state lookup failed during %s (container=%s, image=%s): %s",
                reason,
                container_id,
                self._image,
                e,
            )
            return

        with contextlib.suppress(Exception):
            container.reload()
        state = container.attrs.get("State", {})
        logger.warning(
            "Container state during %s (container=%s, image=%s): status=%s running=%s exit_code=%s "
            "oom_killed=%s error=%s",
            reason,
            container_id,
            self._image,
            state.get("Status"),
            state.get("Running"),
            state.get("ExitCode"),
            state.get("OOMKilled"),
            state.get("Error"),
        )

    def write_file(self, path: str, content: str | bytes) -> None:
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")

        if isinstance(content, str):
            content = content.encode("utf-8")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            info = tarfile.TarInfo(name=os.path.basename(path))
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        tar_stream.seek(0)
        self._container.put_archive(os.path.dirname(path) or "/", tar_stream)

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")

        try:
            tar_chunks, _stat = self._container.get_archive(path)
        except docker_sdk.errors.NotFound:
            raise FileNotFoundError(f"File not found in container: '{path}'") from None
        except docker_sdk.errors.APIError as e:
            raise FileNotFoundError(f"Failed to read file '{path}': {e}") from None

        tar_bytes = b"".join(tar_chunks)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            member = tar.getmembers()[0]
            extracted = tar.extractfile(member)
            if extracted is None:
                raise IsADirectoryError(f"Path '{path}' is a directory, not a file.")
            content_raw = extracted.read()

        if binary:
            return content_raw
        return content_raw.decode("utf-8", errors="replace")

    def close(self) -> None:
        if self._container is not None:
            cid = self._container.short_id
            logger.info(f"Closing Docker container: {cid} (image={self._image})")
            try:
                self._container.kill()
                logger.info(f"Killed Docker container: {cid}")
            except Exception:
                try:
                    self._container.stop(timeout=3)
                    logger.info(f"Stopped Docker container: {cid}")
                except Exception as e:
                    logger.warning(f"Error stopping container {cid}: {e}")
            self._container = None


def create_backend(backend_type: str, **kwargs) -> SandboxBackend:
    """Factory function to create a sandbox backend.

    Args:
        backend_type: Currently only "docker" is supported.
        **kwargs: Backend-specific arguments.

    Returns:
        SandboxBackend instance (not yet started).
    """
    if backend_type == "docker":
        return DockerBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Currently only 'docker' is supported.")
