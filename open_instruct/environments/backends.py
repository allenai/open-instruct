"""Sandbox backend abstraction for code/command execution."""

import atexit
import base64
import contextlib
import errno
import fcntl
import io
import os
import posixpath
import random
import shlex
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

import docker as docker_sdk
import requests

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

DOCKER_HOST_CONNECTIVITY_ERROR_MARKERS = (
    "error while fetching server api version",
    "unixhttpconnectionpool",
    "read timed out",
    "connection refused",
    "connection aborted",
    "broken pipe",
    "connection reset",
)


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


def is_docker_host_connectivity_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in DOCKER_HOST_CONNECTIVITY_ERROR_MARKERS)


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
    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        """Extract a tar archive inside the sandbox, rooted at ``root``.

        ``tar_bytes`` is the raw bytes of a tar archive whose entries are
        interpreted as paths relative to ``root`` inside the sandbox.
        """

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

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")
        start_time = time.perf_counter()
        self._container.put_archive(root, tar_bytes)
        elapsed_s = time.perf_counter() - start_time
        if self._TIMING_LOGS and elapsed_s >= self._TIMING_LOG_THRESHOLD_S:
            logger.info(
                "DockerBackend.put_archive timing image=%s container=%s root=%s bytes=%s total=%.3fs",
                self._image,
                self._container.short_id,
                root,
                len(tar_bytes),
                elapsed_s,
            )

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


# ---------------------------------------------------------------------------
# Apptainer
# ---------------------------------------------------------------------------


# Track live Apptainer instance names so we can stop them if the Python process
# exits abruptly. Apptainer instances do not auto-reap on parent death (unlike
# Docker --rm), so without this the tmpfs overlay lingers until reboot.
_APPTAINER_LIVE_INSTANCES: set[str] = set()


def _apptainer_cleanup_all() -> None:
    for name in list(_APPTAINER_LIVE_INSTANCES):
        with contextlib.suppress(Exception):
            subprocess.run(["apptainer", "instance", "stop", name], capture_output=True, timeout=10)
    _APPTAINER_LIVE_INSTANCES.clear()


atexit.register(_apptainer_cleanup_all)


def _normalize_apptainer_image(image: str) -> str:
    """Return an Apptainer-compatible image reference.

    - URIs with a scheme (``docker://``, ``oras://``, ``shub://``) pass through.
    - Absolute/relative paths and ``*.sif`` strings pass through as filesystem
      paths.
    - Plain Docker tags like ``repo:tag`` get ``docker://`` prepended.
    """
    if "://" in image:
        return image
    if image.startswith(("/", "./")) or image.endswith(".sif"):
        return image
    return f"docker://{image}"


class ApptainerBackend(SandboxBackend):
    """Apptainer backend using ``apptainer instance start/exec/stop``.

    Uses a tmpfs overlay for per-rollout container state (``--writable-tmpfs``)
    and ``--fakeroot`` so commands inside see uid 0 regardless of the host uid.
    All file I/O goes through ``apptainer exec`` with stdin/stdout piping — no
    bind mounts — so the container's filesystem is fully isolated from the
    host. Instances are stopped (and their tmpfs reclaimed) on ``close()`` or
    at process exit via ``atexit``.
    """

    _MAX_OUTPUT_BYTES = 1_000_000

    # Flags that define the sandbox's isolation posture. Kept as a class
    # attribute so tests / subclasses can tune them in one place.
    _DEFAULT_START_FLAGS: tuple[str, ...] = (
        "--fakeroot",
        "--writable-tmpfs",
        "--containall",
        "--no-home",
        "--cleanenv",
    )

    def __init__(
        self,
        image: str = "docker://ubuntu:22.04",
        timeout: int = 1800,
        mem_limit: str | None = None,
        pwd: str = "/workspace",
        cache_dir: str | None = None,
        tmp_dir: str | None = None,
        extra_start_flags: tuple[str, ...] = (),
        apptainer_binary: str = "apptainer",
    ):
        """
        Args:
            image: Image reference. Accepts ``docker://repo:tag``, a plain
                ``repo:tag`` (``docker://`` is prepended), or a path to a
                ``.sif`` file.
            timeout: Per-command timeout in seconds (default: 1800 / 30 min).
            mem_limit: Ignored in fakeroot-fallback mode (no rootless cgroups).
                Present for API symmetry with ``DockerBackend`` so callers can
                pass the same kwargs. Slurm job-level ``--mem`` should be used
                as the real enforcement mechanism.
            pwd: Default cwd inside the container for ``run_command``. Exposed
                via ``APPTAINER_PWD`` so we don't have to pass ``--pwd`` on
                every exec.
            cache_dir: If set, exported as ``APPTAINER_CACHEDIR``. Keep this on
                fast shared storage; the default ``$HOME/.apptainer`` is
                usually quota-limited on HPC.
            tmp_dir: If set, exported as ``APPTAINER_TMPDIR``.
            extra_start_flags: Additional flags appended to
                ``apptainer instance start``.
            apptainer_binary: Name or path of the apptainer CLI. Override for
                testing or to use ``singularity``.
        """
        self._image = _normalize_apptainer_image(image)
        self._timeout = timeout
        self._mem_limit = mem_limit  # Kept for API symmetry; ignored by Apptainer.
        self._pwd = pwd
        self._cache_dir = cache_dir
        self._tmp_dir = tmp_dir
        self._start_flags = tuple(self._DEFAULT_START_FLAGS) + tuple(extra_start_flags)
        self._apptainer = apptainer_binary
        self._name: str | None = None

    # ---- env helpers ------------------------------------------------------

    def _exec_env(self) -> dict:
        env = dict(os.environ)
        env["APPTAINER_PWD"] = self._pwd
        if self._cache_dir:
            env["APPTAINER_CACHEDIR"] = self._cache_dir
        if self._tmp_dir:
            env["APPTAINER_TMPDIR"] = self._tmp_dir
        return env

    def _ensure_binary(self) -> None:
        if shutil.which(self._apptainer) is None:
            raise RuntimeError(
                f"Apptainer binary {self._apptainer!r} not found on PATH. "
                "Install Apptainer >= 1.1 or adjust 'apptainer_binary'."
            )

    def _ensure_started(self) -> None:
        if self._name is None:
            raise RuntimeError("Instance not started. Call start() first.")

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._ensure_binary()
        # Stop any previous instance before starting a new one (supports the
        # "close then start" pattern used in SWERLSandboxEnv._do_reset).
        if self._name is not None:
            self.close()

        name = f"swerl-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        cmd = [self._apptainer, "instance", "start", *self._start_flags, self._image, name]
        logger.info(
            "Starting Apptainer instance (name=%s, image=%s, flags=%s)", name, self._image, " ".join(self._start_flags)
        )
        proc = subprocess.run(cmd, capture_output=True, env=self._exec_env())
        if proc.returncode != 0:
            raise RuntimeError(
                "apptainer instance start failed "
                f"(image={self._image}, exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        self._name = name
        _APPTAINER_LIVE_INSTANCES.add(name)
        logger.info(f"Apptainer instance started: {name}")

    def close(self) -> None:
        if self._name is None:
            return
        name = self._name
        logger.info(f"Closing Apptainer instance: {name}")
        with contextlib.suppress(Exception):
            subprocess.run([self._apptainer, "instance", "stop", name], capture_output=True, timeout=30)
        _APPTAINER_LIVE_INSTANCES.discard(name)
        self._name = None

    # ---- exec -------------------------------------------------------------

    def _exec(
        self, argv: list[str], *, stdin: bytes | None = None, check: bool = False
    ) -> subprocess.CompletedProcess:
        """Run ``apptainer exec instance://<name> <argv>``.

        Pass-through for stdin/capture. Does not wrap in ``timeout`` — callers
        that need a time budget should compose one themselves (see
        ``run_command``).
        """
        self._ensure_started()
        cmd = [self._apptainer, "exec", f"instance://{self._name}", *argv]
        return subprocess.run(cmd, input=stdin, capture_output=True, env=self._exec_env(), check=check)

    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        self._ensure_started()
        effective_timeout = self._timeout if timeout is None else timeout
        wrapped = (
            f"timeout --signal=TERM --kill-after=10 {shlex.quote(str(effective_timeout))} "
            f"bash -c {shlex.quote(command)}"
        )
        logger.debug(
            "Apptainer exec start (instance=%s, image=%s, timeout=%ss, command=%r)",
            self._name,
            self._image,
            effective_timeout,
            command,
        )
        proc = self._exec(["bash", "-c", wrapped])
        stdout = proc.stdout[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = proc.stderr[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if proc.returncode == 124:
            stderr = f"Command timed out after {effective_timeout}s.\n" + stderr
        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)

    # ---- file I/O (exec-piped, no bind mounts) ----------------------------

    def write_file(self, path: str, content: str | bytes) -> None:
        self._ensure_started()
        if isinstance(content, str):
            content = content.encode("utf-8")
        dir_part = os.path.dirname(path) or "/"
        sh_cmd = f"mkdir -p {shlex.quote(dir_part)} && cat > {shlex.quote(path)}"
        proc = self._exec(["sh", "-c", sh_cmd], stdin=content)
        if proc.returncode != 0:
            raise RuntimeError(
                f"write_file failed for {path!r} (exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        self._ensure_started()
        # Use `test -f` to distinguish "missing" from "is a directory" so the
        # caller gets the same exceptions DockerBackend raises.
        check = self._exec(["sh", "-c", f"test -e {shlex.quote(path)}"])
        if check.returncode != 0:
            raise FileNotFoundError(f"File not found in instance: '{path}'")
        is_dir = self._exec(["sh", "-c", f"test -d {shlex.quote(path)}"])
        if is_dir.returncode == 0:
            raise IsADirectoryError(f"Path '{path}' is a directory, not a file.")

        proc = self._exec(["cat", path])
        if proc.returncode != 0:
            raise RuntimeError(f"read_file failed for {path!r}: {proc.stderr.decode('utf-8', 'replace').strip()}")
        if binary:
            return proc.stdout
        return proc.stdout.decode("utf-8", errors="replace")

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        self._ensure_started()
        proc = self._exec(["tar", "-xf", "-", "-C", root], stdin=tar_bytes)
        if proc.returncode != 0:
            raise RuntimeError(
                f"put_archive failed at root={root!r} "
                f"(exit={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )


# ---------------------------------------------------------------------------
# LiteRegistry Podman gateway
# ---------------------------------------------------------------------------


# Server-side contract (literegistry PodmanRequest / PodmanAffinityConfig).
_SERVER_EXEC_TIMEOUT_CAP_S = 60.0
_SERVER_STDIN_LIMIT_BYTES = 1024 * 1024
# Base64 inflates by 4/3; keep a healthy margin below the stdin limit.
_UPLOAD_CHUNK_BYTES = 512 * 1024
# Stay well below the replica's 1MB stdout / 256KB stderr caps: exceeding them
# does not truncate, it aborts the exec (and the gateway then retries it).
_MAX_STDOUT_BYTES = 512 * 1024
_MAX_STDERR_BYTES = 128 * 1024
_READ_CHUNK_BYTES = 384 * 1024

# One wait-exec blocks inside the container until the command finishes or the
# in-container wait budget expires, so completed commands return in one round
# trip and long commands cost one cheap request per ~30s. The gap between the
# two budgets is headroom for `podman exec` spawn/teardown on a loaded replica
# (observed >20s under fleet-wide load); the replica 408s the whole exec if the
# in-container wait plus that overhead exceeds its own budget.
_WAIT_EXEC_TIMEOUT_S = 55.0
_IN_CONTAINER_WAIT_S = 30
# A wait exec that returns neither marker was itself cut short (e.g. its bash
# reaped by the session's cgroup OOM killer alongside the command); the command
# state lives in files, so re-issuing the same exec is safe.
_MAX_MARKERLESS_WAITS = 3
_DONE_MARKER = "__SWERL_GATEWAY_DONE__"
_PENDING_MARKER = "__SWERL_GATEWAY_PENDING__"

_WORK_DIR = "/tmp/.swerl_gateway"


class GatewayBackendError(RuntimeError):
    """A gateway request failed or returned an invalid response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class GatewaySessionLostError(GatewayBackendError):
    """The affinity binding or container no longer exists on the gateway."""


class GatewayBackend(SandboxBackend):
    """Run sandbox commands in a remote Podman container via a LiteRegistry gateway."""

    _HANDSHAKE_ATTEMPTS = 5
    _HANDSHAKE_TIMEOUT_S = 240.0
    _REQUEST_ATTEMPTS = 4
    _RETRY_BASE_DELAY_S = 0.5
    _RETRY_MAX_DELAY_S = 8.0
    # Strict affinity answers 503 with this text when the replica that owns the
    # binding has dropped off the registry roster. A live replica reappears
    # within one retry; a dead/rescheduled one stays off the roster until the
    # binding TTL (~15 min) expires, so after this many such answers the
    # session is treated as lost and re-handshaken instead of retried.
    _OWNER_UNREGISTERED_MARKER = "no longer registered"
    _OWNER_UNREGISTERED_ATTEMPTS = 2
    _TIMING_LOGS = _env_flag("SWERL_SANDBOX_TIMING_LOGS", False)
    _TIMING_LOG_THRESHOLD_S = _env_float("SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S", 1.0)

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: int = 1800,
        mem_limit: str | None = None,
        gateway_url: str = "",
        service: str = "podman",
        client_id: str | None = None,
        keepalive_interval_s: float | None = None,
    ):
        """
        Args:
            image: OCI image for the session container (pulled by the replica,
                normally through the gateway's own docker.io mirror).
            timeout: Default per-command timeout in seconds.
            mem_limit: Ignored. The LiteRegistry Podman replicas do not expose
                per-container memory limits yet; kept for kwarg symmetry with
                ``DockerBackend``.
            gateway_url: Base URL of the LiteRegistry gateway, e.g.
                ``http://host:8080``. Falls back to ``$SWERL_GATEWAY_URL``.
            service: Affinity service name registered in the gateway.
            client_id: Optional identifier sent with the handshake (shows up in
                replica logs); defaults to a per-backend UUID.
            keepalive_interval_s: Seconds between keepalive execs that refresh
                the gateway's affinity TTL while the container sits idle
                between agent turns (generation can outlast the binding TTL).
                Defaults to ``$SWERL_GATEWAY_KEEPALIVE_S`` or 300; ``0``
                disables the keepalive thread.
        """
        resolved_url = (gateway_url or os.getenv("SWERL_GATEWAY_URL", "")).rstrip("/")
        if not resolved_url:
            raise ValueError(
                "GatewayBackend requires a gateway URL. Set env_config.gateway_url or $SWERL_GATEWAY_URL."
            )
        self._gateway_url = resolved_url
        self._image = image
        self._timeout = timeout
        if mem_limit:
            logger.debug("GatewayBackend ignores mem_limit=%s (not supported by the Podman replicas)", mem_limit)
        self._service = service
        self._client_id = client_id or f"open-instruct-{uuid.uuid4().hex[:12]}"
        if keepalive_interval_s is None:
            keepalive_interval_s = _env_float("SWERL_GATEWAY_KEEPALIVE_S", 300.0)
        self._keepalive_interval_s = keepalive_interval_s
        self._affinity_id: str | None = None
        self._instance_id: str | None = None
        self._session = requests.Session()
        self._exec_lock = threading.Lock()
        self._last_request_monotonic = time.monotonic()
        self._keepalive_stop: threading.Event | None = None
        self._keepalive_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ HTTP

    def _post(
        self,
        endpoint: str,
        payload: dict,
        request_timeout: float,
        attempts: int | None = None,
        http_session: requests.Session | None = None,
    ) -> dict:
        """POST one JSON payload to the gateway with bounded retries.

        Retries cover connection errors, request timeouts, and 500/502/503/504
        (replica connection dropped mid-request / temporarily unreachable /
        gateway rediscovering) — safe because every exec this backend issues is
        idempotent. 404 maps to :class:`GatewaySessionLostError` so callers can
        re-handshake, as does a repeated 503 saying the binding's owner is no
        longer registered (the replica died; waiting out the TTL would only
        fail every request until then).
        """
        url = f"{self._gateway_url}/{endpoint.lstrip('/')}"
        max_attempts = attempts or self._REQUEST_ATTEMPTS
        session = http_session or self._session
        last_error: Exception | None = None
        owner_unregistered_answers = 0
        for attempt in range(1, max_attempts + 1):
            self._last_request_monotonic = time.monotonic()
            try:
                response = session.post(url, json=payload, timeout=request_timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = GatewayBackendError(f"Gateway request to {endpoint} failed: {e}")
            else:
                if response.status_code == 404:
                    raise GatewaySessionLostError(
                        f"Gateway affinity session lost ({endpoint}): {response.text[:500]}", status_code=404
                    )
                if response.status_code == 503 and self._OWNER_UNREGISTERED_MARKER in response.text:
                    owner_unregistered_answers += 1
                    if owner_unregistered_answers >= self._OWNER_UNREGISTERED_ATTEMPTS:
                        raise GatewaySessionLostError(
                            f"Gateway affinity owner unregistered ({endpoint}): {response.text[:500]}", status_code=503
                        )
                if response.status_code in (500, 502, 503, 504, 408):
                    last_error = GatewayBackendError(
                        f"Gateway returned HTTP {response.status_code} for {endpoint}: {response.text[:500]}",
                        status_code=response.status_code,
                    )
                elif response.status_code >= 400:
                    raise GatewayBackendError(
                        f"Gateway returned HTTP {response.status_code} for {endpoint}: {response.text[:500]}",
                        status_code=response.status_code,
                    )
                else:
                    try:
                        result = response.json()
                    except ValueError as e:
                        raise GatewayBackendError(f"Gateway returned non-JSON for {endpoint}: {e}") from e
                    if not isinstance(result, dict):
                        raise GatewayBackendError(f"Gateway returned a non-object response for {endpoint}")
                    return result
            if attempt < max_attempts:
                delay = min(self._RETRY_BASE_DELAY_S * (2 ** (attempt - 1)), self._RETRY_MAX_DELAY_S)
                delay += random.uniform(0.0, 0.5)
                logger.warning(
                    "Gateway request %s attempt %s/%s failed (%s); retrying in %.2fs",
                    endpoint,
                    attempt,
                    max_attempts,
                    last_error,
                    delay,
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _exec(
        self,
        command: str,
        stdin: str = "",
        exec_timeout: float = 30.0,
        workdir: str = "/",
        http_session: requests.Session | None = None,
    ) -> dict:
        """Run one bounded exec in the session container via the gateway."""
        if self._affinity_id is None:
            raise GatewayBackendError("Container not started. Call start() first.")
        exec_timeout = min(exec_timeout, _SERVER_EXEC_TIMEOUT_CAP_S - 1.0)
        return self._post(
            "affinity/podman",
            {
                "service": self._service,
                "affinity_id": self._affinity_id,
                "command": command,
                "stdin": stdin,
                "timeout": exec_timeout,
                "workdir": workdir,
            },
            # The replica itself allows exec_timeout + 5s; leave headroom for
            # gateway queueing before giving up on the HTTP request.
            request_timeout=exec_timeout + 20.0,
            http_session=http_session,
        )

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        previous = self._affinity_id
        if previous is not None:
            self.close()
        start_time = time.perf_counter()
        result = self._post(
            "affinity/handshake",
            {"service": self._service, "image": self._image, "client_id": self._client_id},
            request_timeout=self._HANDSHAKE_TIMEOUT_S,
            attempts=self._HANDSHAKE_ATTEMPTS,
        )
        affinity_id = result.get("affinity_id")
        if not isinstance(affinity_id, str) or not affinity_id:
            raise GatewayBackendError(f"Gateway handshake returned no affinity_id: {result}")
        self._affinity_id = affinity_id
        self._instance_id = result.get("instance_id")
        elapsed_s = time.perf_counter() - start_time
        if self._TIMING_LOGS and elapsed_s >= self._TIMING_LOG_THRESHOLD_S:
            logger.info(
                "GatewayBackend.start timing image=%s container=%s instance=%s total=%.3fs",
                self._image,
                affinity_id[:12],
                self._instance_id,
                elapsed_s,
            )
        logger.info(
            "Gateway container started: %s (image=%s, instance=%s, previous=%s)",
            affinity_id[:12],
            self._image,
            self._instance_id,
            previous[:12] if previous else None,
        )
        self._start_keepalive()

    def close(self) -> None:
        self._stop_keepalive()
        if self._affinity_id is None:
            return
        affinity_id = self._affinity_id
        self._affinity_id = None
        self._instance_id = None
        try:
            self._post(
                "affinity/close",
                {"service": self._service, "affinity_id": affinity_id},
                request_timeout=60.0,
                attempts=2,
            )
            logger.info("Closed gateway container: %s", affinity_id[:12])
        except GatewaySessionLostError:
            logger.info("Gateway container already gone at close: %s", affinity_id[:12])
        except GatewayBackendError as e:
            # The replica reaps leftover containers on restart; do not fail
            # the episode over a close error.
            logger.warning("Failed to close gateway container %s: %s", affinity_id[:12], e)

    # -------------------------------------------------------------- commands

    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        if self._affinity_id is None:
            raise GatewayBackendError("Container not started. Call start() first.")
        effective_timeout = self._timeout if timeout is None else timeout
        with self._exec_lock:
            start_time = time.perf_counter()
            try:
                result = self._run_command_once(command, effective_timeout)
            except GatewaySessionLostError as e:
                # Mirrors DockerBackend's restart-and-retry-once semantics when
                # the container disappears mid-episode.
                logger.warning(
                    "Gateway container disappeared before exec (%s). Restarting and retrying command once.", e
                )
                self._affinity_id = None
                self.start()
                result = self._run_command_once(command, effective_timeout)
            elapsed_s = time.perf_counter() - start_time
            if self._TIMING_LOGS and elapsed_s >= self._TIMING_LOG_THRESHOLD_S:
                logger.info(
                    "GatewayBackend.exec timing image=%s container=%s total=%.3fs",
                    self._image,
                    (self._affinity_id or "?")[:12],
                    elapsed_s,
                )
            return result

    def _run_command_once(self, command: str, effective_timeout: float) -> ExecutionResult:
        token = uuid.uuid4().hex[:16]
        job_dir = f"{_WORK_DIR}/{token}"
        quoted_dir = shlex.quote(job_dir)
        # Launch the command detached. The bare `mkdir` acts as a lock: if the
        # gateway retries this exec after the first attempt actually ran, the
        # second attempt exits without launching a duplicate.
        # `{ ... & }` groups the background operator with just the setsid
        # command; otherwise `&` would background the whole `&&` chain and
        # `cat` would read /dev/null instead of the request stdin. If the job
        # dir already exists, a previous attempt of this same request already
        # launched the command (the gateway retried a request whose response
        # was lost), so skip straight to waiting instead of double-launching.
        detached = shlex.quote(
            f"timeout --signal=TERM --kill-after=10 {shlex.quote(str(effective_timeout))} "
            f"bash {job_dir}/cmd.sh > {job_dir}/out 2> {job_dir}/err < /dev/null; "
            f"echo $? > {job_dir}/rc.tmp && mv {job_dir}/rc.tmp {job_dir}/rc"
        )
        launcher = (
            f"mkdir -p {shlex.quote(_WORK_DIR)} && "
            f"if mkdir {quoted_dir} 2> /dev/null; then "
            f"cat > {quoted_dir}/cmd.sh && "
            "{ setsid nohup bash -c " + detached + " > /dev/null 2>&1 < /dev/null & } && echo LAUNCHED; "
            "else echo LAUNCHED; fi"
        )
        launch = self._exec(launcher, stdin=command, exec_timeout=30.0)
        if "LAUNCHED" not in launch.get("stdout", ""):
            raise GatewayBackendError(
                f"Gateway command launcher failed (exit={launch.get('exit_code')}): {launch.get('stderr', '')[:500]}"
            )

        # Wait for the rc file, then emit bounded output in the same exec.
        waiter = (
            f"if timeout {_IN_CONTAINER_WAIT_S} bash -c 'until [ -f {job_dir}/rc ]; do sleep 0.1; done'; then "
            f'echo {_DONE_MARKER}; echo "RC=$(cat {quoted_dir}/rc)"; '
            f"head -c {_MAX_STDOUT_BYTES} {quoted_dir}/out; "
            f"head -c {_MAX_STDERR_BYTES} {quoted_dir}/err >&2; "
            f"else echo {_PENDING_MARKER}; fi"
        )
        # Budget: command timeout, plus TERM->KILL grace, plus slack for the
        # detached launcher and filesystem latency.
        deadline = time.monotonic() + effective_timeout + 60.0
        markerless_waits = 0
        while True:
            response = self._exec(waiter, exec_timeout=_WAIT_EXEC_TIMEOUT_S, workdir="/")
            stdout = response.get("stdout", "")
            if _DONE_MARKER in stdout:
                break
            if _PENDING_MARKER not in stdout:
                markerless_waits += 1
                if markerless_waits >= _MAX_MARKERLESS_WAITS:
                    raise GatewayBackendError(
                        f"Gateway wait exec returned no marker {markerless_waits}x in a row "
                        f"(exit={response.get('exit_code')}): "
                        f"stdout={stdout[:200]!r} stderr={response.get('stderr', '')[:200]!r}"
                    )
                logger.warning(
                    "Gateway wait exec returned no marker (exit=%s); retrying (%s/%s)",
                    response.get("exit_code"),
                    markerless_waits,
                    _MAX_MARKERLESS_WAITS,
                )
            if time.monotonic() > deadline:
                self._cleanup_job_dir(quoted_dir)
                raise GatewayBackendError(
                    f"Gateway command did not finish within {effective_timeout}s plus grace; giving up."
                )

        payload = stdout.split(_DONE_MARKER, 1)[1].lstrip("\n")
        rc_line, _, out = payload.partition("\n")
        try:
            exit_code = int(rc_line.removeprefix("RC=").strip())
        except ValueError:
            raise GatewayBackendError(f"Gateway wait exec returned an invalid rc line: {rc_line!r}") from None
        stderr = response.get("stderr", "")
        self._cleanup_job_dir(quoted_dir)
        if exit_code == 124:
            stderr = f"Command timed out after {effective_timeout}s.\n" + stderr
        return ExecutionResult(stdout=out, stderr=stderr, exit_code=exit_code)

    def _cleanup_job_dir(self, quoted_dir: str) -> None:
        """Best-effort removal of a finished command's scratch dir.

        The command's result is already in hand by the time this runs, so a
        failure here (typically a 408 from `podman exec` latency on a loaded
        replica) must never turn a completed command into a tool error. The
        dir is tiny and dies with the session container anyway.
        """
        try:
            self._exec(f"rm -rf {quoted_dir}", exec_timeout=15.0)
        except GatewayBackendError as e:
            logger.warning("Gateway job-dir cleanup failed (ignored): %s", str(e)[:200])

    # ------------------------------------------------------------- keepalive

    def _start_keepalive(self) -> None:
        if self._keepalive_interval_s <= 0 or self._keepalive_thread is not None:
            return
        self._keepalive_stop = threading.Event()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, name="gateway-backend-keepalive", daemon=True
        )
        self._keepalive_thread.start()

    def _stop_keepalive(self) -> None:
        if self._keepalive_stop is not None:
            self._keepalive_stop.set()
        self._keepalive_thread = None
        self._keepalive_stop = None

    def _keepalive_loop(self) -> None:
        """Refresh the affinity TTL while the container idles between turns.

        The binding TTL (15 minutes by default) only refreshes on traffic; a
        long generation pause between agent turns would otherwise let it
        expire mid-episode. Uses its own HTTP session (requests.Session is not
        thread-safe) and skips ticks when a command exec is in flight.
        """
        stop = self._keepalive_stop
        session = requests.Session()
        assert stop is not None
        while not stop.wait(self._keepalive_interval_s):
            affinity_id = self._affinity_id
            if affinity_id is None:
                continue
            if time.monotonic() - self._last_request_monotonic < self._keepalive_interval_s / 2:
                continue
            if not self._exec_lock.acquire(blocking=False):
                continue  # A real command is in flight; it refreshes the TTL.
            try:
                self._exec("true", exec_timeout=15.0, http_session=session)
            except Exception as e:
                logger.warning("Gateway keepalive failed for %s: %s", affinity_id[:12], e)
            finally:
                self._exec_lock.release()
        session.close()

    # --------------------------------------------------------------- file IO

    def write_file(self, path: str, content: str | bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        quoted = shlex.quote(path)
        parent = shlex.quote(posixpath.dirname(path) or "/")
        with self._exec_lock:
            if len(content) <= _UPLOAD_CHUNK_BYTES:
                result = self._exec(
                    f"mkdir -p {parent} && base64 -d > {quoted}",
                    stdin=base64.b64encode(content).decode(),
                    exec_timeout=30.0,
                )
                if result.get("exit_code") != 0:
                    raise GatewayBackendError(f"write_file failed for {path!r}: {result.get('stderr', '')[:500]}")
                return
            self._upload_chunked(content, target_command=f"cat > {quoted}", parent_dir=parent, label=path)

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        quoted_root = shlex.quote(root)
        with self._exec_lock:
            start_time = time.perf_counter()
            if len(tar_bytes) <= _UPLOAD_CHUNK_BYTES:
                result = self._exec(
                    f"mkdir -p {quoted_root} && base64 -d | tar -xf - -C {quoted_root}",
                    stdin=base64.b64encode(tar_bytes).decode(),
                    exec_timeout=45.0,
                )
                if result.get("exit_code") != 0:
                    raise GatewayBackendError(f"put_archive failed at root={root!r}: {result.get('stderr', '')[:500]}")
            else:
                self._upload_chunked(
                    tar_bytes, target_command=f"tar -xf - -C {quoted_root}", parent_dir=quoted_root, label=root
                )
            elapsed_s = time.perf_counter() - start_time
            if self._TIMING_LOGS and elapsed_s >= self._TIMING_LOG_THRESHOLD_S:
                logger.info(
                    "GatewayBackend.put_archive timing image=%s root=%s bytes=%s total=%.3fs",
                    self._image,
                    root,
                    len(tar_bytes),
                    elapsed_s,
                )

    def _upload_chunked(self, data: bytes, target_command: str, parent_dir: str, label: str) -> None:
        """Upload ``data`` in idempotent base64 chunk files, then assemble.

        Each chunk exec overwrites its own part file, so a gateway-level retry
        of any single request cannot corrupt the payload.
        """
        token = uuid.uuid4().hex[:16]
        stage_dir = f"{_WORK_DIR}/upload_{token}"
        quoted_stage = shlex.quote(stage_dir)
        chunks = [data[i : i + _UPLOAD_CHUNK_BYTES] for i in range(0, len(data), _UPLOAD_CHUNK_BYTES)]
        result = self._exec(f"mkdir -p {quoted_stage}", exec_timeout=15.0)
        if result.get("exit_code") != 0:
            raise GatewayBackendError(f"upload staging failed for {label!r}: {result.get('stderr', '')[:500]}")
        for index, chunk in enumerate(chunks):
            result = self._exec(
                f"base64 -d > {quoted_stage}/part.{index:06d}",
                stdin=base64.b64encode(chunk).decode(),
                exec_timeout=30.0,
            )
            if result.get("exit_code") != 0:
                raise GatewayBackendError(
                    f"upload chunk {index} failed for {label!r}: {result.get('stderr', '')[:500]}"
                )
        result = self._exec(
            f"mkdir -p {parent_dir} && cat {quoted_stage}/part.* | {target_command} && rm -rf {quoted_stage}",
            exec_timeout=_WAIT_EXEC_TIMEOUT_S,
        )
        if result.get("exit_code") != 0:
            raise GatewayBackendError(f"upload assembly failed for {label!r}: {result.get('stderr', '')[:500]}")

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        quoted = shlex.quote(path)
        with self._exec_lock:
            probe = self._exec(
                f"if [ ! -e {quoted} ]; then exit 40; elif [ -d {quoted} ]; then exit 41; fi; wc -c < {quoted}",
                exec_timeout=15.0,
            )
            exit_code = probe.get("exit_code")
            if exit_code == 40:
                raise FileNotFoundError(f"File not found in container: '{path}'")
            if exit_code == 41:
                raise IsADirectoryError(f"Path '{path}' is a directory, not a file.")
            if exit_code != 0:
                raise GatewayBackendError(f"read_file probe failed for {path!r}: {probe.get('stderr', '')[:500]}")
            try:
                size = int(probe.get("stdout", "").strip())
            except ValueError:
                raise GatewayBackendError(
                    f"read_file probe returned an invalid size for {path!r}: {probe.get('stdout', '')!r}"
                ) from None

            parts: list[bytes] = []
            for offset in range(0, max(size, 1), _READ_CHUNK_BYTES):
                result = self._exec(
                    f"tail -c +{offset + 1} {quoted} | head -c {_READ_CHUNK_BYTES} | base64 -w0", exec_timeout=30.0
                )
                if result.get("exit_code") != 0:
                    raise GatewayBackendError(f"read_file failed for {path!r}: {result.get('stderr', '')[:500]}")
                parts.append(base64.b64decode(result.get("stdout", "").strip()))
        content = b"".join(parts)
        if binary:
            return content
        return content.decode("utf-8", errors="replace")


def create_backend(backend_type: str, **kwargs) -> SandboxBackend:
    """Factory function to create a sandbox backend.

    Args:
        backend_type: ``"docker"``, ``"apptainer"``, or ``"gateway"``.
        **kwargs: Backend-specific arguments.

    Returns:
        SandboxBackend instance (not yet started).
    """
    if backend_type != "gateway":
        kwargs.pop("gateway_url", None)
    if backend_type == "docker":
        return DockerBackend(**kwargs)
    if backend_type == "apptainer":
        return ApptainerBackend(**kwargs)
    if backend_type == "gateway":
        return GatewayBackend(**kwargs)
    raise ValueError(f"Unknown backend type: {backend_type}. Supported: 'docker', 'apptainer', 'gateway'.")
