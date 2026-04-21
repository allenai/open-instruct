"""Sandbox backend abstraction for code/command execution."""

import io
import os
import shlex
import tarfile
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import docker as docker_sdk

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


@dataclass
class ExecutionResult:
    """Result from code or command execution."""

    stdout: str
    stderr: str
    exit_code: int


class SandboxBackend(ABC):
    """Abstract interface for code/command execution backends."""

    @abstractmethod
    def start(self) -> None:
        """Initialize the sandbox. Must be called before other operations."""

    @abstractmethod
    def run_command(self, command: str) -> ExecutionResult:
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

    def __init__(self, image: str = "python:3.12-slim", timeout: int = 1800, mem_limit: str = "4g"):
        """
        Args:
            image: Docker image to use (default: python:3.12-slim)
            timeout: Per-command timeout in seconds (default: 1800 / 30 min)
            mem_limit: Memory limit for the container (default: 4g)
        """
        self._image = image
        self._timeout = timeout
        self._mem_limit = mem_limit
        self._container = None
        self._client = None

    def start(self) -> None:
        previous_cid = self._container.short_id if self._container is not None else None
        logger.info(f"Starting Docker container (image={self._image}, previous_container={previous_cid})")
        if self._client is None:
            self._client = docker_sdk.from_env(timeout=300)
        self._container = self._client.containers.run(
            self._image,
            command="sleep infinity",
            detach=True,
            remove=True,
            mem_limit=self._mem_limit,
            memswap_limit=self._mem_limit,
        )
        logger.info(f"Docker container started: {self._container.short_id}")

    def run_command(self, command: str) -> ExecutionResult:
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")

        container_id = self._container.short_id
        logger.debug(
            "Docker exec start (container=%s, image=%s, timeout=%ss, command=%r)",
            container_id,
            self._image,
            self._timeout,
            command,
        )
        wrapped = (
            f"timeout --signal=TERM --kill-after=10 {shlex.quote(str(self._timeout))} bash -c {shlex.quote(command)}"
        )
        try:
            exit_code, output = self._container.exec_run(["bash", "-c", wrapped], demux=True)
        except docker_sdk.errors.NotFound:
            self._log_container_state("exec_not_found", container_id)
            # The container can disappear between reset/start and exec (e.g. daemon cleanup
            # or external removal). Recreate it once and retry this command.
            logger.warning(
                "Docker container disappeared before exec (container=%s, image=%s). "
                "Restarting and retrying command once.",
                container_id,
                self._image,
            )
            self.start()
            if self._container is None:
                raise RuntimeError("Failed to restart Docker container after NotFound error.")
            logger.info(
                "Retrying command after container restart (old_container=%s, new_container=%s)",
                container_id,
                self._container.short_id,
            )
            exit_code, output = self._container.exec_run(["bash", "-c", wrapped], demux=True)
        except docker_sdk.errors.APIError as e:
            # Capture state for conflicts like "container is not running" before bubbling up.
            self._log_container_state("exec_api_error", container_id)
            logger.warning(
                "Docker exec APIError (container=%s, image=%s): %s",
                container_id,
                self._image,
                e,
            )
            raise
        stdout_raw = (output[0] or b"") if output else b""
        stderr_raw = (output[1] or b"") if output else b""
        stdout = stdout_raw[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        stderr = stderr_raw[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if exit_code == 124:
            stderr = f"Command timed out after {self._timeout}s.\n" + stderr
        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

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
