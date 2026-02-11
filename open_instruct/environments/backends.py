"""Sandbox backend abstraction for code/command execution."""

import base64
import contextlib
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    from e2b_code_interpreter import Sandbox as E2BSandbox

    HAS_E2B = True
except ImportError:
    E2BSandbox = None
    HAS_E2B = False

try:
    import docker as docker_sdk

    HAS_DOCKER = True
except ImportError:
    docker_sdk = None
    HAS_DOCKER = False

try:
    from daytona import Daytona, DaytonaConfig

    HAS_DAYTONA = True
except ImportError:
    Daytona = None
    DaytonaConfig = None
    HAS_DAYTONA = False

logger = logging.getLogger(__name__)


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
        pass

    @abstractmethod
    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """
        Execute code in the sandbox.

        Args:
            code: Source code to execute
            language: Programming language (default: python)

        Returns:
            ExecutionResult with stdout, stderr, and exit code
        """
        pass

    @abstractmethod
    def run_command(self, command: str) -> ExecutionResult:
        """
        Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute

        Returns:
            ExecutionResult with stdout, stderr, and exit code
        """
        pass

    @abstractmethod
    def write_file(self, path: str, content: str | bytes) -> None:
        """
        Write a file to the sandbox filesystem.

        Args:
            path: Absolute path in the sandbox
            content: File content (string or bytes)
        """
        pass

    @abstractmethod
    def read_file(self, path: str) -> str | bytes:
        """
        Read a file from the sandbox filesystem.

        Args:
            path: Absolute path in the sandbox

        Returns:
            File content
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanup sandbox resources."""
        pass


class E2BBackend(SandboxBackend):
    """
    E2B cloud sandbox backend.

    Uses the E2B API for code execution. Requires E2B_API_KEY environment variable.
    No local Docker required - execution happens in E2B's cloud infrastructure.

    Features:
    - Firecracker microVM isolation
    - Fast startup (~200ms in same region)
    - Up to 24h sessions
    - Python, JavaScript, and shell execution
    """

    def __init__(self, template: str = "base", timeout: int = 300):
        """
        Initialize E2B backend.

        Args:
            template: E2B sandbox template (default: "base")
            timeout: Sandbox timeout in seconds (default: 300)
        """
        self._template = template
        self._timeout = timeout
        self._sandbox = None

    def start(self) -> None:
        """Initialize the E2B sandbox."""
        if not HAS_E2B:
            raise ImportError("e2b_code_interpreter not installed. Run: pip install e2b-code-interpreter")

        logger.info(f"Starting E2B sandbox (template={self._template}, timeout={self._timeout})")
        self._sandbox = E2BSandbox(template=self._template, timeout=self._timeout)
        logger.info(f"E2B sandbox started: {self._sandbox.id}")

    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in the E2B sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        result = self._sandbox.run_code(code)
        return ExecutionResult(
            stdout=result.text or "",
            stderr=str(result.error) if result.error else "",
            exit_code=0 if not result.error else 1,
        )

    def run_command(self, command: str) -> ExecutionResult:
        """Execute a shell command in the E2B sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        result = self._sandbox.commands.run(command)
        return ExecutionResult(stdout=result.stdout or "", stderr=result.stderr or "", exit_code=result.exit_code)

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file to the E2B sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        self._sandbox.files.write(path, content)

    def read_file(self, path: str) -> str | bytes:
        """Read a file from the E2B sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        return self._sandbox.files.read(path)

    def close(self) -> None:
        """Shutdown the E2B sandbox."""
        if self._sandbox is not None:
            logger.info(f"Closing E2B sandbox: {self._sandbox.id}")
            self._sandbox.kill()
            self._sandbox = None


class DockerBackend(SandboxBackend):
    """
    Local Docker backend using the ``docker`` Python SDK.

    Runs code in a Docker container on the local machine.
    Requires Docker to be running and the ``docker`` pip package installed.

    Features:
    - Full OS capabilities
    - File I/O via exec
    - Bash/shell execution
    """

    def __init__(self, image: str = "ubuntu:24.04"):
        """
        Initialize Docker backend.

        Args:
            image: Docker image to use (default: ubuntu:24.04)
        """
        self._image = image
        self._container = None
        self._client = None

    def start(self) -> None:
        """Start the Docker container."""
        if not HAS_DOCKER:
            raise ImportError("docker SDK not installed. Run: pip install docker")

        logger.info(f"Starting Docker container (image={self._image})")
        # Reuse client to avoid leaking connections across reset() calls
        if self._client is None:
            self._client = docker_sdk.from_env()
        self._container = self._client.containers.run(self._image, command="sleep infinity", detach=True, remove=True)
        logger.info(f"Docker container started: {self._container.short_id}")

    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in the Docker container."""
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")

        filename = f"/tmp/code_{uuid.uuid4().hex}.py"
        self.write_file(filename, code)

        try:
            if language == "python":
                return self.run_command(f"python {filename}")
            else:
                return self.run_command(code)
        finally:
            self.run_command(f"rm -f {filename}")

    def run_command(self, command: str) -> ExecutionResult:
        """Execute a shell command in the Docker container."""
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")

        exit_code, output = self._container.exec_run(["bash", "-c", command], demux=True)
        stdout = (output[0] or b"").decode("utf-8", errors="replace") if output else ""
        stderr = (output[1] or b"").decode("utf-8", errors="replace") if output else ""
        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file to the Docker container via exec."""
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")

        if isinstance(content, bytes):
            encoded_content = base64.b64encode(content).decode("ascii")
            self._container.exec_run(["bash", "-c", f"echo '{encoded_content}' | base64 -d > {path}"])
        else:
            # Use base64 encoding to safely transfer arbitrary text content
            encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
            self._container.exec_run(["bash", "-c", f"echo '{encoded_content}' | base64 -d > {path}"])

    def read_file(self, path: str) -> str | bytes:
        """Read a file from the Docker container."""
        if self._container is None:
            raise RuntimeError("Container not started. Call start() first.")

        exit_code, output = self._container.exec_run(["cat", path])
        return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output

    def close(self) -> None:
        """Stop and remove the Docker container."""
        if self._container is not None:
            logger.info(f"Closing Docker container: {self._container.short_id}")
            with contextlib.suppress(Exception):
                self._container.stop(timeout=5)
            with contextlib.suppress(Exception):
                self._container.kill()
            self._container = None


def _daytona_retry(fn, *args, max_retries: int = 3, **kwargs):
    """Call a Daytona SDK function with retry on transient errors."""
    import time

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Daytona call {fn.__qualname__} attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2**attempt)
            else:
                raise


class DaytonaBackend(SandboxBackend):
    """
    Daytona cloud sandbox backend.

    Uses the Daytona SDK for code execution in cloud sandboxes.
    Requires DAYTONA_API_KEY environment variable or explicit api_key.

    Features:
    - Cloud-based sandbox isolation
    - Full OS capabilities
    - File I/O and shell execution
    - Automatic retry on transient errors
    """

    def __init__(
        self,
        image: str = "ubuntu:24.04",
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        timeout: int = 300,
        resources: dict | None = None,
    ):
        """
        Initialize Daytona backend.

        Args:
            image: Container image to use (default: ubuntu:24.04)
            api_key: Daytona API key (default: from DAYTONA_API_KEY env var)
            api_url: Daytona API URL (optional)
            target: Daytona target (optional)
            timeout: Sandbox timeout in seconds (default: 300)
            resources: Resource configuration dict (optional)
        """
        self._image = image
        self._api_key = api_key
        self._api_url = api_url
        self._target = target
        self._timeout = timeout
        self._resources = resources
        self._sandbox = None
        self._daytona = None

    def start(self) -> None:
        """Initialize the Daytona sandbox."""
        if not HAS_DAYTONA:
            raise ImportError("daytona not installed. Run: pip install daytona")

        from daytona import CreateSandboxFromImageParams

        config_kwargs = {}
        if self._api_key is not None:
            config_kwargs["api_key"] = self._api_key
        if self._api_url is not None:
            config_kwargs["api_url"] = self._api_url
        if self._target is not None:
            config_kwargs["target"] = self._target

        config = DaytonaConfig(**config_kwargs) if config_kwargs else None

        logger.info(f"Starting Daytona sandbox (image={self._image}, timeout={self._timeout})")
        self._daytona = Daytona(config) if config else Daytona()

        params = CreateSandboxFromImageParams(
            image=self._image,
            auto_stop_interval=10,  # minutes — auto-stop if idle (leak protection)
            auto_archive_interval=0,  # don't archive, just stop
        )
        self._sandbox = _daytona_retry(self._daytona.create, params, timeout=self._timeout)
        logger.info("Daytona sandbox started")

    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in the Daytona sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        response = _daytona_retry(self._sandbox.process.code_run, code=code)
        return ExecutionResult(
            stdout=response.result or "", stderr=getattr(response, "stderr", "") or "", exit_code=response.exit_code
        )

    def run_command(self, command: str) -> ExecutionResult:
        """Execute a shell command in the Daytona sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        response = _daytona_retry(self._sandbox.process.exec, command)
        return ExecutionResult(
            stdout=response.result or "", stderr=getattr(response, "stderr", "") or "", exit_code=response.exit_code
        )

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file to the Daytona sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        if isinstance(content, str):
            content = content.encode("utf-8")
        _daytona_retry(self._sandbox.fs.upload_file, content, path)

    def read_file(self, path: str) -> str:
        """Read a file from the Daytona sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        data = _daytona_retry(self._sandbox.fs.download_file, path)
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return data

    def close(self) -> None:
        """Delete the Daytona sandbox."""
        if self._sandbox is not None:
            logger.info("Closing Daytona sandbox")
            with contextlib.suppress(Exception):
                self._daytona.delete(self._sandbox)
            self._sandbox = None
            self._daytona = None


def create_backend(backend_type: str, **kwargs) -> SandboxBackend:
    """
    Factory function to create a sandbox backend.

    Args:
        backend_type: "e2b", "docker", or "daytona"
        **kwargs: Backend-specific arguments

    Returns:
        SandboxBackend instance (not yet started)
    """
    if backend_type == "e2b":
        return E2BBackend(**kwargs)
    elif backend_type == "docker":
        return DockerBackend(**kwargs)
    elif backend_type == "daytona":
        return DaytonaBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Must be 'e2b', 'docker', or 'daytona'")
