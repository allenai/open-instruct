"""Sandbox backend abstraction for code/command execution."""

import base64
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
    from llm_in_sandbox import DockerRuntime

    HAS_LLM_IN_SANDBOX = True
except ImportError:
    DockerRuntime = None
    HAS_LLM_IN_SANDBOX = False

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
    Local Docker backend using llm-in-sandbox.

    Runs code in a Docker container on the local machine.
    Requires Docker to be installed and running.

    Features:
    - Full OS capabilities
    - File I/O with mounted directories
    - Bash/shell execution
    """

    def __init__(self, image: str = "cdx123/llm-in-sandbox:v0.1"):
        """
        Initialize Docker backend.

        Args:
            image: Docker image to use (default: llm-in-sandbox)
        """
        self._image = image
        self._runtime = None

    def start(self) -> None:
        """Start the Docker container."""
        if not HAS_LLM_IN_SANDBOX:
            raise ImportError("llm_in_sandbox not installed. Run: pip install llm-in-sandbox")

        logger.info(f"Starting Docker container (image={self._image})")
        self._runtime = DockerRuntime(docker_image=self._image)
        logger.info("Docker container started")

    def run_code(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in the Docker container."""
        if self._runtime is None:
            raise RuntimeError("Container not started. Call start() first.")

        filename = f"/tmp/code_{uuid.uuid4().hex}.py"
        self._runtime.run(f"cat > {filename} << 'CODEEOF'\n{code}\nCODEEOF")

        try:
            if language == "python":
                output, exit_code = self._runtime.run(f"python {filename}")
            else:
                output, exit_code = self._runtime.run(code)
        finally:
            self._runtime.run(f"rm -f {filename}")

        return ExecutionResult(stdout=output, stderr="", exit_code=0 if str(exit_code) == "0" else 1)

    def run_command(self, command: str) -> ExecutionResult:
        """Execute a shell command in the Docker container."""
        if self._runtime is None:
            raise RuntimeError("Container not started. Call start() first.")

        output, exit_code = self._runtime.run(command)
        return ExecutionResult(stdout=output, stderr="", exit_code=0 if str(exit_code) == "0" else 1)

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file to the Docker container."""
        if self._runtime is None:
            raise RuntimeError("Container not started. Call start() first.")

        if isinstance(content, bytes):
            encoded_content = base64.b64encode(content).decode("ascii")
            self._runtime.run(f"echo '{encoded_content}' | base64 -d > {path}")
        else:
            self._runtime.run(f"cat > {path} << 'FILEEOF'\n{content}\nFILEEOF")

    def read_file(self, path: str) -> str | bytes:
        """Read a file from the Docker container."""
        if self._runtime is None:
            raise RuntimeError("Container not started. Call start() first.")

        output, _ = self._runtime.run(f"cat {path}")
        return output

    def close(self) -> None:
        """Stop and remove the Docker container."""
        if self._runtime is not None:
            logger.info("Closing Docker container")
            self._runtime.close()
            self._runtime = None


def create_backend(backend_type: str, **kwargs) -> SandboxBackend:
    """
    Factory function to create a sandbox backend.

    Args:
        backend_type: "e2b" or "docker"
        **kwargs: Backend-specific arguments

    Returns:
        SandboxBackend instance (not yet started)
    """
    if backend_type == "e2b":
        return E2BBackend(**kwargs)
    elif backend_type == "docker":
        return DockerBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Must be 'e2b' or 'docker'")
