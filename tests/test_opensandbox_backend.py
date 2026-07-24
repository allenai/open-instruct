"""Unit tests for OpenSandboxBackend.

These tests don't require the ``opensandbox`` package or a running
OpenSandbox service — they patch the SDK names inside ``backends`` with test
doubles and assert the backend drives the OpenSandbox API correctly and
honors the abstract SandboxBackend contract.
"""

from __future__ import annotations

import io
import itertools
import os
import tarfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from open_instruct.environments import backends
from open_instruct.environments.backends import _OPENSANDBOX_LIVE_SANDBOXES, OpenSandboxBackend, create_backend


class _FakeOpenSandboxException(Exception):
    pass


def _make_execution(exit_code: int | None = 0, stdout: str = "", stderr: str = "", error=None):
    logs = SimpleNamespace(
        stdout=[SimpleNamespace(text=stdout)] if stdout else [],
        stderr=[SimpleNamespace(text=stderr)] if stderr else [],
    )
    return SimpleNamespace(exit_code=exit_code, logs=logs, error=error)


class _FakeConnectionConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRunCommandOpts:
    def __init__(self, timeout: timedelta | None = None):
        self.timeout = timeout


class _FakeWriteEntry:
    def __init__(self, path: str, data=None, mode: int = 755):
        self.path = path
        self.data = data
        self.mode = mode


class _FakeSandboxState:
    PENDING = "Pending"
    RUNNING = "Running"
    TERMINATED = "Terminated"


class _FakeSandboxFilter:
    def __init__(self, states=None, metadata=None, page=None, page_size=None):
        self.states = states
        self.metadata = metadata
        self.page = page
        self.page_size = page_size


class _FakeManager:
    def __init__(self, fake: _FakeOpenSandboxSync):
        self.fake = fake
        self.killed: list[str] = []
        self.closed = False
        self.last_filter: _FakeSandboxFilter | None = None

    def list_sandbox_infos(self, filter):
        self.last_filter = filter
        infos = self.fake.adoption_pages.pop(0) if self.fake.adoption_pages else []
        return SimpleNamespace(sandbox_infos=infos)

    def kill_sandbox(self, sandbox_id: str) -> None:
        self.killed.append(sandbox_id)

    def close(self) -> None:
        self.closed = True


class _FakeManagerFactory:
    def __init__(self, fake: _FakeOpenSandboxSync):
        self.fake = fake

    def create(self, connection_config=None) -> _FakeManager:
        manager = _FakeManager(self.fake)
        self.fake.managers.append(manager)
        return manager


class _FakeCommands:
    def __init__(self, sandbox: _FakeSandbox):
        self._sandbox = sandbox
        self.run_calls: list[dict] = []

    def run(self, command: str, *, opts=None, handlers=None):
        self.run_calls.append({"command": command, "opts": opts})
        result = self._sandbox.fake.route_exec(command)
        if isinstance(result, Exception):
            raise result
        return result if result is not None else _make_execution()


class _FakeFiles:
    def __init__(self):
        self.write_calls: list[list[_FakeWriteEntry]] = []
        self.write_exceptions: list[Exception] = []  # raised (and consumed) by successive write_files() calls
        self.read_bytes_results: dict[str, bytes | Exception] = {}

    def write_files(self, entries):
        if self.write_exceptions:
            raise self.write_exceptions.pop(0)
        self.write_calls.append(entries)

    def read_bytes(self, path: str) -> bytes:
        result = self.read_bytes_results.get(path)
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise _FakeOpenSandboxException(f"no scripted content for {path}")
        return result


class _FakeSandbox:
    _ids = itertools.count()

    def __init__(self, fake: _FakeOpenSandboxSync, create_kwargs: dict):
        self.fake = fake
        self.create_kwargs = create_kwargs
        self.id = f"osb-{next(self._ids)}"
        self.state = _FakeSandboxState.RUNNING
        self.killed = False
        self.closed = False
        self.kill_exceptions: list[Exception] = []  # raised (and consumed) by successive kill() calls
        self.commands = _FakeCommands(self)
        self.files = _FakeFiles()

    def get_info(self):
        return SimpleNamespace(status=SimpleNamespace(state=self.state))

    def kill(self) -> None:
        if self.kill_exceptions:
            raise self.kill_exceptions.pop(0)
        self.killed = True
        self.state = _FakeSandboxState.TERMINATED

    def close(self) -> None:
        self.closed = True


class _FakeOpenSandboxSync:
    """Test double for the ``opensandbox`` SDK's ``SandboxSync``.

    ``script_for_substring`` maps a substring of the exec command to a fake
    execution (or an exception to raise); unmatched commands succeed with
    empty output. One-shot scripts are consumed on first match.
    """

    def __init__(self):
        self.sandboxes: list[_FakeSandbox] = []
        self.create_exceptions: list[Exception] = []  # raised (and consumed) by successive create() calls
        self.adoption_pages: list[list] = []  # successive list_sandbox_infos results for _FakeManager
        self.managers: list[_FakeManager] = []
        self.connect_calls: list[str] = []
        self._scripted: list[dict] = []

    def create(self, image, **kwargs):
        if self.create_exceptions:
            raise self.create_exceptions.pop(0)
        sandbox = _FakeSandbox(self, {"image": image, **kwargs})
        self.sandboxes.append(sandbox)
        return sandbox

    def connect(self, sandbox_id, connection_config=None, **kwargs):
        self.connect_calls.append(sandbox_id)
        sandbox = _FakeSandbox(self, {"connected": sandbox_id})
        sandbox.id = sandbox_id
        self.sandboxes.append(sandbox)
        return sandbox

    def script_for_substring(self, substring: str, result: object, once: bool = False) -> None:
        self._scripted.append({"substring": substring, "result": result, "once": once})

    def route_exec(self, command: str) -> object | None:
        for entry in self._scripted:
            if entry["substring"] in command:
                if entry["once"]:
                    self._scripted.remove(entry)
                return entry["result"]
        return None


class OpenSandboxBackendTestCase(unittest.TestCase):
    """Base class that installs the fake SDK names and cleans global state."""

    def setUp(self):
        self.fake = _FakeOpenSandboxSync()
        for name, replacement in [
            ("OpenSandboxSync", self.fake),
            ("OpenSandboxConnectionConfig", _FakeConnectionConfig),
            ("OpenSandboxException", _FakeOpenSandboxException),
            ("OpenSandboxRunCommandOpts", _FakeRunCommandOpts),
            ("OpenSandboxWriteEntry", _FakeWriteEntry),
            ("OpenSandboxManagerSync", _FakeManagerFactory(self.fake)),
            ("OpenSandboxFilter", _FakeSandboxFilter),
            ("OpenSandboxState", _FakeSandboxState),
        ]:
            patcher = patch.object(backends, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SWERL_OPENSANDBOX_")}
        env["SWERL_OPENSANDBOX_DOMAIN"] = "sandbox.test:8080"
        env_patcher = patch.dict(os.environ, env, clear=True)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        self.addCleanup(_OPENSANDBOX_LIVE_SANDBOXES.clear)

    def _started_backend(self, **kwargs) -> OpenSandboxBackend:
        kwargs.setdefault("image", "python:3.12-slim")
        backend = OpenSandboxBackend(**kwargs)
        backend.start()
        return backend


class TestFactory(OpenSandboxBackendTestCase):
    def test_create_backend_opensandbox(self):
        backend = create_backend("opensandbox", image="python:3.12-slim")
        self.assertIsInstance(backend, OpenSandboxBackend)

    def test_create_backend_unknown_mentions_opensandbox(self):
        with self.assertRaisesRegex(ValueError, "opensandbox"):
            create_backend("bogus")


class TestInitRequirements(unittest.TestCase):
    def test_init_raises_without_opensandbox_package(self):
        with patch.object(backends, "OpenSandboxSync", None), self.assertRaisesRegex(RuntimeError, "opensandbox"):
            OpenSandboxBackend()

    def test_init_raises_without_domain(self):
        env = {k: v for k, v in os.environ.items() if k != "SWERL_OPENSANDBOX_DOMAIN"}
        with (
            patch.object(backends, "OpenSandboxSync", _FakeOpenSandboxSync()),
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(RuntimeError, "SWERL_OPENSANDBOX_DOMAIN"),
        ):
            OpenSandboxBackend()


class TestStartAndClose(OpenSandboxBackendTestCase):
    def test_start_creates_sandbox_with_expected_resources(self):
        backend = self._started_backend(
            image="ubuntu:22.04", mem_limit="2g", cpu=0.5, sandbox_lifetime=1200, ready_timeout=90
        )
        [sandbox] = self.fake.sandboxes
        self.assertEqual(sandbox.create_kwargs["image"], "ubuntu:22.04")
        self.assertEqual(sandbox.create_kwargs["timeout"], timedelta(seconds=1200))
        self.assertEqual(sandbox.create_kwargs["ready_timeout"], timedelta(seconds=90))
        self.assertEqual(sandbox.create_kwargs["resource"], {"cpu": "0.5", "memory": "2048Mi"})
        self.assertEqual(sandbox.create_kwargs["metadata"]["open_instruct"], "swerl_sandbox")
        self.assertIn(sandbox, _OPENSANDBOX_LIVE_SANDBOXES)
        backend.close()

    def test_start_builds_connection_config_with_stretched_request_timeout(self):
        backend = self._started_backend(timeout=600)
        [sandbox] = self.fake.sandboxes
        config = sandbox.create_kwargs["connection_config"]
        self.assertEqual(config.kwargs["domain"], "sandbox.test:8080")
        self.assertEqual(config.kwargs["request_timeout"], timedelta(seconds=600 + 300))
        backend.close()

    def test_start_uses_env_defaults(self):
        with patch.dict(
            os.environ,
            {
                "SWERL_OPENSANDBOX_CPU": "2.0",
                "SWERL_OPENSANDBOX_LIFETIME_S": "999",
                "SWERL_OPENSANDBOX_APP_NAME": "my-app",
            },
        ):
            backend = self._started_backend()
        [sandbox] = self.fake.sandboxes
        self.assertEqual(sandbox.create_kwargs["resource"]["cpu"], "2.0")
        self.assertEqual(sandbox.create_kwargs["timeout"], timedelta(seconds=999))
        self.assertEqual(sandbox.create_kwargs["metadata"]["open_instruct_app"], "my-app")
        backend.close()

    def test_close_kills_and_is_idempotent(self):
        backend = self._started_backend()
        [sandbox] = self.fake.sandboxes
        backend.close()
        self.assertTrue(sandbox.killed)
        self.assertTrue(sandbox.closed)
        self.assertNotIn(sandbox, _OPENSANDBOX_LIVE_SANDBOXES)
        backend.close()  # No sandbox anymore; must be a no-op.

    def test_close_retries_kill_once(self):
        backend = self._started_backend()
        [sandbox] = self.fake.sandboxes
        sandbox.kill_exceptions = [RuntimeError("transient")]
        backend.close()
        self.assertTrue(sandbox.killed)

    def test_close_survives_kill_failing_twice(self):
        backend = self._started_backend()
        [sandbox] = self.fake.sandboxes
        sandbox.kill_exceptions = [RuntimeError("boom"), RuntimeError("boom again")]
        backend.close()  # Must not raise.
        self.assertFalse(sandbox.killed)
        self.assertNotIn(sandbox, _OPENSANDBOX_LIVE_SANDBOXES)

    def test_start_after_start_replaces_sandbox(self):
        backend = self._started_backend()
        backend.start()
        first, second = self.fake.sandboxes
        self.assertTrue(first.killed)
        self.assertFalse(second.killed)
        self.assertNotIn(first, _OPENSANDBOX_LIVE_SANDBOXES)
        self.assertIn(second, _OPENSANDBOX_LIVE_SANDBOXES)
        backend.close()


class TestRunCommand(OpenSandboxBackendTestCase):
    def test_run_command_wraps_with_timeout_and_bash(self):
        backend = self._started_backend(timeout=120)
        self.fake.script_for_substring("echo hi", _make_execution(exit_code=0, stdout="hi\n"))
        result = backend.run_command("echo hi")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "hi\n")
        [sandbox] = self.fake.sandboxes
        [call] = sandbox.commands.run_calls
        self.assertIn("timeout --signal=TERM --kill-after=10 120", call["command"])
        self.assertIn("bash -c", call["command"])
        self.assertEqual(call["opts"].timeout, timedelta(seconds=120 + 60))

    def test_run_command_per_call_timeout_overrides_default(self):
        backend = self._started_backend(timeout=1800)
        backend.run_command("true", timeout=30)
        [sandbox] = self.fake.sandboxes
        [call] = sandbox.commands.run_calls
        self.assertIn("timeout --signal=TERM --kill-after=10 30", call["command"])
        self.assertEqual(call["opts"].timeout, timedelta(seconds=30 + 60))

    def test_run_command_annotates_timeout_exit_code(self):
        backend = self._started_backend(timeout=5)
        self.fake.script_for_substring("sleep", _make_execution(exit_code=124))
        result = backend.run_command("sleep 100")
        self.assertEqual(result.exit_code, 124)
        self.assertIn("timed out after 5s", result.stderr)

    def test_run_command_raises_if_not_started(self):
        backend = OpenSandboxBackend()
        with self.assertRaisesRegex(RuntimeError, "not started"):
            backend.run_command("echo hi")

    def test_run_command_surfaces_missing_exit_code_as_failure(self):
        backend = self._started_backend()
        error = SimpleNamespace(name="SpawnError", value="cannot start process")
        self.fake.script_for_substring("doomed", _make_execution(exit_code=None, error=error))
        result = backend.run_command("doomed")
        self.assertEqual(result.exit_code, -1)
        self.assertIn("SpawnError", result.stderr)
        self.assertIn("cannot start process", result.stderr)

    def test_run_command_restarts_and_retries_once_when_sandbox_died(self):
        backend = self._started_backend()
        [first] = self.fake.sandboxes
        first.state = _FakeSandboxState.TERMINATED
        self.fake.script_for_substring("echo hi", _FakeOpenSandboxException("sandbox gone"), once=True)
        self.fake.script_for_substring("echo hi", _make_execution(exit_code=0, stdout="hi\n"))
        result = backend.run_command("echo hi")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "hi\n")
        self.assertEqual(len(self.fake.sandboxes), 2)
        self.assertTrue(first.killed or first.state == _FakeSandboxState.TERMINATED)

    def test_run_command_does_not_restart_on_transient_error_while_alive(self):
        backend = self._started_backend()
        self.fake.script_for_substring("echo hi", _FakeOpenSandboxException("network blip"))
        with self.assertRaisesRegex(_FakeOpenSandboxException, "network blip"):
            backend.run_command("echo hi")
        # The sandbox was alive, so no replacement must have been created.
        self.assertEqual(len(self.fake.sandboxes), 1)


class TestFileIO(OpenSandboxBackendTestCase):
    def test_write_file_creates_dirs_and_uploads_bytes(self):
        backend = self._started_backend()
        backend.write_file("/workspace/sub/hello.txt", "hello")
        [sandbox] = self.fake.sandboxes
        [mkdir_call] = sandbox.commands.run_calls
        self.assertIn("mkdir -p /workspace/sub", mkdir_call["command"])
        [[entry]] = sandbox.files.write_calls
        self.assertEqual(entry.path, "/workspace/sub/hello.txt")
        self.assertEqual(entry.data, b"hello")

    def test_write_file_accepts_bytes(self):
        backend = self._started_backend()
        backend.write_file("/tmp/blob.bin", b"\x00\x01")
        [sandbox] = self.fake.sandboxes
        [[entry]] = sandbox.files.write_calls
        self.assertEqual(entry.data, b"\x00\x01")

    def test_write_file_raises_on_mkdir_failure(self):
        backend = self._started_backend()
        self.fake.script_for_substring("mkdir -p", _make_execution(exit_code=1, stderr="denied"))
        with self.assertRaisesRegex(RuntimeError, "denied"):
            backend.write_file("/etc/nope/file.txt", "x")

    def test_write_file_raises_on_upload_failure(self):
        backend = self._started_backend()
        [sandbox] = self.fake.sandboxes
        sandbox.files.write_exceptions = [_FakeOpenSandboxException("upload broke")]
        with self.assertRaisesRegex(RuntimeError, "upload broke"):
            backend.write_file("/tmp/file.txt", "x")

    def test_read_file_raises_not_found(self):
        backend = self._started_backend()
        self.fake.script_for_substring("exit 40", _make_execution(exit_code=40))
        with self.assertRaises(FileNotFoundError):
            backend.read_file("/no/such/file")

    def test_read_file_raises_is_a_directory(self):
        backend = self._started_backend()
        self.fake.script_for_substring("exit 40", _make_execution(exit_code=41))
        with self.assertRaises(IsADirectoryError):
            backend.read_file("/tmp")

    def test_read_file_returns_str_then_bytes(self):
        backend = self._started_backend()
        [sandbox] = self.fake.sandboxes
        sandbox.files.read_bytes_results["/tmp/data.txt"] = b"content"
        self.assertEqual(backend.read_file("/tmp/data.txt"), "content")
        self.assertEqual(backend.read_file("/tmp/data.txt", binary=True), b"content")

    def test_read_file_wraps_filesystem_errors(self):
        backend = self._started_backend()
        [sandbox] = self.fake.sandboxes
        sandbox.files.read_bytes_results["/tmp/data.txt"] = _FakeOpenSandboxException("download broke")
        with self.assertRaisesRegex(RuntimeError, "download broke"):
            backend.read_file("/tmp/data.txt")

    def test_put_archive_uploads_then_extracts(self):
        backend = self._started_backend()
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            info = tarfile.TarInfo(name="hello.txt")
            payload = b"hello"
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        tar_bytes = tar_stream.getvalue()

        backend.put_archive("/workspace", tar_bytes)
        [sandbox] = self.fake.sandboxes
        [[entry]] = sandbox.files.write_calls
        self.assertEqual(entry.data, tar_bytes)
        [extract_call] = sandbox.commands.run_calls
        self.assertIn(f"tar -xf {entry.path} -C /workspace", extract_call["command"])
        self.assertIn("rm -f", extract_call["command"])

    def test_put_archive_raises_on_extract_failure(self):
        backend = self._started_backend()
        self.fake.script_for_substring("tar -xf", _make_execution(exit_code=2, stderr="bad archive"))
        with self.assertRaisesRegex(RuntimeError, "bad archive"):
            backend.put_archive("/workspace", b"not a tar")


class TestAdoptOnGatewayTimeout(OpenSandboxBackendTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(OpenSandboxBackend, "_ADOPT_POLL_INTERVAL_S", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _gateway_timeout_error() -> _FakeOpenSandboxException:
        error = _FakeOpenSandboxException("Create sandbox failed: HTTP 504 | request_id=abc123")
        error.status_code = 504
        return error

    @staticmethod
    def _info(sandbox_id: str, state: str = _FakeSandboxState.RUNNING):
        return SimpleNamespace(id=sandbox_id, status=SimpleNamespace(state=state))

    def test_start_adopts_sandbox_after_gateway_timeout(self):
        self.fake.create_exceptions = [self._gateway_timeout_error()]
        # First poll sees nothing (pod still provisioning), second finds it.
        self.fake.adoption_pages = [[], [self._info("orphan-1")]]
        backend = OpenSandboxBackend()
        backend.start()
        self.assertEqual(self.fake.connect_calls, ["orphan-1"])
        [manager] = self.fake.managers
        self.assertTrue(manager.closed)
        self.assertIn("open_instruct_create_id", manager.last_filter.metadata)
        # The adopted sandbox must be fully usable.
        result = backend.run_command("echo adopted")
        self.assertEqual(result.exit_code, 0)
        backend.close()

    def test_adoption_filter_matches_the_create_id_sent_to_create(self):
        self.fake.create_exceptions = [self._gateway_timeout_error()]
        self.fake.adoption_pages = [[self._info("orphan-1")]]
        backend = OpenSandboxBackend()
        backend.start()
        # create() raised, so grab the id from the filter and check its shape:
        # a fresh uuid4 hex per start().
        [manager] = self.fake.managers
        create_id = manager.last_filter.metadata["open_instruct_create_id"]
        self.assertEqual(len(create_id), 32)
        int(create_id, 16)  # must be valid hex

    def test_start_kills_duplicate_pods_when_adopting(self):
        self.fake.create_exceptions = [self._gateway_timeout_error()]
        self.fake.adoption_pages = [
            [self._info("orphan-1"), self._info("orphan-2"), self._info("orphan-3", state=_FakeSandboxState.PENDING)]
        ]
        backend = OpenSandboxBackend()
        backend.start()
        self.assertEqual(self.fake.connect_calls, ["orphan-1"])
        [manager] = self.fake.managers
        self.assertEqual(sorted(manager.killed), ["orphan-2", "orphan-3"])

    def test_start_reraises_when_no_sandbox_appears(self):
        self.fake.create_exceptions = [self._gateway_timeout_error()]
        backend = OpenSandboxBackend(ready_timeout=0)
        with self.assertRaisesRegex(_FakeOpenSandboxException, "HTTP 504"):
            backend.start()
        self.assertEqual(self.fake.connect_calls, [])
        [manager] = self.fake.managers
        self.assertTrue(manager.closed)

    def test_start_raises_non_timeout_errors_without_adoption(self):
        self.fake.create_exceptions = [_FakeOpenSandboxException("HTTP 403 Forbidden")]
        backend = OpenSandboxBackend()
        with self.assertRaisesRegex(_FakeOpenSandboxException, "403"):
            backend.start()
        self.assertEqual(self.fake.managers, [])

    def test_is_gateway_timeout_classification(self):
        self.assertTrue(OpenSandboxBackend._is_gateway_timeout(self._gateway_timeout_error()))
        self.assertTrue(
            OpenSandboxBackend._is_gateway_timeout(RuntimeError("Create sandbox failed: HTTP 504 | request_id=x"))
        )
        self.assertFalse(OpenSandboxBackend._is_gateway_timeout(RuntimeError("HTTP 503 upstream unavailable")))


if __name__ == "__main__":
    unittest.main()
