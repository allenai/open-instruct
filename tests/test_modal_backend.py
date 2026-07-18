"""Unit tests for ModalBackend.

These tests don't require the ``modal`` package or Modal credentials — they
patch the ``modal`` module reference inside ``backends`` with a test double
and assert the backend drives the Modal Sandbox API correctly and honors the
abstract SandboxBackend contract.
"""

from __future__ import annotations

import io
import itertools
import os
import tarfile
import unittest
from unittest.mock import patch

from open_instruct.environments import backends
from open_instruct.environments.backends import (
    _MODAL_LIVE_SANDBOXES,
    ModalBackend,
    create_backend,
    parse_mem_limit_mib,
)


class _FakeStream:
    def __init__(self, data: bytes = b""):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeStdin:
    def __init__(self):
        self.written = b""
        self.eof = False
        self.drained = False

    def write(self, data: bytes) -> None:
        self.written += data

    def write_eof(self) -> None:
        self.eof = True

    def drain(self) -> None:
        self.drained = True


class _FakeProcess:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


class _FakeSandbox:
    _ids = itertools.count()

    def __init__(self, fake_modal: _FakeModal, create_kwargs: dict):
        self._fake_modal = fake_modal
        self.create_kwargs = create_kwargs
        self.object_id = f"sb-{next(self._ids)}"
        self.exec_calls: list[dict] = []
        self.terminated = False
        self.terminate_exceptions: list[Exception] = []  # raised (and consumed) by successive terminate() calls

    def exec(self, *argv, **kwargs):
        call = {"argv": list(argv), **kwargs}
        self.exec_calls.append(call)
        result = self._fake_modal.route_exec(self, list(argv))
        if isinstance(result, Exception):
            raise result
        process = result or _FakeProcess()
        call["process"] = process
        return process

    def terminate(self) -> None:
        if self.terminate_exceptions:
            raise self.terminate_exceptions.pop(0)
        self.terminated = True


class _FakeModalExceptions:
    class SandboxTerminatedError(Exception):
        pass

    class SandboxTimeoutError(Exception):
        pass

    class RemoteError(Exception):
        pass


class _FakeModal:
    """Test double for the ``modal`` module.

    ``script_for_substring`` maps a substring of the joined exec argv to a
    ``_FakeProcess`` (or an exception to raise); unmatched execs succeed with
    empty output.
    """

    exception = _FakeModalExceptions

    def __init__(self):
        self.sandboxes: list[_FakeSandbox] = []
        self.lookup_calls: list[tuple[str, bool, str | None]] = []
        self.from_registry_calls: list[tuple[str, str | None]] = []
        self.create_failures: dict[str, Exception] = {}  # image value -> exception
        self._scripted: list[tuple[str, object]] = []
        fake = self

        class App:
            @staticmethod
            def lookup(name: str, create_if_missing: bool = False, environment_name: str | None = None):
                fake.lookup_calls.append((name, create_if_missing, environment_name))
                return f"app:{name}"

        class Image:
            @staticmethod
            def from_registry(tag: str, add_python: str | None = None):
                fake.from_registry_calls.append((tag, add_python))
                return f"image:{tag}:python={add_python}"

        class Sandbox:
            @staticmethod
            def create(**kwargs):
                failure = fake.create_failures.get(kwargs.get("image"))
                if failure is not None:
                    raise failure
                sandbox = _FakeSandbox(fake, kwargs)
                fake.sandboxes.append(sandbox)
                return sandbox

        self.App = App
        self.Image = Image
        self.Sandbox = Sandbox

    def script_for_substring(self, substring: str, result: object) -> None:
        self._scripted.append((substring, result))

    def route_exec(self, sandbox: _FakeSandbox, argv: list[str]) -> object | None:
        joined = " ".join(argv)
        for substring, result in self._scripted:
            if substring in joined:
                return result
        return None


class ModalBackendTestCase(unittest.TestCase):
    """Base class that installs the fake modal module and cleans global state."""

    def setUp(self):
        self.fake = _FakeModal()
        patcher = patch.object(backends, "modal", self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        backends._MODAL_APPS.clear()
        backends._MODAL_IMAGES_NEEDING_PYTHON.clear()
        self.addCleanup(backends._MODAL_APPS.clear)
        self.addCleanup(backends._MODAL_IMAGES_NEEDING_PYTHON.clear)
        self.addCleanup(_MODAL_LIVE_SANDBOXES.clear)

    def _started_backend(self, **kwargs) -> ModalBackend:
        kwargs.setdefault("image", "python:3.12-slim")
        backend = ModalBackend(**kwargs)
        backend.start()
        return backend


class TestMemLimitParse(unittest.TestCase):
    def test_docker_style_strings(self):
        self.assertEqual(parse_mem_limit_mib("4g"), 4096)
        self.assertEqual(parse_mem_limit_mib("4gb"), 4096)
        self.assertEqual(parse_mem_limit_mib("512m"), 512)
        self.assertEqual(parse_mem_limit_mib("2048k"), 2)
        self.assertEqual(parse_mem_limit_mib("1073741824"), 1024)
        self.assertEqual(parse_mem_limit_mib("1.5g"), 1536)

    def test_int_bytes(self):
        self.assertEqual(parse_mem_limit_mib(4 * 1024 * 1024 * 1024), 4096)

    def test_none_uses_default(self):
        self.assertEqual(parse_mem_limit_mib(None), 4096)
        self.assertEqual(parse_mem_limit_mib(None, default_mib=2048), 2048)

    def test_invalid_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot parse memory limit"):
            parse_mem_limit_mib("lots")


class TestFactory(ModalBackendTestCase):
    def test_create_backend_modal(self):
        backend = create_backend("modal", image="python:3.12-slim")
        self.assertIsInstance(backend, ModalBackend)

    def test_create_backend_unknown_mentions_modal(self):
        with self.assertRaisesRegex(ValueError, "'modal'"):
            create_backend("podman")


class TestInitRequiresModal(unittest.TestCase):
    def test_init_raises_without_modal_package(self):
        with patch.object(backends, "modal", None), self.assertRaisesRegex(RuntimeError, "pip install modal"):
            ModalBackend()


class TestStartAndClose(ModalBackendTestCase):
    def test_start_creates_sandbox_with_expected_resources(self):
        backend = self._started_backend(
            image="ubuntu:22.04",
            mem_limit="2g",
            cpu=2.0,
            app_name="my-app",
            environment_name="staging",
            sandbox_lifetime=1234,
        )
        self.assertEqual(self.fake.lookup_calls, [("my-app", True, "staging")])
        self.assertEqual(self.fake.from_registry_calls, [("ubuntu:22.04", None)])
        sandbox = self.fake.sandboxes[0]
        self.assertEqual(sandbox.create_kwargs["app"], "app:my-app")
        self.assertEqual(sandbox.create_kwargs["image"], "image:ubuntu:22.04:python=None")
        self.assertEqual(sandbox.create_kwargs["timeout"], 1234)
        self.assertEqual(sandbox.create_kwargs["cpu"], 2.0)
        self.assertEqual(sandbox.create_kwargs["memory"], 2048)
        self.assertIn(sandbox, _MODAL_LIVE_SANDBOXES)
        backend.close()

    def test_start_uses_default_app_name_and_environment(self):
        backend = self._started_backend()
        self.assertEqual(self.fake.lookup_calls[0], ("open-instruct-sandbox", True, "agent-training"))
        backend.close()

    def test_environment_name_defaults_from_env_var(self):
        with patch.dict(os.environ, {"SWERL_MODAL_ENVIRONMENT": "rl-prod"}):
            backend = self._started_backend()
        self.assertEqual(self.fake.lookup_calls[0][2], "rl-prod")
        backend.close()

    def test_app_lookup_cache_is_keyed_by_environment(self):
        backend_a = self._started_backend(environment_name="dev")
        backend_b = self._started_backend(environment_name="prod")
        self.assertEqual([call[2] for call in self.fake.lookup_calls], ["dev", "prod"])
        backend_a.close()
        backend_b.close()

    def test_app_lookup_is_cached_per_process(self):
        backend = self._started_backend()
        backend.start()
        self.assertEqual(len(self.fake.lookup_calls), 1)
        backend.close()

    def test_close_retries_terminate_once(self):
        backend = self._started_backend()
        sandbox = self.fake.sandboxes[0]
        sandbox.terminate_exceptions = [ConnectionError("transient")]
        backend.close()
        self.assertTrue(sandbox.terminated)
        self.assertNotIn(sandbox, _MODAL_LIVE_SANDBOXES)
        self.assertIsNone(backend._sandbox)

    def test_close_survives_terminate_failing_twice(self):
        backend = self._started_backend()
        sandbox = self.fake.sandboxes[0]
        sandbox.terminate_exceptions = [ConnectionError("down"), ConnectionError("still down")]
        backend.close()  # must not raise
        self.assertFalse(sandbox.terminated)
        self.assertNotIn(sandbox, _MODAL_LIVE_SANDBOXES)
        self.assertIsNone(backend._sandbox)

    def test_close_terminates_and_is_idempotent(self):
        backend = self._started_backend()
        sandbox = self.fake.sandboxes[0]
        backend.close()
        backend.close()  # second call must not raise
        self.assertTrue(sandbox.terminated)
        self.assertNotIn(sandbox, _MODAL_LIVE_SANDBOXES)
        self.assertIsNone(backend._sandbox)

    def test_start_falls_back_to_add_python_on_image_build_failure(self):
        # The plain (no add_python) build fails, mimicking an image with no
        # Python/pip; the add_python build succeeds.
        self.fake.create_failures["image:no-python-img:latest:python=None"] = _FakeModalExceptions.RemoteError(
            "Image build for im-123 failed. See build logs for more details."
        )
        backend = self._started_backend(image="no-python-img:latest")
        self.assertEqual(
            self.fake.from_registry_calls, [("no-python-img:latest", None), ("no-python-img:latest", "3.12")]
        )
        self.assertEqual(self.fake.sandboxes[0].create_kwargs["image"], "image:no-python-img:latest:python=3.12")
        self.assertIn("no-python-img:latest", backends._MODAL_IMAGES_NEEDING_PYTHON)
        # The fallback must repair python3 resolution: the standalone
        # interpreter would otherwise shadow the image's own python3.
        repair_calls = [
            c for c in self.fake.sandboxes[0].exec_calls if "rm -f /usr/local/bin/python" in " ".join(c["argv"])
        ]
        self.assertEqual(len(repair_calls), 1)
        backend.close()

        # A second backend for the same tag skips the doomed plain attempt.
        self.fake.from_registry_calls.clear()
        backend2 = self._started_backend(image="no-python-img:latest")
        self.assertEqual(self.fake.from_registry_calls, [("no-python-img:latest", "3.12")])
        backend2.close()

    def test_start_without_fallback_skips_python_repair(self):
        backend = self._started_backend()
        self.assertEqual(self.fake.sandboxes[0].exec_calls, [])
        backend.close()

    def test_start_propagates_non_build_remote_errors(self):
        self.fake.create_failures["image:ubuntu:22.04:python=None"] = _FakeModalExceptions.RemoteError(
            "workspace has reached its container limit"
        )
        backend = ModalBackend(image="ubuntu:22.04")
        with self.assertRaisesRegex(_FakeModalExceptions.RemoteError, "container limit"):
            backend.start()
        self.assertNotIn("ubuntu:22.04", backends._MODAL_IMAGES_NEEDING_PYTHON)

    def test_start_after_start_replaces_sandbox(self):
        backend = self._started_backend()
        first = self.fake.sandboxes[0]
        backend.start()
        second = self.fake.sandboxes[1]
        self.assertTrue(first.terminated)
        self.assertNotIn(first, _MODAL_LIVE_SANDBOXES)
        self.assertIn(second, _MODAL_LIVE_SANDBOXES)
        self.assertIs(backend._sandbox, second)
        backend.close()


class TestRunCommand(ModalBackendTestCase):
    def test_run_command_wraps_with_timeout_and_bash(self):
        self.fake.script_for_substring("echo hello", _FakeProcess(stdout=b"hello\n"))
        backend = self._started_backend(timeout=42)
        result = backend.run_command("echo hello")
        call = self.fake.sandboxes[0].exec_calls[-1]
        self.assertEqual(call["argv"][:2], ["bash", "-c"])
        self.assertIn("timeout --signal=TERM --kill-after=10 42", call["argv"][2])
        self.assertIn("echo hello", call["argv"][2])
        self.assertFalse(call["text"])
        # stdin must be closed even when nothing is piped in.
        self.assertTrue(call["process"].stdin.eof)
        self.assertEqual(result.stdout, "hello\n")
        self.assertEqual(result.exit_code, 0)
        backend.close()

    def test_run_command_annotates_timeout_exit_code(self):
        self.fake.script_for_substring("sleep 100", _FakeProcess(returncode=124))
        backend = self._started_backend(timeout=5)
        result = backend.run_command("sleep 100")
        self.assertEqual(result.exit_code, 124)
        self.assertIn("timed out after 5s", result.stderr)
        backend.close()

    def test_run_command_raises_if_not_started(self):
        backend = ModalBackend()
        with self.assertRaisesRegex(RuntimeError, "not started"):
            backend.run_command("echo hi")

    def test_run_command_restarts_and_retries_once_when_sandbox_died(self):
        backend = self._started_backend()
        first = self.fake.sandboxes[0]

        # The first sandbox is dead: every exec on it raises. The replacement
        # sandbox created by the retry path succeeds.
        def route(sandbox, argv):
            if sandbox is first:
                return _FakeModalExceptions.SandboxTerminatedError("sandbox terminated")
            return _FakeProcess(stdout=b"hi\n")

        self.fake.route_exec = route
        result = backend.run_command("echo hi")
        self.assertEqual(result.stdout, "hi\n")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(self.fake.sandboxes), 2)
        self.assertTrue(first.terminated)
        backend.close()


class TestFileIO(ModalBackendTestCase):
    def test_write_file_pipes_content_via_cat(self):
        backend = self._started_backend()
        backend.write_file("/workspace/foo.txt", "hello world")
        call = self.fake.sandboxes[0].exec_calls[-1]
        self.assertEqual(call["argv"][:2], ["sh", "-c"])
        # Must mkdir -p the parent and write to the exact path.
        self.assertIn("mkdir -p /workspace", call["argv"][2])
        self.assertIn("cat > /workspace/foo.txt", call["argv"][2])
        self.assertEqual(call["process"].stdin.written, b"hello world")
        self.assertTrue(call["process"].stdin.eof)
        backend.close()

    def test_write_file_accepts_bytes(self):
        backend = self._started_backend()
        payload = b"\x00\x01\x02binary"
        backend.write_file("/workspace/bin", payload)
        call = self.fake.sandboxes[0].exec_calls[-1]
        self.assertEqual(call["process"].stdin.written, payload)
        backend.close()

    def test_write_file_raises_on_failure(self):
        self.fake.script_for_substring("cat >", _FakeProcess(returncode=1, stderr=b"disk full"))
        backend = self._started_backend()
        with self.assertRaisesRegex(RuntimeError, "write_file failed.*disk full"):
            backend.write_file("/workspace/foo.txt", "x")
        backend.close()

    def test_read_file_raises_not_found(self):
        self.fake.script_for_substring("cat /no/such", _FakeProcess(returncode=40))
        backend = self._started_backend()
        with self.assertRaises(FileNotFoundError):
            backend.read_file("/no/such")
        backend.close()

    def test_read_file_raises_is_a_directory(self):
        self.fake.script_for_substring("cat /workspace", _FakeProcess(returncode=41))
        backend = self._started_backend()
        with self.assertRaises(IsADirectoryError):
            backend.read_file("/workspace")
        backend.close()

    def test_read_file_returns_str_then_bytes(self):
        self.fake.script_for_substring("cat /workspace/out.txt", _FakeProcess(stdout=b"hello\xe2\x98\x83"))
        backend = self._started_backend()
        self.assertEqual(backend.read_file("/workspace/out.txt"), "hello☃")
        self.assertEqual(backend.read_file("/workspace/out.txt", binary=True), b"hello\xe2\x98\x83")
        backend.close()

    def test_put_archive_streams_tar_to_exec(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name="workspace/hello.txt")
            data = b"hi"
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_bytes = buf.getvalue()

        backend = self._started_backend()
        backend.put_archive("/", tar_bytes)
        call = self.fake.sandboxes[0].exec_calls[-1]
        self.assertEqual(call["argv"], ["tar", "-xf", "-", "-C", "/"])
        self.assertEqual(call["process"].stdin.written, tar_bytes)
        backend.close()

    def test_put_archive_raises_on_failure(self):
        self.fake.script_for_substring("tar -xf", _FakeProcess(returncode=2, stderr=b"corrupt archive"))
        backend = self._started_backend()
        with self.assertRaisesRegex(RuntimeError, "put_archive failed.*corrupt archive"):
            backend.put_archive("/", b"not a tar")
        backend.close()


if __name__ == "__main__":
    unittest.main()
