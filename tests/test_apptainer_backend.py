"""Unit tests for ApptainerBackend.

These tests don't require an actual Apptainer installation — they mock
``subprocess.run`` and ``shutil.which`` and assert the backend assembles the
right command lines and honors the abstract SandboxBackend contract.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
import unittest
from unittest.mock import patch

from open_instruct.environments.backends import (
    ApptainerBackend,
    _APPTAINER_LIVE_INSTANCES,
    _normalize_apptainer_image,
    create_backend,
)


def _completed(
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class _FakeApptainer:
    """Test double that records every ``subprocess.run`` invocation and lets
    tests script the return values / stdout / stderr per call.

    Call sites are matched by the argv prefix via ``script_for_prefix``.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self._scripted: list[tuple[list[str], subprocess.CompletedProcess]] = []

    def script_for_prefix(
        self, prefix: list[str], result: subprocess.CompletedProcess
    ) -> None:
        self._scripted.append((prefix, result))

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append({"argv": list(argv), **kwargs})
        for prefix, result in self._scripted:
            if list(argv[: len(prefix)]) == prefix:
                return result
        return _completed()


def _make_backend(**kwargs) -> ApptainerBackend:
    kwargs.setdefault("image", "docker://ubuntu:22.04")
    return ApptainerBackend(**kwargs)


def _started_backend(fake: _FakeApptainer, **kwargs) -> ApptainerBackend:
    """Start a backend with ``subprocess.run`` and ``shutil.which`` patched."""
    backend = _make_backend(**kwargs)
    with (
        patch("open_instruct.environments.backends.subprocess.run", side_effect=fake),
        patch("open_instruct.environments.backends.shutil.which", return_value="/usr/bin/apptainer"),
    ):
        backend.start()
    return backend


class TestImageNormalization(unittest.TestCase):
    def test_plain_tag_gets_docker_prefix(self):
        self.assertEqual(_normalize_apptainer_image("ubuntu:22.04"), "docker://ubuntu:22.04")
        self.assertEqual(
            _normalize_apptainer_image("hamishi740/swerl-tmax-v3:abc123"),
            "docker://hamishi740/swerl-tmax-v3:abc123",
        )

    def test_uri_passes_through(self):
        self.assertEqual(_normalize_apptainer_image("docker://foo:bar"), "docker://foo:bar")
        self.assertEqual(_normalize_apptainer_image("oras://registry.io/x:y"), "oras://registry.io/x:y")

    def test_paths_pass_through(self):
        self.assertEqual(_normalize_apptainer_image("/shared/images/foo.sif"), "/shared/images/foo.sif")
        self.assertEqual(_normalize_apptainer_image("./local.sif"), "./local.sif")
        self.assertEqual(_normalize_apptainer_image("bare.sif"), "bare.sif")


class TestFactory(unittest.TestCase):
    def test_create_backend_apptainer(self):
        backend = create_backend("apptainer", image="docker://ubuntu:22.04")
        self.assertIsInstance(backend, ApptainerBackend)

    def test_create_backend_unknown(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend type"):
            create_backend("podman")


class TestStartAndClose(unittest.TestCase):
    def test_start_invokes_instance_start_with_expected_flags(self):
        fake = _FakeApptainer()
        backend = _started_backend(fake)
        try:
            start_calls = [c for c in fake.calls if c["argv"][:3] == ["apptainer", "instance", "start"]]
            self.assertEqual(len(start_calls), 1)
            argv = start_calls[0]["argv"]
            # Key isolation flags must be present — these define the sandbox.
            for flag in ("--fakeroot", "--writable-tmpfs", "--containall", "--no-home", "--cleanenv"):
                self.assertIn(flag, argv)
            # Image is normalized and comes after flags.
            self.assertIn("docker://ubuntu:22.04", argv)
            # Instance name is last and identifies this backend.
            self.assertTrue(argv[-1].startswith("swerl-"))
            self.assertIn(argv[-1], _APPTAINER_LIVE_INSTANCES)
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()

    def test_start_propagates_apptainer_pwd_and_cache(self):
        fake = _FakeApptainer()
        backend = _started_backend(
            fake, pwd="/workspace", cache_dir="/scratch/cache", tmp_dir="/scratch/tmp"
        )
        try:
            env = fake.calls[0]["env"]
            self.assertEqual(env["APPTAINER_PWD"], "/workspace")
            self.assertEqual(env["APPTAINER_CACHEDIR"], "/scratch/cache")
            self.assertEqual(env["APPTAINER_TMPDIR"], "/scratch/tmp")
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()

    def test_start_raises_when_binary_missing(self):
        backend = _make_backend()
        with patch("open_instruct.environments.backends.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "not found on PATH"):
                backend.start()

    def test_start_raises_when_apptainer_fails(self):
        fake = _FakeApptainer()
        fake.script_for_prefix(
            ["apptainer", "instance", "start"],
            _completed(stderr=b"FATAL: image pull failed", returncode=255),
        )
        backend = _make_backend()
        with (
            patch("open_instruct.environments.backends.subprocess.run", side_effect=fake),
            patch("open_instruct.environments.backends.shutil.which", return_value="/usr/bin/apptainer"),
        ):
            with self.assertRaisesRegex(RuntimeError, "apptainer instance start failed"):
                backend.start()

    def test_close_is_idempotent(self):
        fake = _FakeApptainer()
        backend = _started_backend(fake)
        with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
            backend.close()
            backend.close()  # second call must not raise
        self.assertIsNone(backend._name)

    def test_close_removes_instance_from_live_set(self):
        fake = _FakeApptainer()
        backend = _started_backend(fake)
        name = backend._name
        with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
            backend.close()
        self.assertNotIn(name, _APPTAINER_LIVE_INSTANCES)

    def test_start_after_start_replaces_instance(self):
        fake = _FakeApptainer()
        backend = _started_backend(fake)
        first_name = backend._name
        with (
            patch("open_instruct.environments.backends.subprocess.run", side_effect=fake),
            patch("open_instruct.environments.backends.shutil.which", return_value="/usr/bin/apptainer"),
        ):
            backend.start()
        self.assertNotEqual(backend._name, first_name)
        self.assertNotIn(first_name, _APPTAINER_LIVE_INSTANCES)
        self.assertIn(backend._name, _APPTAINER_LIVE_INSTANCES)
        with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
            backend.close()


class TestRunCommand(unittest.TestCase):
    def test_run_command_wraps_with_timeout_and_bash(self):
        fake = _FakeApptainer()
        fake.script_for_prefix(
            ["apptainer", "exec"],
            _completed(stdout=b"hello\n", stderr=b""),
        )
        backend = _started_backend(fake, timeout=42)
        try:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                result = backend.run_command("echo hello")
            exec_calls = [c for c in fake.calls if c["argv"][:2] == ["apptainer", "exec"]]
            self.assertTrue(exec_calls)
            argv = exec_calls[-1]["argv"]
            self.assertEqual(argv[:3], ["apptainer", "exec", f"instance://{backend._name}"])
            self.assertEqual(argv[3:5], ["bash", "-c"])
            # Timeout wrapper surrounds the user command.
            self.assertIn("timeout --signal=TERM --kill-after=10 42", argv[5])
            self.assertIn("echo hello", argv[5])
            self.assertEqual(result.stdout, "hello\n")
            self.assertEqual(result.exit_code, 0)
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()

    def test_run_command_annotates_timeout_exit_code(self):
        fake = _FakeApptainer()
        fake.script_for_prefix(
            ["apptainer", "exec"],
            _completed(returncode=124, stderr=b""),
        )
        backend = _started_backend(fake, timeout=5)
        try:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                result = backend.run_command("sleep 100")
            self.assertEqual(result.exit_code, 124)
            self.assertIn("timed out after 5s", result.stderr)
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()

    def test_run_command_raises_if_not_started(self):
        backend = _make_backend()
        with self.assertRaisesRegex(RuntimeError, "not started"):
            backend.run_command("echo hi")


class TestFileIO(unittest.TestCase):
    def test_write_file_pipes_content_via_cat(self):
        fake = _FakeApptainer()
        backend = _started_backend(fake)
        try:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.write_file("/workspace/foo.txt", "hello world")
            write_calls = [
                c for c in fake.calls
                if c["argv"][:2] == ["apptainer", "exec"] and "cat >" in " ".join(c["argv"])
            ]
            self.assertEqual(len(write_calls), 1)
            call = write_calls[0]
            self.assertEqual(call["input"], b"hello world")
            # Must mkdir -p the parent and write to the exact path.
            # ``shlex.quote`` leaves simple paths unquoted.
            shell = call["argv"][-1]
            self.assertIn("mkdir -p /workspace", shell)
            self.assertIn("cat > /workspace/foo.txt", shell)
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()

    def test_write_file_accepts_bytes(self):
        fake = _FakeApptainer()
        backend = _started_backend(fake)
        try:
            payload = b"\x00\x01\x02binary"
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.write_file("/workspace/bin", payload)
            write_calls = [c for c in fake.calls if "cat >" in " ".join(c["argv"])]
            self.assertEqual(write_calls[-1]["input"], payload)
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()

    def test_read_file_raises_not_found(self):
        fake = _FakeApptainer()
        # test -e returns non-zero => file missing
        fake.script_for_prefix(["apptainer", "exec"], _completed(returncode=1))
        backend = _started_backend(fake)
        try:
            with (
                patch("open_instruct.environments.backends.subprocess.run", side_effect=fake),
                self.assertRaises(FileNotFoundError),
            ):
                backend.read_file("/no/such")
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()

    def test_read_file_returns_str_then_bytes(self):
        fake = _FakeApptainer()
        script = {
            "test -e": _completed(returncode=0),
            "test -d": _completed(returncode=1),  # not a directory
            "cat": _completed(stdout=b"hello\xe2\x98\x83", returncode=0),
        }

        def router(argv, **kwargs):
            fake.calls.append({"argv": list(argv), **kwargs})
            if argv[:2] == ["apptainer", "exec"]:
                tail = argv[3:]
                if tail[:1] == ["sh"] and "test -e" in tail[-1]:
                    return script["test -e"]
                if tail[:1] == ["sh"] and "test -d" in tail[-1]:
                    return script["test -d"]
                if tail[:1] == ["cat"]:
                    return script["cat"]
            return _completed()

        backend = _started_backend(fake)
        try:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=router):
                text = backend.read_file("/workspace/out.txt")
                raw = backend.read_file("/workspace/out.txt", binary=True)
            self.assertEqual(text, "hello\u2603")
            self.assertEqual(raw, b"hello\xe2\x98\x83")
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()

    def test_put_archive_streams_tar_to_exec(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name="workspace/hello.txt")
            data = b"hi"
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_bytes = buf.getvalue()

        fake = _FakeApptainer()
        backend = _started_backend(fake)
        try:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.put_archive("/", tar_bytes)
            tar_calls = [
                c for c in fake.calls
                if c["argv"][:2] == ["apptainer", "exec"] and "tar" in c["argv"]
            ]
            self.assertEqual(len(tar_calls), 1)
            argv = tar_calls[0]["argv"]
            self.assertEqual(argv[-5:], ["tar", "-xf", "-", "-C", "/"])
            self.assertEqual(tar_calls[0]["input"], tar_bytes)
        finally:
            with patch("open_instruct.environments.backends.subprocess.run", side_effect=fake):
                backend.close()


if __name__ == "__main__":
    unittest.main()
