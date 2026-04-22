"""End-to-end smoke test for ``ApptainerBackend``.

Run this inside an ``salloc`` slot on a cluster that has ``apptainer`` on PATH
(e.g. Tillicum) to validate that the backend works with a given image before
wiring it into a full training run.

What it checks:
    1. ``apptainer`` binary is present and usable.
    2. Instance start succeeds with ``--fakeroot --writable-tmpfs``.
    3. ``run_command`` roundtrips stdout/stderr/exit codes.
    4. ``write_file`` + ``read_file`` on paths under ``/workspace``.
    5. ``put_archive`` extracts a tar into the instance.
    6. Filesystem state persists across ``run_command`` calls
       (the core requirement for rollouts).
    7. Timeout wrapping fires at the configured budget.
    8. Clean shutdown; instance is removed from ``apptainer instance list``.

Usage:
    uv run python scripts/debug/apptainer_backend_smoke.py \\
        --image docker://ubuntu:22.04 \\
        --cache-dir /scr/$USER/apptainer-cache \\
        --tmp-dir   /scr/$USER/apptainer-tmp

Exit code is 0 on success, 1 if any check failed.
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from contextlib import contextmanager

from open_instruct.environments.backends import ApptainerBackend


OK = "\033[32mOK\033[0m"
FAIL = "\033[31mFAIL\033[0m"


@contextmanager
def _step(name: str, results: list[tuple[str, bool, float]]):
    """Context manager that records pass/fail + timing for a named step."""
    print(f"--- {name}")
    t0 = time.monotonic()
    try:
        yield
    except Exception:
        elapsed = time.monotonic() - t0
        print(traceback.format_exc())
        print(f"  [{FAIL}] {name}  ({elapsed:.2f}s)")
        results.append((name, False, elapsed))
        return
    elapsed = time.monotonic() - t0
    print(f"  [{OK}] {name}  ({elapsed:.2f}s)")
    results.append((name, True, elapsed))


def _make_tar(files: dict[str, bytes]) -> bytes:
    """Build a tar archive in memory from ``{relative_path: bytes}``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, data in files.items():
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _instance_exists(name: str) -> bool:
    """Return True if ``apptainer instance list`` knows about ``name``."""
    proc = subprocess.run(
        ["apptainer", "instance", "list", name],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0 and name.encode() in proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="docker://ubuntu:22.04",
        help="Image reference: docker://repo:tag, plain repo:tag, or path to .sif",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Passed as APPTAINER_CACHEDIR; should be fast shared storage",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Passed as APPTAINER_TMPDIR",
    )
    parser.add_argument(
        "--pwd",
        default="/workspace",
        help="Default cwd inside the container for run_command",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-command timeout in seconds (deliberately small so the "
        "timeout-check step is fast)",
    )
    parser.add_argument(
        "--keep-instance-on-failure",
        action="store_true",
        help="Skip the final ``close()`` on any failure, for post-mortem debugging",
    )
    args = parser.parse_args()

    if shutil.which("apptainer") is None:
        print(f"[{FAIL}] apptainer binary not found on PATH")
        return 1

    print(f"Image:       {args.image}")
    print(f"Cache dir:   {args.cache_dir or '(default)'}")
    print(f"Tmp dir:     {args.tmp_dir or '(default)'}")
    print(f"Timeout:     {args.timeout}s")
    print()

    results: list[tuple[str, bool, float]] = []
    backend = ApptainerBackend(
        image=args.image,
        timeout=args.timeout,
        pwd=args.pwd,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
    )

    with _step("start instance", results):
        backend.start()
        assert backend._name is not None
        assert _instance_exists(backend._name), "instance not visible to apptainer"

    if not all(ok for _, ok, _ in results):
        print("\nstart failed; aborting")
        return 1

    try:
        with _step("run_command: echo roundtrip", results):
            r = backend.run_command("echo hello-stdout; echo err >&2; exit 0")
            assert r.exit_code == 0, r
            assert "hello-stdout" in r.stdout, r
            assert "err" in r.stderr, r

        with _step("run_command: uid is 0 under fakeroot", results):
            r = backend.run_command("id -u")
            assert r.exit_code == 0, r
            assert r.stdout.strip() == "0", r

        with _step("run_command: APPTAINER_PWD takes effect", results):
            r = backend.run_command("pwd")
            assert r.exit_code == 0, r
            # Backend defaulted pwd to /workspace; create it first in case the
            # image doesn't ship with the directory.
            backend.run_command("mkdir -p /workspace")
            r = backend.run_command("pwd")
            assert r.stdout.strip() == args.pwd, f"expected {args.pwd}, got {r.stdout!r}"

        with _step("write_file + read_file (text)", results):
            backend.run_command("mkdir -p /workspace")
            backend.write_file("/workspace/hello.txt", "hello smoke\n")
            got = backend.read_file("/workspace/hello.txt")
            assert got == "hello smoke\n", repr(got)

        with _step("write_file + read_file (bytes)", results):
            payload = bytes(range(256))
            backend.write_file("/workspace/bin", payload)
            got = backend.read_file("/workspace/bin", binary=True)
            assert got == payload, "binary roundtrip mismatch"

        with _step("read_file: missing => FileNotFoundError", results):
            try:
                backend.read_file("/workspace/definitely-does-not-exist")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("expected FileNotFoundError")

        with _step("put_archive: extracts at /", results):
            tar_bytes = _make_tar(
                {
                    "tests/test.sh": b"#!/bin/sh\necho PASSED\n",
                    "tests/helper.txt": b"aux\n",
                }
            )
            backend.put_archive("/", tar_bytes)
            r = backend.run_command("cat /tests/test.sh && cat /tests/helper.txt")
            assert r.exit_code == 0, r
            assert "PASSED" in r.stdout and "aux" in r.stdout, r

        with _step("state persists across run_command calls", results):
            backend.run_command("echo persisted > /workspace/state.txt")
            r = backend.run_command("cat /workspace/state.txt")
            assert r.exit_code == 0, r
            assert r.stdout.strip() == "persisted", r.stdout

        with _step(
            f"timeout wrapper fires at {args.timeout}s (quick check with sleep)",
            results,
        ):
            # Sleep twice the budget; expect timeout exit code 124.
            sleep_for = max(args.timeout + 2, 3)
            t0 = time.monotonic()
            r = backend.run_command(f"sleep {sleep_for}")
            elapsed = time.monotonic() - t0
            assert r.exit_code == 124, f"expected 124 (timeout), got {r.exit_code}: {r}"
            assert "timed out" in r.stderr.lower(), r.stderr
            # Must not have waited the full sleep duration.
            assert elapsed < sleep_for, (
                f"timeout did not fire early: elapsed={elapsed:.1f}s, "
                f"sleep={sleep_for}s"
            )
    finally:
        name = backend._name
        any_failed = not all(ok for _, ok, _ in results)
        if any_failed and args.keep_instance_on_failure and name is not None:
            print(
                f"\n--keep-instance-on-failure set; leaving {name} running. "
                f"Stop it manually with: apptainer instance stop {name}"
            )
        else:
            with _step("close instance", results):
                backend.close()
                if name is not None:
                    assert not _instance_exists(name), (
                        f"instance {name} still exists after close()"
                    )

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    total_time = sum(dt for _, _, dt in results)
    print(f"Summary: {passed}/{total} checks passed in {total_time:.2f}s")
    if passed != total:
        for name, ok, dt in results:
            if not ok:
                print(f"  [{FAIL}] {name}  ({dt:.2f}s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
