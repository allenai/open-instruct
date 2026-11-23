"""The router TCP_USER_TIMEOUT shim is preloaded into sgl-router only and drops just that option."""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

SHIM_DIR = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "miles" / "router_shim"


def _preload_module():
    spec = importlib.util.spec_from_file_location("oi_router_preload", SHIM_DIR / "oi_router_preload.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Not the router: importing must not re-exec pytest.
    return module


@pytest.fixture
def shim(tmp_path):
    path = tmp_path / "no_tcp_user_timeout.so"
    path.write_bytes(b"")
    return str(path)


ROUTER = ["/usr/bin/python3", "-m", "sglang_router.launch_router", "--port", "8000"]


@pytest.mark.parametrize(
    "argv",
    [
        ROUTER,
        ["python", "-u", "-m", "sglang_router.launch_router"],
        ["python", "-msglang_router.launch_router"],
        ["python", "-X", "faulthandler", "-m", "sglang_router.launch_router"],
    ],
)
def test_router_reexecs_with_shim_first(shim, argv):
    module = _preload_module()
    environ = {"LD_PRELOAD": "/other.so"}
    assert module.preload_argv(argv, environ, shim) == [sys.executable, *argv[1:]]
    assert environ["LD_PRELOAD"] == f"{shim}:/other.so"
    # The re-executed process already has the shim and must not loop.
    assert module.preload_argv(argv, environ, shim) is None


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-m", "sglang.launch_server"],
        ["python", "-m", "miles.router.router"],
        ["python", "script.py", "-m", "sglang_router.launch_router"],
        ["python", "-c", "pass", "-m", "sglang_router.launch_router"],
        ["python", "-c", "pass"],
        [],
    ],
)
def test_other_processes_untouched(shim, argv):
    environ: dict[str, str] = {}
    assert _preload_module().preload_argv(argv, environ, shim) is None
    assert environ == {}


def test_opt_out_and_missing_shim(tmp_path, shim):
    module = _preload_module()
    environ = {"OI_MILES_ROUTER_SHIM": "0"}
    assert module.preload_argv(ROUTER, environ, shim) is None
    assert "LD_PRELOAD" not in environ
    assert module.preload_argv(ROUTER, {}, str(tmp_path / "absent.so")) is None


@pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("cc") is None, reason="needs Linux and cc")
def test_shim_drops_only_tcp_user_timeout(tmp_path):
    library = tmp_path / "no_tcp_user_timeout.so"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Werror",
            "-o",
            library,
            SHIM_DIR / "no_tcp_user_timeout.c",
            "-ldl",
        ],
        check=True,
    )
    probe = (
        "import socket; s = socket.socket(); t = socket.IPPROTO_TCP\n"
        "s.setsockopt(t, socket.TCP_USER_TIMEOUT, 30000); s.setsockopt(t, socket.TCP_KEEPIDLE, 45)\n"
        "print(s.getsockopt(t, socket.TCP_USER_TIMEOUT), s.getsockopt(t, socket.TCP_KEEPIDLE))"
    )
    plain = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert plain.stdout.split() == ["30000", "45"]
    shimmed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LD_PRELOAD": str(library)},
    )
    assert shimmed.stdout.split() == ["0", "45"]
    assert "dropped TCP_USER_TIMEOUT=30000 ms" in shimmed.stderr
