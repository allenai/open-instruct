"""Container interface selection and fallback inheritance, without GPUs."""

import json
import os
import subprocess
import sys

import pytest

from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import preflight_network


@pytest.mark.parametrize(
    "selector,expected",
    [
        ("ib", ["ib0", "ib1"]),
        ("eth,ib", ["eth0", "eth1", "ib0", "ib1"]),
        ("=eth0", ["eth0"]),
        ("=eth0,ib1", ["eth0", "ib1"]),
        ("^ib", ["lo", "eth0", "eth1"]),
        ("^=eth0,ib1", ["lo", "eth1", "ib0"]),
        ("missing", []),
        ("^lo,eth,ib", []),
    ],
)
def test_nccl_selectors(selector, expected):
    assert preflight_network.matching_interfaces(selector, ["lo", "eth0", "eth1", "ib0", "ib1"]) == expected


def test_single_node_bridge_fallback_warns_and_removes_override(monkeypatch, caplog):
    monkeypatch.setattr(preflight_network.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ib")
    preflight_network.check(replicas=1, network_mode="bridge")
    assert "NCCL_SOCKET_IFNAME" not in os.environ
    assert "NCCL_SOCKET_IFNAME='ib'" in caplog.text
    assert "eth0" in caplog.text
    assert "automatically" in caplog.text


@pytest.mark.parametrize("selector", [None, "", "ib", "=ib0", "^eth", "eth,ib"])
def test_valid_or_unset_overrides_are_preserved(monkeypatch, caplog, selector):
    monkeypatch.setattr(preflight_network.socket, "if_nameindex", lambda: [(1, "lo"), (2, "ib0")])
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    if selector is not None:
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", selector)
    preflight_network.check(replicas=1, network_mode="bridge")
    assert os.environ.get("NCCL_SOCKET_IFNAME") == selector
    assert not caplog.records


@pytest.mark.parametrize(
    "selector,replicas,mode",
    [
        ("ib", 2, "host"),
        ("ib", 1, "host"),
        ("ib", 2, "bridge"),
        ("=ib0", 1, "bridge"),
        ("typo", 1, "bridge"),
        ("^lo,eth", 1, "bridge"),
    ],
)
def test_other_unmatched_overrides_fail_without_mutation(monkeypatch, selector, replicas, mode):
    monkeypatch.setattr(preflight_network.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", selector)
    with pytest.raises(InputError, match="matches no container interfaces.*eth0"):
        preflight_network.check(replicas=replicas, network_mode=mode)
    assert os.environ["NCCL_SOCKET_IFNAME"] == selector


@pytest.mark.parametrize("mode,exit_code", [("bridge", 7), ("host", 2)])
def test_exec_inherits_fallback_and_preserves_exit_status(mode, exit_code):
    # The child is a real process: catch a preflight that only changes its own
    # environment and accidentally starts training with the original override.
    bootstrap = (
        "from open_instruct.miles.execution import preflight_network; "
        "preflight_network.socket.if_nameindex = lambda: [(1, 'lo'), (2, 'eth0')]; "
        "preflight_network.main()"
    )
    workload = (
        "import json, os, sys; "
        "print(json.dumps({key: os.environ[key] for key in "
        "('NCCL_SOCKET_IFNAME', 'PREFLIGHT_TEST_SENTINEL') if key in os.environ})); sys.exit(7)"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            bootstrap,
            "--replicas",
            "1",
            "--network-mode",
            mode,
            "--",
            sys.executable,
            "-c",
            workload,
        ],
        env={**os.environ, "NCCL_SOCKET_IFNAME": "ib", "PREFLIGHT_TEST_SENTINEL": "preserved"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == exit_code
    if mode == "bridge":
        environment = json.loads(result.stdout)
        assert "NCCL_SOCKET_IFNAME" not in environment
        assert environment["PREFLIGHT_TEST_SENTINEL"] == "preserved"
        assert "WARNING" in result.stderr
    else:
        assert result.stdout == ""
        assert "matches no container interfaces" in result.stderr
