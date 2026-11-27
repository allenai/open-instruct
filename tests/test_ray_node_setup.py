"""CPU regressions for ray_node_setup.sh exit codes; ray and host lookup are replaced locally."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "configs/beaker_configs/ray_node_setup.sh"

RAY = """#!/bin/bash
case "$1" in
  stop) exit 0 ;;
  start)
    [[ "$2" == --head ]] && exit 0
    [[ "$START_FAILS" == 1 ]] && exit 1
    exit 0 ;;
  status)
    seen=$(($(cat "$STATUS_COUNT" 2>/dev/null || echo 0) + 1))
    echo "$seen" > "$STATUS_COUNT"
    [[ "$seen" -le "$STATUS_OK_TIMES" ]] && exit 0
    exit 1 ;;
esac
exit 0
"""

# An empty name makes the real `getent hosts` list every host, which is the #2205 condition.
GETENT = """#!/bin/bash
[[ -z "$2" ]] && { printf '127.0.0.1 localhost\n172.17.0.6 container\n'; exit 0; }
printf '10.0.0.5 %s\n' "$2"
"""


def executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


def run_script(tmp_path, rank=None, leader="leader", start_fails="0", status_ok_times="0"):
    executable(tmp_path / "ray", RAY)
    executable(tmp_path / "getent", GETENT)
    executable(tmp_path / "python", "#!/bin/sh\necho 110926_120000\n")
    executable(tmp_path / "sleep", "#!/bin/sh\nexit 0\n")
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "BEAKER_LEADER_REPLICA_HOSTNAME": leader,
        "START_FAILS": start_fails,
        "STATUS_OK_TIMES": status_ok_times,
        "STATUS_COUNT": str(tmp_path / "status-count"),
    }
    env.pop("BEAKER_REPLICA_RANK", None)
    if rank is not None:
        env["BEAKER_REPLICA_RANK"] = rank
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize(
    "leader,start_fails,status_ok_times,expected,message",
    [
        # A spec without `replicas` leaves BEAKER_REPLICA_RANK unset, so the task takes the
        # worker branch with no head to join. It must not report success.
        ("", "0", "0", 1, "missing 'replicas'"),
        ("leader", "1", "0", 1, "Failed to join"),
        # `ray start` can exit 0 without this node being registered with the head.
        ("leader", "0", "0", 1, "is not reachable"),
        # Joined and confirmed, then the head goes away: the normal end of a multi-node run.
        ("leader", "0", "1", 0, "Monitoring Ray head"),
    ],
)
def test_worker_exit_code(tmp_path, leader, start_fails, status_ok_times, expected, message):
    result = run_script(
        tmp_path, leader=leader, start_fails=start_fails, status_ok_times=status_ok_times
    )
    output = result.stdout + result.stderr
    assert result.returncode == expected, output
    assert message in output, output


def test_head_replica_is_unaffected(tmp_path):
    result = run_script(tmp_path, rank="0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Starting Ray head node" in result.stdout
