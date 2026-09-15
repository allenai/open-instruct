"""Allocate fresh coordination state for each complete replica attempt."""

import fcntl
import json
import time
import uuid
from pathlib import Path

from open_instruct.miles.errors import InputError


def poll_round(root, rank, count, member):
    """Require a new process identity from every replica before a new round.

    Serialize membership changes on WEKA. A restarted replica cannot consume
    old readiness/failure/completion files while waiting for peers to restart.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".rendezvous.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = root / "rendezvous.json"
        state = json.loads(path.read_text()) if path.exists() else {"count": count, "members": {}}
        if state["count"] != count:
            raise InputError("Replica count changed within a launch; submit a new launch identity")
        state["members"][str(rank)] = member
        previous = state.get("round", {})
        members = state["members"]
        if len(members) == count and all(
            members[str(i)] != previous.get("members", {}).get(str(i)) for i in range(count)
        ):
            state["round"] = {"id": uuid.uuid4().hex, "members": dict(members)}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(path)
        current = state.get("round", {})
        return current.get("id") if current.get("members", {}).get(str(rank)) == member else None


def join(root, rank, count, timeout):
    if count < 1 or rank not in range(count):
        raise InputError("Invalid replica membership")
    member = uuid.uuid4().hex
    deadline = time.monotonic() + timeout
    while True:
        attempt = poll_round(root, rank, count, member)
        if attempt is not None:
            return Path(root) / attempt
        if time.monotonic() >= deadline:
            raise TimeoutError("Waiting for fresh processes from every replica before restarting the cluster")
        time.sleep(1)
