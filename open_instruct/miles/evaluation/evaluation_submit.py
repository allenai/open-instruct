"""Create one background-evaluation experiment in an isolated Beaker API process.
The coordinator invokes this worker with a rendered specification and a deadline,
so an unresponsive submission cannot block training indefinitely. It returns the
experiment ID on success or a bounded diagnostic with environment credentials
redacted on failure; its own alarm remains active if the caller exits.
"""

import json
import os
import re
import signal
import sys

from beaker import Beaker
from beaker.exceptions import BeakerExperimentConflict, BeakerWorkloadNotFound


def diagnostic(error):
    """Keep useful API errors without persisting credentials echoed by dependencies."""
    message = str(error)
    for name, value in os.environ.items():
        if value and any(part in name.upper() for part in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", message)
    return {"error": type(error).__name__, "message": message[:2000]}


def create(client, path, workspace, name, *, reconcile):
    if not reconcile:
        return client.experiment.create(spec=path, name=name).experiment.id
    workspace_id = client.workspace.get(workspace).id
    qualified = f"{client.user.get().name}/{name}"

    def existing():
        workload = client.workload.get(qualified)
        if workload.experiment.workspace_id != workspace_id:
            raise ValueError(f"Experiment {qualified} exists in a different workspace")
        return workload.experiment.id

    try:
        return existing()
    except BeakerWorkloadNotFound:
        try:
            return client.experiment.create(spec=path, name=name).experiment.id
        except BeakerExperimentConflict:
            return existing()


def main():
    path, workspace, name, deadline, *options = sys.argv[1:]
    # The deadline survives trainer exit (the daemon caller may disappear first).
    signal.alarm(int(deadline))
    try:
        with Beaker.from_env(default_workspace=workspace, check_for_upgrades=False) as client:
            experiment = create(client, path, workspace, name, reconcile=options == ["--reconcile"])
            print(json.dumps({"id": experiment}))
    except Exception as error:
        print(json.dumps(diagnostic(error)), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
